"""CPU tensor checks for constant-load replay; no Isaac or policy validation."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "active_adaptation/envs/mdp/commands"
spec = importlib.util.spec_from_file_location("standing_admittance", COMMANDS / "admittance.py")
admittance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admittance)


def tensor_command():
    """Load the production methods without starting an Isaac application."""
    namespace = dict(torch=torch, math=math, clamp_norm=admittance.clamp_norm)
    tree = ast.parse((ROOT / "active_adaptation/utils/math.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"quat_apply", "quat_apply_inverse"}:
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), "math.py", "exec"), namespace)
    tree = ast.parse((COMMANDS / "standing_impedance.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StandingImpedanceCommand")
    methods = {}
    names = {"set_eval_force_schedule", "set_eval_stable_force", "set_eval_spring",
             "_is_stable_replay", "_stable_force_w", "_update_eval_schedule",
             "before_update", "step", "_current_hand_b", "_limit_net_wrench"}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "standing_impedance.py", "exec"), namespace)
            methods[node.name] = namespace[node.name]
    return type("TensorStanding", (), methods)


TensorStanding = tensor_command()


def make_command(magnitude=10., direction=(1., 0., 0.), bodies=None, **schedule):
    c = TensorStanding()
    c.num_envs, c.num_force_bodies, c.device = 1, 6, "cpu"
    names = ["left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_wrist_roll_link",
             "right_wrist_roll_link", "left_hand_mimic", "right_hand_mimic", "pelvis"]
    c.asset = SimpleNamespace(body_names=names, has_external_wrench=False)
    c.asset.data = SimpleNamespace(
        root_pos_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_lin_vel_w=torch.zeros(1, 3), body_pos_w=torch.zeros(1, 7, 3),
        body_quat_w=torch.tensor([1., 0., 0., 0.]).expand(1, 7, 4).clone(),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]),
    )
    c.env = SimpleNamespace(step_dt=.02)
    c.force_apply_idx_asset = torch.arange(6)
    c.torso_idx_asset = 6
    for name in ["nominal_hand_b", "force_origin_b", "force_dir_b", "force_keypoint_b",
                 "force_keypoint_w", "force_keypoint_vel_w", "force_keypoint_w_prev",
                 "force_applied_w", "force_applied_b", "force_expected_w", "force_expected_b",
                 "force_pos_delta"]:
        setattr(c, name, torch.zeros(1, 6, 3))
    c.force_apply_buffer = torch.zeros(1, 7, 3)
    c.torque_apply_buffer = torch.zeros_like(c.force_apply_buffer)
    c.force_kp_nominal = torch.full((1, 6, 1), 100.)
    c.force_kp = c.force_kp_nominal.clone()
    c.force_enable = torch.zeros(1, 6, 1, dtype=torch.bool)
    c.force_scale = torch.zeros(1, 1)
    c.eval_force_body_mask = torch.ones(6, dtype=torch.bool)
    c.eval_force_step = torch.zeros(1, dtype=torch.long)
    c.nominal_initialized = torch.ones(1, dtype=torch.bool)
    c.safe_force_limit = torch.full((1, 1), 10.)
    c.root_vel_ref_w = torch.zeros(1, 3)
    c.root_pos_ref_w = torch.zeros(1, 3)
    c.root_height_ref = torch.zeros(1, 1)
    c._cum_error = torch.zeros(1, 3)
    c.compliance = True
    c.max_force, c.hand_offset = 60., .2
    c.net_force_limit, c.net_torque_limit = torch.tensor(60.), torch.tensor(35.)
    c.body_mass, c.body_damping = 35., 45.
    c.body_acc_limit, c.body_vel_limit = 2., .8
    c.admit = admittance.AdmittanceMassChain(1, 6, .005, 1, "cpu", mass=.1,
                                            damping=2., vel_clip=4., acc_clip=1000.)
    c.set_eval_force_schedule("stable", **schedule)
    c.set_eval_spring(body_names=bodies or ["right_hand_mimic"])
    c.set_eval_stable_force(magnitude, list(direction))
    return c


class StableReplayTests(unittest.TestCase):
    def test_world_force_ignores_pose_displacement_and_spring_caps(self):
        c = make_command(50., (3., 4., 0.))
        c.eval_max_force = .1
        c.net_force_limit = torch.tensor(.1)
        for angle in [0., math.pi / 2, math.pi]:
            c.asset.data.root_quat_w[:] = torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
            c.asset.data.body_pos_w.uniform_(-3., 3.)
            c.before_update()
            for substep in range(4):
                c.step(substep)
                torch.testing.assert_close(c.force_applied_w[0, 5], torch.tensor([30., 40., 0.]))
                torch.testing.assert_close(c.force_apply_buffer[0, :5], torch.zeros(5, 3))
                torch.testing.assert_close(c.force_apply_buffer[0, 6], torch.zeros(3))
                torch.testing.assert_close(c.torque_apply_buffer, torch.zeros_like(c.torque_apply_buffer))
            torch.testing.assert_close(c.force_expected_w, c.force_applied_w)

    def test_two_selected_bodies_receive_magnitude_each(self):
        c = make_command(5., bodies=["left_hand_mimic", "right_hand_mimic"])
        c.before_update()
        c.step(0)
        torch.testing.assert_close(c.force_applied_w[0, 4:, 0], torch.tensor([5., 5.]))
        self.assertEqual(float(c.force_apply_buffer.sum()), 10.)

    def test_window_ramps_and_release(self):
        c = make_command(start_s=2., end_s=6., ramp_s=1.)
        for time_s, scale in [(0., 0.), (2., 0.), (2.5, .5), (3., 1.),
                              (5., 1.), (5.5, .5), (6., 0.), (7., 0.)]:
            c.eval_force_step.fill_(round(time_s / .02))
            c.before_update()
            c.step(0)
            self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), 10 * scale, places=4)

    def test_open_ended_step_stays_constant(self):
        c = make_command(start_s=2.)
        for time_s, force in [(0., 0.), (1., 0.), (2., 10.), (20., 10.)]:
            c.eval_force_step.fill_(round(time_s / .02))
            c.before_update()
            c.step(0)
            self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), force)

    def test_reference_moves_and_body_reference_approaches_force_over_damping(self):
        c = make_command()
        for _ in range(500):
            c.before_update()
            c.step(0)
        self.assertAlmostEqual(float(c.admit.x[0, 0, 5, 0]), .05, places=3)
        self.assertAlmostEqual(float(c.root_vel_ref_w[0, 0]), 10 / 45, places=4)
        torch.testing.assert_close(c.force_expected_w[0, 5], torch.tensor([10., 0., 0.]))

    def test_spring_still_depends_on_position(self):
        c = make_command()
        c.set_eval_force_schedule("spring")
        c.force_origin_b[0, 5, 0] = .2
        c.before_update()
        c.step(0)
        self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), 20.)
        c.asset.data.body_pos_w[0, 5, 0] = .1
        c.step(1)
        self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), 10.)

    def test_invalid_force_parameters_rejected(self):
        c = make_command()
        for magnitude, direction in [(float("nan"), [1., 0., 0.]), (-1., [1., 0., 0.]),
                                     (1., [0., 0., 0.]), (1., [float("inf"), 0., 0.])]:
            with self.assertRaises(ValueError):
                c.set_eval_stable_force(magnitude, direction)


if __name__ == "__main__":
    unittest.main()
