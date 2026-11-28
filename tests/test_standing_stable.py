"""CPU tensor checks for constant-load replay; no Isaac or policy validation."""

import ast
import csv
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile

import numpy as np
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
    exp_error = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_exp_error")
    exec(compile(ast.Module(body=[exp_error], type_ignores=[]), "standing_impedance.py", "exec"), namespace)
    methods = {}
    names = {"set_eval_force_schedule", "set_eval_stable_force", "set_eval_spring",
             "_is_stable_replay", "_stable_force_w", "_update_eval_schedule",
             "before_update", "step", "_current_hand_b", "_limit_net_wrench",
             "_bound_hand_reference", "_sample_offset", "_sample_force", "_moving",
             "_active_force_mean", "_heading_error", "body_yield_velocity",
             "body_rest_velocity", "body_yield_stall", "body_heading_tracking",
             "body_yaw_rate", "force_reward", "force_exd_penalty", "stand_lower_body",
             "yield_step_air_time", "yield_command", "_accumulate_tracking_metrics",
             "tracking_diagnostics"}
    names.add("_sample_active_mask")
    names.update({"_constant_mask", "_constant_load_w", "_physical_load_w",
                  "_update_body_reference", "_sample_constant_magnitude", "step_schedule",
                  "_update_mixed_schedule", "_update_schedule", "_balance_factor",
                  "body_velocity_error", "command", "_reward_value",
                  "force_target_tracking", "force_target_vel_tracking", "heading_command", "force_priv"})
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), "standing_impedance.py", "exec"), namespace)
            methods[node.name] = namespace[node.name]
    return type("TensorStanding", (), methods)


TensorStanding = tensor_command()


def force_helpers():
    namespace = dict(np=np, csv=csv)
    names = {"_force_reference_snapshot", "_force_snapshot", "_force_record_rows", "_write_force_csv",
             "_following_motion_fields", "_following_reference_snapshot"}
    tree = ast.parse((ROOT / "scripts/utils/helpers.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "helpers.py", "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


ForceHelpers = force_helpers()


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
        root_ang_vel_w=torch.zeros(1, 3), body_lin_vel_w=torch.zeros(1, 7, 3),
    )
    c.env = SimpleNamespace(step_dt=.02)
    c.force_apply_idx_asset = torch.arange(6)
    c.torso_idx_asset = 6
    for name in ["nominal_hand_b", "force_origin_b", "force_dir_b", "force_keypoint_b",
                 "force_keypoint_w", "force_keypoint_vel_w", "force_keypoint_w_prev",
                 "force_applied_w", "force_applied_b", "force_expected_w", "force_expected_b",
                 "force_pos_delta", "force_interact_b", "force_interact_w", "force_drive_b", "force_drive_w"]:
        setattr(c, name, torch.zeros(1, 6, 3))
    c.force_apply_buffer = torch.zeros(1, 7, 3)
    c.torque_apply_buffer = torch.zeros_like(c.force_apply_buffer)
    c.force_kp_nominal = torch.full((1, 6, 1), 100.)
    c.force_kp = c.force_kp_nominal.clone()
    c.force_offset = torch.full((1, 1, 1), .2)
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
    c.following_rewards = False
    c.threshold_following = False
    c._reward_reference = {}
    c._reward_reference_valid = torch.zeros(1, dtype=torch.bool)
    c.constant_load_probability = .5
    c.constant_excess_range = (2., 20.)
    c.load_curriculum_fraction = .25
    c.student_train = False
    c.load_curriculum_scale = 1.
    c.load_is_constant = torch.zeros(1, dtype=torch.bool)
    c.constant_force_w = torch.zeros(1, 6, 3)
    c.follow_input_force_w = torch.zeros(1, 3)
    c.follow_excess_force_w = torch.zeros(1, 3)
    c.load_age = torch.zeros(1, dtype=torch.long)
    c.load_ramp_steps = torch.ones(1, dtype=torch.long)
    c.load_hold_steps = torch.ones(1, dtype=torch.long)
    c.hand_reference_limit = None
    c.hand_offset_range = None
    c.hand_offset_random_prob = .5
    c.force_body_sampling = "independent"
    c.hand_force_indices = [4, 5]
    c.tracking_sigma, c.moving_speed_threshold, c.tracking_tolerance = .1, .05, .05
    c.heading_ref_w = torch.tensor([[1., 0., 0.]])
    c.metric_sums = torch.zeros(16)
    c.threshold_metric_sums = torch.zeros(6)
    c.admit = admittance.AdmittanceMassChain(1, 6, .005, 1, "cpu", mass=.1,
                                            damping=2., vel_clip=4., acc_clip=1000.)
    c.set_eval_force_schedule("stable", **schedule)
    c.set_eval_spring(body_names=bodies or ["right_hand_mimic"])
    c.set_eval_stable_force(magnitude, list(direction))
    c.env.command_manager = c
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

    def test_record_pairs_reference_with_applied_interval_at_onset_and_release(self):
        c = make_command(start_s=.04, end_s=.08)
        c.before_update()
        forces = []
        for i in range(6):
            reference = ForceHelpers._force_reference_snapshot(c.env)
            c.step(0)
            # Production environment prepares the next reference before logging.
            c.before_update()
            snap = ForceHelpers._force_snapshot(c.env, reference=reference)
            row = ForceHelpers._force_record_rows(snap, i, .02, .005)[5]
            self.assertAlmostEqual(row["applied_interact_error_N"], 0.)
            self.assertAlmostEqual(row["applied_expected_error_N"], 0.)
            self.assertEqual(row["applied_sample_time_s"], row["interact_reference_time_s"])
            forces.append(row["f_applied_x_N"])
        self.assertEqual(forces, [0., 0., 10., 10., 0., 0.])

    def test_csv_records_vector_error_and_keeps_legacy_columns(self):
        snap = dict(names=["right_hand_mimic"], applied=np.array([[10., 0., 0.]]),
                    expected=np.array([[-10., 0., 0.]]), interact=np.array([[0., 10., 0.]]), limit=10.)
        rows = ForceHelpers._force_record_rows(snap, 0, .02, .005)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forces.csv"
            ForceHelpers._write_force_csv(path, rows)
            with path.open() as stream:
                saved = list(csv.DictReader(stream))[0]
        self.assertEqual(float(saved["f_mag_N"]), float(saved["f_applied_mag_N"]))
        self.assertEqual(float(saved["f_expected_mag_N"]), 10.)
        self.assertEqual(float(saved["f_interact_y_N"]), 10.)
        # Equal magnitudes alone would miss a reversed force vector.
        self.assertEqual(float(saved["applied_expected_error_N"]), 20.)
        self.assertAlmostEqual(float(saved["applied_interact_error_N"]), math.sqrt(200))

    def test_virtual_integrator_force_is_distinct_from_final_expected_force(self):
        c = make_command()
        c.set_eval_force_schedule("spring")
        c.force_origin_b[0, 5, 0] = .2
        c.before_update()
        self.assertGreater(float(c.force_interact_w[0, 5, 0]),
                           float(c.force_expected_w[0, 5, 0]))


if __name__ == "__main__":
    unittest.main()
