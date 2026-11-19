"""CPU physics/accounting checks; these do not validate a trained policy or PhysX."""

import ast
import csv
from collections import UserDict, UserList
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "active_adaptation/envs/mdp/commands"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interaction_module = load_file("evaluation_interaction", COMMANDS / "evaluation_interaction.py")
utils = load_file("command_utils", COMMANDS / "utils.py")
admittance = load_file("admittance", COMMANDS / "admittance.py")
metrics = load_file("interaction_metrics", ROOT / "scripts/utils/interaction_metrics.py")


def native_methods():
    # Isolate production tensor methods from the Isaac Sim import/launcher machinery.
    namespace = dict(torch=torch, clamp_norm=utils.clamp_norm,
                     normalize=lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    tree = ast.parse((ROOT / "active_adaptation/utils/math.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("quat_apply", "quat_apply_inverse"):
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), "math.py", "exec"), namespace)
    tree = ast.parse((COMMANDS / "motion_tracking.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MotionTrackingCommand_impedance")
    names = ["force_apply", "project_pos_diff", "force_update_origin_and_target"]
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "motion_tracking.py", "exec"), namespace)
    return type("TensorCommand", (), {name: namespace[name] for name in names})


TensorCommand = native_methods()


def make_command(mode="spring", safe=10., mass=1., **overrides):
    c = TensorCommand()
    c.device = "cpu"
    c.num_envs = 1
    c.compliance = True
    c.force_apply_idx_asset = torch.arange(6)
    c.force_apply_idx_motion = torch.arange(6)
    names = ["left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_wrist_roll_link",
             "right_wrist_roll_link", "left_hand_mimic", "right_hand_mimic", "torso_link", "right_elbow_link"]
    quat = torch.tensor([1., 0., 0., 0.])
    data = SimpleNamespace(body_pos_w=torch.zeros(1, 8, 3), body_quat_w=quat.expand(1, 8, 4),
                           root_pos_w=torch.zeros(1, 3), root_quat_w=quat.expand(1, 4),
                           root_lin_vel_w=torch.zeros(1, 3), root_ang_vel_w=torch.zeros(1, 3),
                           body_lin_vel_w=torch.zeros(1, 8, 3), joint_pos=torch.zeros(1, 1),
                           applied_torque=torch.zeros(1, 1), soft_joint_pos_limits=torch.tensor([[[-1., 1.]]]))
    c.asset = SimpleNamespace(body_names=names, data=data, joint_names=["right_elbow_joint"])
    c.dataset = SimpleNamespace(body_names=names)
    c.env = SimpleNamespace(step_dt=.02, physics_dt=.005, scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    for name in ["force_applied_w", "force_applied_b", "force_expected_w", "force_expected_b",
                 "force_pos_delta", "force_origin_w", "force_dir_w", "ref_pos_b_prev", "force_keypoint_w",
                 "force_keypoint_b", "force_keypoint_w_prev", "force_keypoint_vel_w"]:
        setattr(c, name, torch.zeros(1, 6, 3))
    c.force_kp_scaled = torch.zeros(1, 6, 1)
    c.force_enable = torch.zeros(1, 6, 1, dtype=torch.bool)
    c.force_apply_buffer = torch.zeros(1, 8, 3)
    c.torque_apply_buffer = torch.zeros(1, 8, 3)
    c.force_safe_bounds = (5., 15.)
    c.force_safe_default = 10.
    c.force_safe_limit_tl = utils.TemporalLerp((1, 1), "cpu", clamp=(5., 15.))
    c.reward_keypoints_w = torch.zeros(1, 8, 3)
    c._motion = SimpleNamespace(body_pos_w=torch.zeros(1, 1, 8, 3), root_pos_w=torch.zeros(1, 1, 3))
    c.force_origin_tl = utils.TemporalLerp((1, 6, 3), "cpu")
    c.last_reset_env_ids = torch.tensor([0])
    c.max_force = 30.
    c.force_alpha = 1.
    c.torso_idx_asset = c.torso_idx_motion = 6
    c.t = torch.zeros(1, dtype=torch.int32)
    c.admit = admittance.AdmittanceMassChain(1, 6, .005, 1, "cpu", mass=.1, damping=2., vel_clip=4., acc_clip=1000.)
    config = dict(mode=mode, safe_force=safe, body="right_hand_mimic", direction=[1., 0., 0.],
                  force_vector=[0., 10., 0.], stiffness=100., offset=.2,
                  start=0., end=10., ramp=0., load_kg=mass)
    config.update(overrides)
    c.eval_interaction = interaction_module.EvaluationInteraction(c, config)
    c.cfg = SimpleNamespace(command=SimpleNamespace(dataset=SimpleNamespace(mem_paths=["synthetic"], fix_motion_id=0), motion_start_frame=0))
    c.env.cfg = c.cfg
    c.env.command_manager = c
    return c


class ControlledInteractionTests(unittest.TestCase):
    def test_pulse_has_baseline_ramps_and_release(self):
        pulse = interaction_module.pulse_scale
        actual = [pulse(t, 3, 12, 1) for t in [0, 3, 3.5, 4, 10, 11.5, 12, 20]]
        self.assertEqual(actual, [0, 0, .5, 1, 1, .5, 0, 0])

    def test_yielding_reduces_spring_force_without_clamping_to_safety_input(self):
        c = make_command(safe=5.)
        c.eval_interaction.prepare_targets()
        c.eval_interaction.apply(0)
        self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), 20., places=4)
        c.asset.data.body_pos_w[0, 5, 0] = .1
        c.eval_interaction.apply(1)
        self.assertAlmostEqual(float(c.force_applied_w[0, 5, 0]), 10., places=4)

    def test_admittance_targets_change_with_threshold(self):
        results = []
        for safe in [5., 15.]:
            c = make_command(safe=safe)
            for _ in range(200):
                c.eval_interaction.prepare_targets()
                c.last_reset_env_ids = None
            results.append((float(c.force_keypoint_w[0, 5, 0]), float(c.force_expected_w[0, 5, 0])))
        self.assertGreater(results[0][0], results[1][0] + .05)
        self.assertLess(results[0][1], results[1][1])

    def test_payload_is_mg_separate_from_spring_without_torso_compensation(self):
        for mass in [0., .5, 1., 1.5]:
            c = make_command(mode="load", mass=mass)
            c.eval_interaction.prepare_targets()
            for substep in range(4):
                c.eval_interaction.apply(substep)
                self.assertAlmostEqual(float(c.force_apply_buffer[0, 5, 2]), -mass*9.81, places=4)
                self.assertEqual(float(c.force_applied_w.abs().sum()), 0.)
                self.assertEqual(float(c.force_apply_buffer[0, 6].abs().sum()), 0.)
            self.assertAlmostEqual(c.eval_interaction.elapsed, .02)

    def test_release_removes_payload_and_baseline_applies_no_force(self):
        for mode in ["load", "force", "none"]:
            c = make_command(mode=mode)
            c.eval_interaction.elapsed = 11.
            c.eval_interaction.prepare_targets()
            c.eval_interaction.apply(0)
            self.assertEqual(float(c.force_apply_buffer.abs().sum()), 0.)

    def test_fixed_world_vector_matches_simulation_and_reference(self):
        c = make_command(mode="force", mass=1.0, safe=5., force_vector=[0., 12., -5.])
        c.eval_interaction.prepare_targets()
        expected = torch.tensor([0., 12., -5.])
        torch.testing.assert_close(c.force_expected_w[0, 5], expected)
        for substep in range(4):
            c.eval_interaction.apply(substep)
            torch.testing.assert_close(c.force_apply_buffer[0, 5], expected)
            torch.testing.assert_close(c.eval_interaction.total_w[0, 5], expected)
        self.assertAlmostEqual(float(c.eval_interaction.direction.norm()), 1.)

    def test_constant_force_changes_compliant_reference_with_safe_threshold(self):
        references = []
        for safe in [5., 15.]:
            c = make_command(mode="force", mass=1., safe=safe, force_vector=[0., 10., 0.])
            for _ in range(200):
                c.eval_interaction.prepare_targets()
                c.last_reset_env_ids = None
            references.append(float(c.force_keypoint_w[0, 5, 1]))
        self.assertGreater(references[0], references[1] + .05)

    def test_fixed_threshold_survives_reset(self):
        c = make_command(safe=5.)
        c.force_safe_limit_tl.set(None, end=15., total_steps=1)
        c.eval_interaction.reset(torch.tensor([0]))
        c.force_safe_limit_tl.update_time()
        self.assertEqual(float(c.force_safe_limit_tl.current[0, 0]), 5.)

    def test_recorder_separates_load_and_keeps_terminal_state(self):
        c = make_command(mode="load", mass=1.5)
        c.eval_interaction.prepare_targets()
        c.eval_interaction.apply(0)
        c.asset.data.body_pos_w[0, 5, 2] = -.1
        with tempfile.TemporaryDirectory() as directory:
            recorder = metrics.InteractionRecorder(
                c.env, directory,
                metadata={"seed": np.int64(4), "nested": UserDict({"force_vector": UserList([0., 10., 0.])})},
            )
            recorder.capture(0, terminated=True, truncated=False)
            recorder.save()
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertTrue(summary["terminated"])
            self.assertEqual(summary["protocol"]["nested"]["force_vector"], [0., 10., 0.])
            self.assertEqual(summary["protocol"]["seed"], 4)
            with (Path(directory) / "bodies.csv").open() as stream:
                row = next(row for row in csv.DictReader(stream) if row["body"] == "right_hand_mimic")
            self.assertEqual(float(row["spring_mag_N"]), 0.)
            self.assertAlmostEqual(float(row["load_mag_N"]), 14.715, places=4)
            self.assertAlmostEqual(float(row["actual_z_m"]), -.1, places=4)
            self.assertEqual(int(row["terminated"]), 1)

    def test_comparison_script_reads_exported_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, safe, label in [("none", 10., "baseline"), ("spring", 5., "safe5"), ("spring", 15., "safe15")]:
                c = make_command(mode=mode, safe=safe, start=.1, end=.3, ramp=.05)
                recorder = metrics.InteractionRecorder(c.env, root / label)
                for i in range(25):
                    c.eval_interaction.prepare_targets()
                    c.eval_interaction.apply(0)
                    c.last_reset_env_ids = None
                    recorder.capture(i, False, i == 24)
                recorder.save()
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/compare_interactions.py"), "--runs",
                 *[str(root / name) for name in ["baseline", "safe5", "safe15"]],
                 "--output", str(root / "comparison")], capture_output=True, text=True,
                env=dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib")),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ["comparison.png", "comparison.pdf", "comparison.csv"]:
                self.assertGreater((root / "comparison" / name).stat().st_size, 100)

    def test_motion_listing_uses_segment_ids_and_durations(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "motions"
            dataset.mkdir()
            (dataset / "meta_motion.json").write_text(json.dumps(dict(
                starts=[0, 100], ends=[100, 1100], info=dict(source=["short", "upper-body"]))))
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/list_motions.py"), "--dataset", "motions", "--min-seconds", "18"],
                env=dict(os.environ, MEMPATH=directory), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("motion-id=    1", result.stdout)
            self.assertIn("upper-body", result.stdout)
            self.assertNotIn("motion-id=    0", result.stdout)


if __name__ == "__main__":
    unittest.main()
