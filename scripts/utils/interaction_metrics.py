"""Per-step observations and visible world-space annotations for controlled evaluation."""

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence

import numpy as np

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def _json_safe(value):
    if OmegaConf is not None and OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return _json_safe(tensor.item() if tensor.numel() == 1 else tensor.tolist())
    raise TypeError(f"Unsupported value in experiment summary: {type(value).__name__}")


def _array(tensor):
    return tensor.detach().float().cpu().numpy()


class InteractionRecorder:
    def __init__(self, base_env, directory, metadata=None):
        self.env = base_env
        self.command = base_env.command_manager
        self.interaction = self.command.eval_interaction
        self.directory = directory
        self.metadata = metadata or {}
        self.rows = []
        self.joint_rows = []
        self.steps = []
        self.mapping = []
        c = self.command
        force_ids = c.force_apply_idx_asset.tolist()
        for ai, name in enumerate(c.asset.body_names):
            if name.endswith(("shoulder_yaw_link", "elbow_link", "wrist_roll_link", "hand_mimic")) and name in c.dataset.body_names:
                self.mapping.append((ai, c.dataset.body_names.index(name), force_ids.index(ai) if ai in force_ids else None, name))

    def capture(self, step, terminated, truncated):
        c, interaction = self.command, self.interaction
        p = interaction.config
        time_s = interaction.elapsed
        # Physics sampled the force over [time_s-dt, time_s); position is at time_s.
        sample_time = time_s - self.env.step_dt
        phase = "before" if sample_time < p["start"] else "during" if sample_time < p["end"] else "after"
        root = _array(c.asset.data.root_pos_w[0])
        raw_root = _array(c._motion.root_pos_w[0, 0] + self.env.scene.env_origins[0])
        q = _array(c.asset.data.body_quat_w[0, c.torso_idx_asset])
        tilt = math.degrees(math.acos(float(np.clip(1 - 2*(q[1]**2 + q[2]**2), -1, 1))))
        root_error = float(np.linalg.norm(root - raw_root))
        torso = _array(c.asset.data.body_pos_w[0, c.torso_idx_asset])
        torso_ref = _array(c._motion.body_pos_w[0, 0, c.torso_idx_motion] + self.env.scene.env_origins[0])
        limit = float(c.force_safe_limit_tl.current[0, 0])
        direction = _array(interaction.direction)
        common = dict(step=step, time_s=time_s, force_sample_time_s=sample_time, phase=phase,
                      pulse_scale=interaction.applied_scale, safe_limit_N=limit,
                      root_error_m=root_error, torso_error_m=float(np.linalg.norm(torso-torso_ref)),
                      torso_tilt_deg=tilt, terminated=int(terminated), truncated=int(truncated),
                      motion_frame=int(c.t[0]), load_kg=p["load_kg"] if p["mode"] == "load" else 0.)
        for ai, mi, fi, name in self.mapping:
            actual = _array(c.asset.data.body_pos_w[0, ai])
            reference = _array(c._motion.body_pos_w[0, 0, mi] + self.env.scene.env_origins[0])
            aligned = _array(c.reward_keypoints_w[0, mi])
            target = _array(c.force_keypoint_w[0, fi]) if fi is not None else aligned
            spring = _array(c.force_applied_w[0, fi]) if fi is not None else np.zeros(3)
            load = _array(interaction.load_w[0, fi]) if fi is not None else np.zeros(3)
            expected = _array(c.force_expected_w[0, fi]) if fi is not None else np.zeros(3)
            row = dict(common, body=name, spring_mag_N=float(np.linalg.norm(spring)),
                       load_mag_N=float(np.linalg.norm(load)), total_mag_N=float(np.linalg.norm(spring+load)),
                       external_force_mag_N=float(np.linalg.norm(load)),
                       expected_spring_mag_N=float(np.linalg.norm(expected)),
                       spring_exceeds_limit=int(np.linalg.norm(spring) > limit),
                       tracking_error_m=float(np.linalg.norm(actual-reference)),
                       aligned_tracking_error_m=float(np.linalg.norm(actual-aligned)),
                       compliant_target_error_m=float(np.linalg.norm(actual-target)),
                       yielding_m=float(np.dot(actual-reference, direction)),
                       reference_yielding_m=float(np.dot(target-reference, direction)),
                       actual_minus_target_along_force_m=float(np.dot(actual-target, direction)))
            for label, values in [("actual", actual), ("reference", reference), ("aligned_reference", aligned),
                                  ("compliant_target", target), ("spring", spring), ("load", load), ("total", spring+load),
                                  ("root", root), ("root_reference", raw_root), ("torso", torso), ("torso_reference", torso_ref)]:
                unit = "N" if label in ("spring", "load", "total") else "m"
                row.update({f"{label}_{axis}_{unit}": float(value) for axis, value in zip("xyz", values)})
            self.rows.append(row)
        arm_torque = []
        for ji, name in enumerate(c.asset.joint_names):
            if any(part in name for part in ("shoulder", "elbow", "wrist")):
                pos = float(c.asset.data.joint_pos[0, ji])
                lo, hi = _array(c.asset.data.soft_joint_pos_limits[0, ji])
                torque = float(c.asset.data.applied_torque[0, ji])
                arm_torque.append(abs(torque))
                self.joint_rows.append(dict(step=step, time_s=time_s, phase=phase, joint=name,
                                           position_rad=pos, applied_torque_Nm=torque,
                                           outside_soft_limit=int(pos < lo or pos > hi)))
        self.steps.append(dict(common, arm_peak_torque_Nm=max(arm_torque, default=0.)))

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        self._write_csv(os.path.join(self.directory, "bodies.csv"), self.rows)
        self._write_csv(os.path.join(self.directory, "joints.csv"), self.joint_rows)
        self._write_csv(os.path.join(self.directory, "steps.csv"), self.steps)
        selected = [row for row in self.rows if row["body"] == self.interaction.config["body"]]
        summary = dict(interaction=self.interaction.config,
                       protocol=dict(dataset=list(self.env.cfg.command.dataset.mem_paths),
                                     motion_id=self.env.cfg.command.dataset.fix_motion_id,
                                     motion_start_frame=self.env.cfg.command.motion_start_frame,
                                     step_dt=self.env.step_dt, **self.metadata),
                       recorded_duration_s=self.steps[-1]["time_s"] if self.steps else 0.,
                       terminated=bool(self.steps and self.steps[-1]["terminated"]),
                       truncated=bool(self.steps and self.steps[-1]["truncated"]),
                       phases={})
        for phase in ["before", "during", "after"]:
            rows = [row for row in selected if row["phase"] == phase]
            if not rows:
                continue
            summary["phases"][phase] = dict(
                samples=len(rows),
                peak_spring_N=max(row["spring_mag_N"] for row in rows),
                peak_external_force_N=max(row["external_force_mag_N"] for row in rows),
                peak_total_force_N=max(row["total_mag_N"] for row in rows),
                spring_exceed_fraction=float(np.mean([row["spring_exceeds_limit"] for row in rows])),
                spring_exceed_duration_s=sum(row["spring_exceeds_limit"] for row in rows)*self.env.step_dt,
                tracking_rmse_m=float(np.sqrt(np.mean([row["tracking_error_m"]**2 for row in rows]))),
                mean_yielding_m=float(np.mean([row["yielding_m"] for row in rows])),
                root_error_max_m=max(row["root_error_m"] for row in rows),
                torso_tilt_max_deg=max(row["torso_tilt_deg"] for row in rows),
            )
        summary["outside_arm_soft_limits_fraction"] = float(np.mean([r["outside_soft_limit"] for r in self.joint_rows])) if self.joint_rows else None
        summary["notes"] = [
            "spring/load are commanded external PhysX forces, not measured contact or joint reaction forces.",
            "load is a downward mg force only; no bucket mesh, inertia or pendulum dynamics.",
            "Safety is a policy conditioning input, not a hard clamp on external force.",
            "A record stops at the first termination/truncation; no auto-reset episodes are concatenated.",
            "No torso counter-wrench compensation is used in controlled experiments.",
            "Position is sampled at time_s; force is the last physics-substep force for the interval starting at force_sample_time_s.",
        ]
        with open(os.path.join(self.directory, "summary.json"), "w") as stream:
            json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)
        print(f"[Experiment] Saved bodies.csv, joints.csv, steps.csv, summary.json to {self.directory}")

    def overlay(self, frame):
        """Project 3D force arrows and motion targets into the recorded RGB frame."""
        from PIL import Image, ImageDraw
        from pxr import Gf, UsdGeom
        import omni.usd

        c, interaction = self.command, self.interaction
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        h, w = frame.shape[:2]
        camera = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/OmniverseKit_Persp"))
        frustum = camera.GetCamera().frustum
        view = frustum.ComputeViewMatrix()
        matrix = view * frustum.ComputeProjectionMatrix()

        def project(point):
            p = Gf.Vec3d(*[float(v) for v in point])
            if view.Transform(p)[2] >= 0:
                return None
            ndc = matrix.Transform(p)
            if not all(math.isfinite(v) for v in ndc) or max(abs(ndc[0]), abs(ndc[1])) > 3:
                return None
            return ((ndc[0]+1)*w/2, (1-ndc[1])*h/2)

        def line(a, b, color, arrow=False):
            a, b = project(a), project(b)
            if a is None or b is None:
                return
            draw.line([a, b], fill=color, width=4)
            if arrow:
                v = np.array(a)-np.array(b)
                length = np.linalg.norm(v)
                if length < 2:
                    return
                v = v / length * 12
                perp = np.array([-v[1], v[0]]) * 0.45
                draw.polygon([b, tuple(np.array(b)+v+perp), tuple(np.array(b)+v-perp)], fill=color)

        # Red original-motion arm and green compliant target give a spatial reference.
        side = interaction.config["body"].split("_")[0]
        chain = [f"{side}_shoulder_yaw_link", f"{side}_elbow_link", f"{side}_wrist_roll_link", f"{side}_hand_mimic"]
        points = []
        for name in chain:
            if name in c.dataset.body_names:
                mi = c.dataset.body_names.index(name)
                points.append(_array(c._motion.body_pos_w[0, 0, mi] + self.env.scene.env_origins[0]))
        for a, b in zip(points, points[1:]):
            line(a, b, (255, 100, 100))
        ai = int(c.force_apply_idx_asset[interaction.body])
        actual = _array(c.asset.data.body_pos_w[0, ai])
        target = _array(c.force_keypoint_w[0, interaction.body])
        line(actual, target, (90, 240, 110))
        for force, color in [(c.force_applied_w[0, interaction.body], (80, 180, 255)),
                             (interaction.load_w[0, interaction.body], (255, 220, 60))]:
            force = _array(force)
            line(actual, actual+0.015*force, color, arrow=True)
        phase = self.steps[-1]["phase"] if self.steps else "before"
        text = (f"{interaction.config['mode']} | {phase} | t={interaction.elapsed:.2f}s | "
                f"safe={float(c.force_safe_limit_tl.current[0, 0]):.1f}N | "
                f"external={np.linalg.norm(_array(interaction.load_w[0, interaction.body])):.1f}N\n"
                "RED: original motion arm | GREEN: compliant target | BLUE: spring | YELLOW: fixed external force (0.015m/N)")
        draw.rectangle((6, h-55, w-6, h-5), fill=(15, 15, 15))
        draw.text((14, h-50), text, fill=(255, 255, 255))
        return np.asarray(image)
