"""Stationary whole-body compliance task.

This command keeps the useful part of GentleHumanoid's interaction model
(spring force -> hand admittance target) without requiring a motion dataset.
It also integrates a separate, bounded body-level velocity reference from the
net interaction force.  The latter is a task reference, not a force sensor.
"""

import math
import re

import torch

from active_adaptation.envs.mdp import observation, reward
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import quat_apply, quat_apply_inverse
from .base import Command
from .admittance import AdmittanceMassChain, clamp_norm


def _exp_error(error: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-error / max(float(sigma), 1e-6))


class StandingImpedanceCommand(Command):
    """Default-pose standing with compliant hand interaction and body yielding."""

    def __init__(
        self,
        env,
        max_force: float = 60.0,
        net_force_limit: float = 60.0,
        net_torque_limit: float = 35.0,
        compliance: bool = True,
        hand_offset: float = 0.20,
        hand_offset_range: tuple[float, float] | None = None,
        hand_offset_random_prob: float = 0.5,
        force_body_sampling: str = "independent",
        hand_stiffness_range: tuple[float, float] = (5.0, 250.0),
        safe_force_range: tuple[float, float] = (5.0, 15.0),
        force_time_range: tuple[int, int] = (100, 500),
        ramp_time_range: tuple[int, int] = (30, 80),
        body_mass: float = 35.0,
        body_damping: float = 45.0,
        body_vel_limit: float = 0.8,
        body_acc_limit: float = 2.0,
        drive_mass: float = 0.1,
        drive_damping: float = 2.0,
        drive_vel_limit: float = 4.0,
        drive_acc_limit: float = 1000.0,
        following_rewards: bool = False,
        hand_reference_limit: float | None = None,
        tracking_sigma: float = 0.10,
        moving_speed_threshold: float = 0.05,
        tracking_tolerance: float = 0.05,
        threshold_following: bool = False,
        constant_load_probability: float = 0.5,
        constant_excess_range: tuple[float, float] = (2.0, 20.0),
        load_curriculum_fraction: float = 0.25,
        student_train: bool = False,
        body_patterns: list[str] = [".*shoulder_yaw_link", ".*wrist_roll_link", ".*hand_mimic"],
        lower_joint_pattern: str = ".*(waist|hip|knee|ankle).*",
        **kwargs,
    ):
        super().__init__(env)

        self.max_force = float(max_force)
        self.net_force_limit = torch.as_tensor(net_force_limit, device=self.device)
        self.net_torque_limit = torch.as_tensor(net_torque_limit, device=self.device)
        self.compliance = bool(compliance)
        self.hand_offset = float(hand_offset)
        self.hand_offset_range = None if hand_offset_range is None else tuple(float(x) for x in hand_offset_range)
        self.hand_offset_random_prob = float(hand_offset_random_prob)
        if self.hand_offset_range is not None and (
            len(self.hand_offset_range) != 2
            or not all(math.isfinite(x) for x in self.hand_offset_range)
            or not 0.0 <= self.hand_offset_range[0] <= self.hand_offset_range[1]
        ):
            raise ValueError("hand_offset_range must be two finite, ordered non-negative distances")
        if not 0.0 <= self.hand_offset_random_prob <= 1.0:
            raise ValueError("hand_offset_random_prob must be between zero and one")
        self.following_rewards = bool(following_rewards)
        # Opt in through the new task config; old checkpoint configs keep their
        # original observations, reward semantics and replay dynamics.
        self.threshold_following = bool(threshold_following)
        self.constant_load_probability = float(constant_load_probability)
        self.constant_excess_range = tuple(float(x) for x in constant_excess_range)
        self.load_curriculum_fraction = float(load_curriculum_fraction)
        self.student_train = bool(student_train)
        self.load_curriculum_scale = 1.0 if student_train else 0.35
        if not 0 <= self.constant_load_probability <= 1:
            raise ValueError("constant_load_probability must be in [0, 1]")
        if (len(self.constant_excess_range) != 2
                or not all(math.isfinite(x) for x in self.constant_excess_range)
                or not 0 <= self.constant_excess_range[0] <= self.constant_excess_range[1]):
            raise ValueError("constant_excess_range must be finite, ordered and non-negative")
        if not math.isfinite(self.load_curriculum_fraction) or self.load_curriculum_fraction <= 0:
            raise ValueError("load_curriculum_fraction must be positive")
        if force_body_sampling not in {"independent", "balanced_hands"}:
            raise ValueError("Unknown force_body_sampling mode")
        self.force_body_sampling = force_body_sampling
        self.hand_reference_limit = hand_reference_limit
        if hand_reference_limit is not None and (not math.isfinite(hand_reference_limit) or hand_reference_limit <= 0):
            raise ValueError("hand_reference_limit must be finite and positive")
        self.tracking_sigma = float(tracking_sigma)
        self.moving_speed_threshold = float(moving_speed_threshold)
        self.tracking_tolerance = float(tracking_tolerance)
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.tracking_sigma, self.moving_speed_threshold, self.tracking_tolerance)):
            raise ValueError("Tracking scales and thresholds must be finite and positive")
        self.hand_stiffness_range = tuple(float(x) for x in hand_stiffness_range)
        self.safe_force_range = tuple(float(x) for x in safe_force_range)
        if not 5 <= self.safe_force_range[0] <= self.safe_force_range[1] <= 15:
            raise ValueError("safe_force_range must stay within 5..15 N")
        self.force_time_range = tuple(int(x) for x in force_time_range)
        self.ramp_time_range = tuple(int(x) for x in ramp_time_range)
        self.body_mass = float(body_mass)
        self.body_damping = float(body_damping)
        self.body_vel_limit = float(body_vel_limit)
        self.body_acc_limit = float(body_acc_limit)

        body_ids = [
            i for i, name in enumerate(self.asset.body_names)
            if any(re.match(pattern, name) for pattern in body_patterns)
        ]
        if not body_ids:
            raise ValueError(f"No interaction bodies matched {body_patterns}")
        self.force_apply_idx_asset = torch.tensor(body_ids, device=self.device, dtype=torch.long)
        self.num_force_bodies = len(body_ids)
        self.hand_force_indices = [j for j, i in enumerate(body_ids) if "hand_mimic" in self.asset.body_names[i]]
        if force_body_sampling == "balanced_hands" and len(self.hand_force_indices) != 2:
            raise ValueError("balanced_hands sampling requires left and right hand bodies")
        self.torso_idx_asset = next(
            (i for i, name in enumerate(self.asset.body_names) if re.match(".*torso.*|.*pelvis.*", name)),
            0,
        )
        self.lower_joint_ids = torch.tensor(
            [i for i, name in enumerate(self.asset.joint_names) if re.match(lower_joint_pattern, name)],
            device=self.device,
            dtype=torch.long,
        )

        shape = (self.num_envs, self.num_force_bodies, 3)
        self.nominal_hand_b = torch.zeros(shape, device=self.device)
        self.nominal_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_origin_b = torch.zeros(shape, device=self.device)
        self.force_dir_b = torch.zeros(shape, device=self.device)
        # Share the sampled distance across body points in an environment.
        self.force_offset = torch.full((self.num_envs, 1, 1), self.hand_offset, device=self.device)
        self.force_kp_nominal = torch.zeros(self.num_envs, self.num_force_bodies, 1, device=self.device)
        self.force_kp = torch.zeros(self.num_envs, self.num_force_bodies, 1, device=self.device)
        self.force_enable = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.bool, device=self.device)
        self.force_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramp_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramp_total = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramping_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_scale = torch.zeros(self.num_envs, 1, device=self.device)
        self.safe_force_limit = torch.zeros(self.num_envs, 1, device=self.device)
        self.load_is_constant = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.constant_force_w = torch.zeros(shape, device=self.device)
        self.load_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.load_ramp_steps = torch.ones_like(self.load_age)
        self.load_hold_steps = torch.ones_like(self.load_age)

        self.force_keypoint_b = torch.zeros(shape, device=self.device)
        self.force_keypoint_w = torch.zeros(shape, device=self.device)
        self.force_keypoint_vel_w = torch.zeros(shape, device=self.device)
        self.force_keypoint_w_prev = torch.zeros(shape, device=self.device)
        self.force_applied_w = torch.zeros(shape, device=self.device)
        self.force_applied_b = torch.zeros(shape, device=self.device)
        self.force_expected_w = torch.zeros(shape, device=self.device)
        self.force_expected_b = torch.zeros(shape, device=self.device)
        # Last external force actually used by the virtual integrator, before
        # its final substep. Unlike force_expected, this is not recomputed at
        # the final reference position. Populated only for explicit replay.
        self.force_interact_b = torch.zeros(shape, device=self.device)
        self.force_interact_w = torch.zeros(shape, device=self.device)
        self.force_drive_b = torch.zeros(shape, device=self.device)
        self.force_drive_w = torch.zeros(shape, device=self.device)
        self.force_pos_delta = torch.zeros(shape, device=self.device)
        self.force_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, device=self.device)
        self.torque_apply_buffer = torch.zeros_like(self.force_apply_buffer)
        self.force_apply_world = True

        self.root_vel_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Known physical load at the START of the next control interval. This
        # must not be confused with the spring evaluated at the virtual hand.
        self.follow_input_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.follow_excess_force_w = torch.zeros_like(self.follow_input_force_w)
        self.root_pos_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_height_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.heading_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_ref_w[:, 0] = 1.0
        self.metric_sums = torch.zeros(16, device=self.device)
        self.threshold_metric_sums = torch.zeros(6, device=self.device)
        self.contact_sensor = None
        self.foot_asset_ids = []
        self.foot_sensor_ids = []
        if self.following_rewards:
            self.contact_sensor = self.env.scene["contact_forces"]
            self.foot_asset_ids, _ = self.asset.find_bodies(".*ankle_roll_link")
            self.foot_sensor_ids, _ = self.contact_sensor.find_bodies(".*ankle_roll_link")
            if len(self.foot_asset_ids) != 2 or len(self.foot_sensor_ids) != 2:
                raise ValueError("Following rewards require two ankle contact bodies")
        self.joint_pos_boot_protect = self.init_joint_pos.clone()
        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._reward_reference = {}
        self._reward_reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.skip_ref = True
        # Optional deterministic schedule used only by evaluation replay.
        self.eval_force_schedule = None
        self.eval_force_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_spring_k = None
        self.eval_spring_offset = None
        self.eval_max_force = None
        self.eval_safe_force = None
        self.eval_direction_b = None
        self.eval_stable_magnitude = None
        self.eval_stable_direction_w = None
        self.eval_force_body_mask = torch.ones(
            self.num_force_bodies, dtype=torch.bool, device=self.device
        )

        self.admit = AdmittanceMassChain(
            num_envs=self.num_envs,
            num_points=self.num_force_bodies,
            dt=self.env.physics_dt,
            mixed_loop_steps=1,
            device=self.device,
            mass=drive_mass,
            damping=drive_damping,
            vel_clip=drive_vel_limit,
            acc_clip=drive_acc_limit,
        )

    def _current_hand_b(self) -> torch.Tensor:
        root_pos = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        hand_pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        return quat_apply_inverse(root_quat, hand_pos - root_pos)

    def _sample_force(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        if self.eval_direction_b is None:
            direction = torch.randn(n, 1, 3, device=self.device)
            direction[..., 2] *= 0.35
            direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            direction = self.eval_direction_b.reshape(1, 1, 3).expand(n, 1, 3)
        direction = direction.expand(-1, self.num_force_bodies, -1)
        self.force_dir_b[env_ids] = direction
        self.force_offset[env_ids] = self._sample_offset(n)
        self.force_origin_b[env_ids] = self.nominal_hand_b[env_ids] + direction * self.force_offset[env_ids]
        if self.eval_spring_k is None:
            self.force_kp[env_ids] = torch.empty(n, self.num_force_bodies, 1, device=self.device).uniform_(*self.hand_stiffness_range)
        else:
            self.force_kp[env_ids] = self.eval_spring_k
        # Mask stiffness at reset as well, before the first physics step.
        self.force_kp[env_ids] *= self.eval_force_body_mask.reshape(1, -1, 1)
        self.force_kp_nominal[env_ids] = self.force_kp[env_ids]
        self.force_enable[env_ids] = self._sample_active_mask(n)
        self.force_timer[env_ids] = torch.randint(
            self.force_time_range[0], self.force_time_range[1] + 1, (n,),
            device=self.device, dtype=self.force_timer.dtype,
        )
        self.force_scale[env_ids] = 1.0
        if self.eval_safe_force is None:
            self.safe_force_limit[env_ids] = torch.empty(n, 1, device=self.device).uniform_(*self.safe_force_range)
        else:
            self.safe_force_limit[env_ids] = self.eval_safe_force
        if self.threshold_following:
            self.load_is_constant[env_ids] = torch.rand(n, device=self.device) < self.constant_load_probability
            self.load_age[env_ids] = 0
            self.load_ramp_steps[env_ids] = torch.randint(
                self.ramp_time_range[0], self.ramp_time_range[1] + 1, (n,), device=self.device)
            self.load_hold_steps[env_ids] = self.force_timer[env_ids].to(self.load_hold_steps.dtype)
            # Stratify the TOTAL planar load around safe, independent of the
            # number of active links: 1/3 below, 1/3 near, 1/3 clearly above.
            total = self._sample_constant_magnitude(self.safe_force_limit[env_ids])
            planar_b = direction[:, 0].clone()
            planar_b[:, 2] = 0.0
            planar_b /= planar_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            planar_w = quat_apply(self.asset.data.root_quat_w[env_ids], planar_b)
            planar_w[:, 2] = 0.0
            planar_w /= planar_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            mask = self.force_enable[env_ids].to(total.dtype)
            count = mask.sum(dim=1).clamp_min(1.0)
            self.constant_force_w[env_ids] = planar_w.unsqueeze(1) * (total / count).unsqueeze(1) * mask
            # Mixed training starts unloaded and ramps in, including at reset.
            self.force_scale[env_ids] = 0.0

    def _sample_constant_magnitude(self, safe):
        regime = torch.rand_like(safe)
        u = torch.rand_like(safe)
        below = safe * (0.2 + 0.7 * u)
        near = safe * (0.9 + 0.2 * u)
        lo, hi = self.constant_excess_range
        above = safe + (lo + (hi - lo) * u) * self.load_curriculum_scale
        return torch.minimum(torch.where(regime < 1/3, below,
                             torch.where(regime < 2/3, near, above)), self.net_force_limit)

    def step_schedule(self, progress: float):
        if self.student_train:
            self.load_curriculum_scale = 1.0
        else:
            fraction = min(max(progress / self.load_curriculum_fraction, 0.0), 1.0)
            self.load_curriculum_scale = 0.35 + 0.65 * fraction

    def _sample_offset(self, n: int):
        offset = torch.full((n, 1, 1), self.hand_offset, device=self.device)
        if self.eval_spring_offset is not None:
            return offset.fill_(self.eval_spring_offset)
        if self.hand_offset_range is not None:
            sampled = torch.empty_like(offset).uniform_(*self.hand_offset_range)
            randomize = torch.rand(n, 1, 1, device=self.device) < self.hand_offset_random_prob
            offset = torch.where(randomize, sampled, offset)
        return offset

    def _sample_active_mask(self, n: int):
        active = torch.rand(n, self.num_force_bodies, 1, device=self.device) > 0.25
        if self.force_body_sampling == "balanced_hands":
            # Event mixture: 20% idle, 40% one hand, 20% both hands, and
            # 20% independently sampled upper-body contacts.
            choice = torch.rand(n, device=self.device)
            active[choice < 0.8] = False
            hand_ids = torch.as_tensor(self.hand_force_indices, device=self.device)
            selected_hand = hand_ids[torch.randint(0, 2, (n,), device=self.device)]
            row = torch.arange(n, device=self.device)
            single = (choice >= 0.2) & (choice < 0.6)
            active[row[single], selected_hand[single], 0] = True
            both = (choice >= 0.6) & (choice < 0.8)
            active[:, hand_ids, 0] |= both.unsqueeze(-1)
        else:
            active &= ~(torch.rand(n, 1, 1, device=self.device) < 0.25)
        return active & self.eval_force_body_mask.reshape(1, -1, 1)

    def sample_init(self, env_ids: torch.Tensor):
        root_state = self.init_root_state[env_ids].clone()
        root_state[:, :3] += self.env.scene.env_origins[env_ids]
        root_state[:, 2] += 0.04
        self.asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(self.init_joint_pos[env_ids], env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(self.init_joint_vel[env_ids], env_ids=env_ids)
        self.asset.set_joint_position_target(self.init_joint_pos[env_ids], env_ids=env_ids)
        self.asset.write_data_to_sim()
        self.nominal_initialized[env_ids] = False
        self._reward_reference_valid[env_ids] = False
        self.root_vel_ref_w[env_ids] = 0.0
        self.follow_input_force_w[env_ids] = 0.0
        self.follow_excess_force_w[env_ids] = 0.0
        self.root_pos_ref_w[env_ids] = root_state[:, :3]
        self.root_height_ref[env_ids] = root_state[:, 2:3]
        forward = torch.zeros(len(env_ids), 3, device=self.device)
        forward[:, 0] = 1.0
        self.heading_ref_w[env_ids] = quat_apply(root_state[:, 3:7], forward)
        self.force_reset(env_ids)

    def force_reset(self, env_ids: torch.Tensor):
        self.force_applied_w[env_ids] = 0.0
        self.force_applied_b[env_ids] = 0.0
        self.force_expected_w[env_ids] = 0.0
        self.force_expected_b[env_ids] = 0.0
        self.force_interact_b[env_ids] = 0.0
        self.force_interact_w[env_ids] = 0.0
        self.force_keypoint_vel_w[env_ids] = 0.0
        self.force_keypoint_w_prev[env_ids] = 0.0
        self.force_enable[env_ids] = False
        self.force_ramping_down[env_ids] = False
        self.force_ramp_timer[env_ids] = 0
        self._sample_force(env_ids)

    def reset(self, env_ids: torch.Tensor):
        self._cum_error[env_ids] = 0.0

    def set_eval_force_schedule(
        self,
        mode: str,
        start_s: float = 0.0,
        end_s: float | None = None,
        ramp_s: float = 0.0,
    ):
        """Set a deterministic force schedule for replay without changing training."""
        if mode not in {"random", "off", "spring", "on", "pulse", "stable"}:
            raise ValueError(f"Unknown evaluation force mode: {mode}")
        if start_s < 0.0 or ramp_s < 0.0:
            raise ValueError("Evaluation force times must be non-negative")
        if end_s is not None and end_s <= start_s:
            raise ValueError("Evaluation force end must be greater than start")
        self.eval_force_schedule = {
            "mode": mode,
            "start_s": float(start_s),
            "end_s": None if end_s is None else float(end_s),
            "ramp_s": float(ramp_s),
        }
        self.eval_force_step.zero_()

    def set_eval_stable_force(self, magnitude: float, direction: list[float]):
        """Set a constant WORLD-frame force in N per selected body.

        This is an evaluation load, not a spring or force estimate. Both the
        simulator and virtual reference receive it. Torso counter-wrenches and
        spring caps are bypassed so the requested physical load is unchanged.
        """
        if not math.isfinite(magnitude) or magnitude < 0.0:
            raise ValueError("Stable force magnitude must be finite and non-negative")
        direction_t = torch.as_tensor(direction, dtype=torch.float32, device=self.device)
        if direction_t.shape != (3,) or not torch.isfinite(direction_t).all():
            raise ValueError("Stable force direction must contain three finite numbers")
        norm = direction_t.norm()
        if norm < 1e-6:
            raise ValueError("Stable force direction must be nonzero")
        self.eval_stable_magnitude = float(magnitude)
        self.eval_stable_direction_w = direction_t / norm

    def _is_stable_replay(self):
        return self.eval_force_schedule is not None and self.eval_force_schedule["mode"] == "stable"

    def _stable_force_w(self):
        if self.eval_stable_magnitude is None or self.eval_stable_direction_w is None:
            raise RuntimeError("Stable replay requires magnitude and world direction")
        vector = self.eval_stable_direction_w * self.eval_stable_magnitude
        mask = self.eval_force_body_mask.reshape(1, -1, 1)
        return vector.reshape(1, 1, 3) * self.force_scale.unsqueeze(1) * mask

    def _constant_mask(self):
        if self._is_stable_replay():
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        random_schedule = self.eval_force_schedule is None or self.eval_force_schedule["mode"] == "random"
        if self.threshold_following and random_schedule:
            return self.load_is_constant
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _constant_load_w(self):
        if self._is_stable_replay():
            return self._stable_force_w()
        return self.constant_force_w * self.force_scale.unsqueeze(1) * self.force_enable

    def _physical_load_w(self, current_b):
        max_force = self.max_force if self.eval_max_force is None else self.eval_max_force
        spring_b = clamp_norm(self.force_kp * (self.force_origin_b - current_b), max_force)
        spring_w = quat_apply(self.asset.data.root_quat_w.unsqueeze(1), spring_b)
        load = torch.where(self._constant_mask()[:, None, None], self._constant_load_w(), spring_w)
        if self.threshold_following and not self._is_stable_replay():
            # Limit the actual resultant, rather than hiding it with a torso
            # counter-force. Use the identical law in the virtual reference.
            net_norm = load.sum(dim=1).norm(dim=-1, keepdim=True)
            scale = (self.net_force_limit / net_norm.clamp_min(1e-6)).clamp_max(1.0)
            load = load * scale.unsqueeze(-1)
        return load

    def _update_body_reference(self, physical_force_w):
        if self.threshold_following:
            # One threshold on the resultant horizontal load, not one safe
            # budget per link. Opposing two-hand forces do not command walking.
            total = physical_force_w.sum(dim=1).clone()
            total[:, 2] = 0.0
            self.follow_input_force_w[:] = total
            magnitude = total.norm(dim=-1, keepdim=True)
            excess = (magnitude - self.safe_force_limit).clamp_min(0.0)
            total = total * excess / magnitude.clamp_min(1e-6)
            total = clamp_norm(total, self.net_force_limit)
            self.follow_excess_force_w[:] = total
        else:
            total = clamp_norm(self.force_expected_w.sum(dim=1), self.net_force_limit)
            if self.following_rewards:
                total[:, 2] = 0.0
        acceleration = clamp_norm(
            (total - self.body_damping * self.root_vel_ref_w) / self.body_mass, self.body_acc_limit)
        self.root_vel_ref_w[:] = clamp_norm(
            self.root_vel_ref_w + acceleration * self.env.step_dt, self.body_vel_limit)
        self.root_pos_ref_w[:] = self.asset.data.root_pos_w + self.root_vel_ref_w * self.env.step_dt

    def set_eval_spring(
        self,
        stiffness: float | None = None,
        offset: float | None = None,
        max_force: float | None = None,
        direction: list[float] | tuple[float, float, float] | None = None,
        safe_force: float | None = None,
        body_names: list[str] | tuple[str, ...] | None = None,
    ):
        """Override spring parameters for deterministic replay only."""
        if stiffness is not None and stiffness < 0.0:
            raise ValueError("Evaluation spring stiffness must be non-negative")
        if offset is not None and offset < 0.0:
            raise ValueError("Evaluation spring offset must be non-negative")
        if max_force is not None and max_force <= 0.0:
            raise ValueError("Evaluation max force must be positive")
        if safe_force is not None and not 5.0 <= safe_force <= 15.0:
            raise ValueError("Evaluation safe force must be between 5 and 15 N")
        if direction is not None:
            direction_t = torch.as_tensor(direction, dtype=torch.float32, device=self.device)
            if direction_t.numel() != 3 or direction_t.norm() < 1e-6:
                raise ValueError("Evaluation spring direction must be a nonzero 3-vector")
            self.eval_direction_b = direction_t / direction_t.norm()
        self.eval_spring_k = None if stiffness is None else float(stiffness)
        self.eval_spring_offset = None if offset is None else float(offset)
        self.eval_max_force = None if max_force is None else float(max_force)
        self.eval_safe_force = None if safe_force is None else float(safe_force)
        self.eval_force_body_mask[:] = False
        if body_names is None:
            self.eval_force_body_mask[:] = True
        else:
            requested = set(body_names)
            available = {
                self.asset.body_names[i] for i in self.force_apply_idx_asset.tolist()
            }
            unknown = requested - available
            if unknown or not requested:
                raise ValueError(
                    f"Unknown evaluation force bodies: {sorted(unknown)}; "
                    f"select at least one of: {sorted(available)}"
                )
            for force_i, asset_i in enumerate(self.force_apply_idx_asset.tolist()):
                self.eval_force_body_mask[force_i] = self.asset.body_names[asset_i] in requested

    def _update_eval_schedule(self):
        schedule = self.eval_force_schedule
        if schedule is None or schedule["mode"] == "random":
            self._update_schedule()
            return

        time_s = self.eval_force_step.float().unsqueeze(-1) * self.env.step_dt
        mode = schedule["mode"]
        if mode == "off":
            scale = torch.zeros_like(time_s)
        elif mode in {"spring", "on"}:
            scale = torch.ones_like(time_s)
        else:
            end_s = schedule["end_s"]
            if end_s is None and mode != "stable":
                raise ValueError("Pulse evaluation schedule requires end_s")
            active = time_s >= schedule["start_s"]
            if end_s is not None:
                active &= time_s < end_s
            scale = active.float()
            ramp_s = schedule["ramp_s"]
            if ramp_s > 0.0:
                ramp_in = ((time_s - schedule["start_s"]) / ramp_s).clamp(0.0, 1.0)
                if end_s is None:
                    scale = scale * ramp_in
                else:
                    ramp_out = ((end_s - time_s) / ramp_s).clamp(0.0, 1.0)
                    scale = scale * torch.minimum(ramp_in, ramp_out)

        self.force_scale[:] = scale
        self.force_enable[:] = scale.unsqueeze(1) > 0.0
        self.force_enable[:] &= self.eval_force_body_mask.reshape(1, -1, 1)
        self.force_kp = self.force_kp_nominal * self.force_enable * scale.unsqueeze(1)
        self.eval_force_step += 1

    def _update_schedule(self):
        if self.threshold_following:
            self._update_mixed_schedule()
            return
        self.force_timer -= 1
        active = self.force_enable.any(dim=(1, 2))
        begin_down = (self.force_timer <= 0) & active & (~self.force_ramping_down)
        idle_expired = (self.force_timer <= 0) & (~active) & (~self.force_ramping_down)
        if idle_expired.any():
            self._sample_force(idle_expired.nonzero(as_tuple=False).squeeze(-1))
        if begin_down.any():
            ids = begin_down.nonzero(as_tuple=False).squeeze(-1)
            self.force_ramping_down[ids] = True
            self.force_ramp_timer[ids] = torch.randint(
                self.ramp_time_range[0], self.ramp_time_range[1] + 1, (ids.numel(),),
                device=self.device, dtype=self.force_ramp_timer.dtype,
            )
            self.force_ramp_total[ids] = self.force_ramp_timer[ids]
        down = self.force_ramping_down
        if down.any():
            self.force_ramp_timer[down] -= 1
            self.force_scale[down] = (self.force_ramp_timer[down].float() / self.force_ramp_total[down].float().clamp_min(1)).unsqueeze(-1).clamp(0.0, 1.0)
            done = down & (self.force_ramp_timer <= 0)
            if done.any():
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                self._sample_force(ids)
                self.force_ramping_down[ids] = False
        self.force_kp = self.force_kp_nominal * self.force_enable * self.force_scale.unsqueeze(-1)

    def _update_mixed_schedule(self):
        # A force event is ramp-in, hold, ramp-out, then resample. Idle events
        # retain their all-zero mask for the complete interval.
        self.load_age += 1
        total_steps = self.load_hold_steps + 2 * self.load_ramp_steps
        done = self.load_age >= total_steps
        if done.any():
            self._sample_force(done.nonzero(as_tuple=False).squeeze(-1))
        ramp = self.load_ramp_steps.float().clamp_min(1)
        ramp_in = self.load_age.float() / ramp
        ramp_out = (self.load_hold_steps + 2 * self.load_ramp_steps - self.load_age).float() / ramp
        self.force_scale[:] = torch.minimum(ramp_in, ramp_out).clamp(0, 1).unsqueeze(-1)
        self.force_kp = self.force_kp_nominal * self.force_enable * self.force_scale.unsqueeze(1)

    def before_update(self):
        new_ids = None
        if (~self.nominal_initialized).any():
            ids = (~self.nominal_initialized).nonzero(as_tuple=False).squeeze(-1)
            new_ids = ids
            current = self._current_hand_b().detach()
            self.nominal_hand_b[ids] = current[ids]
            self.nominal_initialized[ids] = True
            self.force_origin_b[ids] = self.nominal_hand_b[ids] + self.force_dir_b[ids] * self.force_offset[ids]
            self.admit.reset(ids, self.nominal_hand_b[ids], torch.zeros_like(self.nominal_hand_b[ids]))

        if not self.compliance:
            self.force_enable.zero_()
            self.force_kp.zero_()
        else:
            self._update_eval_schedule()

        root_pos = self.asset.data.root_pos_w.unsqueeze(1)
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        current_b = self._current_hand_b()
        force_limit = self.safe_force_limit.unsqueeze(1)
        drive_kp = force_limit / 0.05
        drive_kd = 2.0 * torch.sqrt(drive_kp * 0.1)
        physical_w = self._physical_load_w(current_b)
        for _ in range(4):
            drive = drive_kp * (self.nominal_hand_b - self.admit.x[0]) - drive_kd * self.admit.v[0]
            drive = clamp_norm(drive, force_limit)
            ext = quat_apply_inverse(root_quat, self._physical_load_w(self.admit.x[0]))
            self.admit.step(drive, ext)
            self._bound_hand_reference()
        self.force_drive_b[:] = drive
        self.force_drive_w[:] = quat_apply(root_quat, drive)

        self.force_interact_b[:] = ext
        self.force_interact_w[:] = quat_apply(root_quat, ext)

        self.force_keypoint_b[:] = self.admit.x[0]
        self.force_keypoint_w[:] = quat_apply(root_quat, self.force_keypoint_b) + root_pos
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        if new_ids is not None:
            self.force_keypoint_w_prev[new_ids] = self.force_keypoint_w[new_ids]
            self.force_keypoint_vel_w[new_ids] = 0.0
        self.force_keypoint_w_prev[:] = self.force_keypoint_w
        expected_w = self._physical_load_w(self.force_keypoint_b)
        expected_b = quat_apply_inverse(root_quat, expected_w)
        self.force_expected_b[:] = expected_b
        self.force_expected_w[:] = expected_w

        # Body-level admittance: the reference is explicitly bounded and is
        # independent from the hand admittance mass used above.
        self._update_body_reference(physical_w)

        self._cum_error[:, 0] = (self.asset.data.root_pos_w[:, 2] - self.root_height_ref[:, 0]).abs()
        self._cum_error[:, 1] = self.asset.data.projected_gravity_b[:, :2].norm(dim=-1)
        self._cum_error[:, 2] = (self.force_keypoint_w - self.asset.data.body_pos_w[:, self.force_apply_idx_asset]).norm(dim=-1).mean(dim=-1)

    def update(self):
        if self.following_rewards:
            self._accumulate_tracking_metrics()

    def _bound_hand_reference(self):
        if self.hand_reference_limit is None:
            return
        delta = self.admit.x[0] - self.nominal_hand_b
        distance = delta.norm(dim=-1, keepdim=True)
        direction = delta / distance.clamp_min(1e-6)
        outside = distance > self.hand_reference_limit
        self.admit.x[0] = self.nominal_hand_b + clamp_norm(delta, self.hand_reference_limit)
        # Anti-windup: discard only outward velocity at the displacement bound.
        outward = (self.admit.v[0] * direction).sum(-1, keepdim=True).clamp_min(0.0)
        self.admit.v[0] -= outside * outward * direction

    def _moving(self):
        target = self._reward_value("velocity", self.root_vel_ref_w)
        return target[:, :2].norm(dim=-1, keepdim=True) > self.moving_speed_threshold

    def _reward_value(self, name, current):
        # BaseEnv prepares the NEXT command before computing this interval's
        # reward. Compare physical outcomes with the command actually executed.
        if not self.threshold_following or name not in self._reward_reference:
            return current
        valid = self._reward_reference_valid.reshape((-1,) + (1,) * (current.ndim - 1))
        return torch.where(valid, self._reward_reference[name], current)

    def _active_force_mean(self, values):
        mask = self._reward_value("enabled", self.force_enable).squeeze(-1).to(values.dtype)
        return (values * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1.0)

    def _heading_error(self):
        forward = torch.zeros_like(self.heading_ref_w)
        forward[:, 0] = 1.0
        current = quat_apply(self.asset.data.root_quat_w, forward)[:, :2]
        reference = self.heading_ref_w[:, :2]
        cross = current[:, 0] * reference[:, 1] - current[:, 1] * reference[:, 0]
        dot = (current * reference).sum(-1)
        return torch.atan2(cross, dot).unsqueeze(-1)

    def _accumulate_tracking_metrics(self):
        moving = self._moving().squeeze(-1)
        speed = self.asset.data.root_lin_vel_w[:, :2]
        target = self._reward_value("velocity", self.root_vel_ref_w)[:, :2]
        target_speed = target.norm(dim=-1)
        error = (speed - target).norm(dim=-1)
        progress = (speed * target).sum(-1) / target_speed.square().clamp_min(1e-6)
        active = self._reward_value("enabled", self.force_enable).squeeze(-1)
        expected = self._reward_value("expected", self.force_expected_w)
        hand_target = self._reward_value("hand_position", self.force_keypoint_w)
        force_error = (self.force_applied_w-expected).norm(dim=-1)
        hand_error = (self.asset.data.body_pos_w[:, self.force_apply_idx_asset]-hand_target).norm(dim=-1)
        contact = self.contact_sensor.data.current_contact_time[:, self.foot_sensor_ids] > 0
        foot_speed = self.asset.data.body_lin_vel_w[:, self.foot_asset_ids, :2].norm(dim=-1)
        tracked = moving & (error < self.tracking_tolerance) & (progress >= 0.8)
        values = [
            moving.sum(), (~moving).sum(), (error * moving).sum(),
            (speed.norm(dim=-1) * moving).sum(), (target_speed * moving).sum(),
            (progress * moving).sum(), tracked.sum(),
            (speed.norm(dim=-1) * (~moving)).sum(),
            self.asset.data.root_ang_vel_w[:, 2].abs().sum(),
            self._heading_error().abs().sum(), (force_error * active).sum(),
            active.sum(), (hand_error * active).sum(), (foot_speed * contact).sum(),
            contact.sum(), (contact.sum(-1) == 1).sum(),
        ]
        self.metric_sums += torch.stack(values)
        if self.threshold_following:
            norm = self._reward_value("input", self.follow_input_force_w).norm(dim=-1)
            excess = self._reward_value("excess", self.follow_excess_force_w).norm(dim=-1)
            bound = (self.force_keypoint_b - self.nominal_hand_b).norm(dim=-1)
            at_bound = (bound >= self.hand_reference_limit - 1e-5).float().mean(-1) if self.hand_reference_limit else torch.zeros_like(norm)
            self.threshold_metric_sums += torch.stack([
                self._reward_value("constant", self._constant_mask()).float().sum(),
                (norm > self._reward_value("safe", self.safe_force_limit)[:, 0]).float().sum(),
                norm.sum(), excess.sum(), at_bound.sum(), torch.ones_like(norm).sum()])

    def tracking_diagnostics(self):
        """Conditional physical metrics since the last training log call."""
        if not self.following_rewards:
            return {}
        s = self.metric_sums
        count = (s[0] + s[1]).clamp_min(1)
        moving_count = s[0].clamp_min(1)
        rest_count = s[1].clamp_min(1)
        active_count = s[11].clamp_min(1)
        names = ["moving_fraction", "moving_velocity_error_m_s", "moving_actual_speed_m_s",
                 "moving_target_speed_m_s", "moving_progress_ratio", "moving_tracking_fraction",
                 "rest_speed_m_s", "yaw_rate_abs_rad_s", "heading_error_abs_rad",
                 "active_force_error_N", "active_hand_error_m", "contact_slip_speed_m_s",
                 "single_support_fraction", "moving_samples", "rest_samples", "active_force_samples"]
        values = torch.stack([
            s[0]/count, s[2]/moving_count, s[3]/moving_count, s[4]/moving_count,
            s[5]/moving_count, s[6]/moving_count, s[7]/rest_count, s[8]/count,
            s[9]/count, s[10]/active_count, s[12]/active_count,
            s[13]/s[14].clamp_min(1), s[15]/count, s[0], s[1], s[11],
        ]).detach().cpu().tolist()
        self.metric_sums.zero_()
        result = {f"settle/{name}": value for name, value in zip(names, values)}
        if self.threshold_following:
            stats = self.threshold_metric_sums
            extra_names = ["constant_load_fraction", "over_safe_fraction", "known_planar_load_N",
                           "follow_excess_N", "hand_reference_bound_fraction"]
            extra_values = (stats[:5] / stats[5].clamp_min(1)).detach().cpu().tolist()
            result.update({f"settle/{name}": value for name, value in zip(extra_names, extra_values)})
            result["settle/load_curriculum_scale"] = self.load_curriculum_scale
            stats.zero_()
        return result

    def step(self, substep: int):
        if substep == 0:
            self.force_apply_buffer.zero_()
            self.torque_apply_buffer.zero_()
            if self.threshold_following:
                self._reward_reference = {name: value.clone() for name, value in dict(
                    velocity=self.root_vel_ref_w, expected=self.force_expected_w,
                    hand_position=self.force_keypoint_w, hand_velocity=self.force_keypoint_vel_w,
                    enabled=self.force_enable, constant=self._constant_mask(), safe=self.safe_force_limit,
                    input=self.follow_input_force_w, excess=self.follow_excess_force_w).items()}
                self._reward_reference_valid[:] = True
        if not self.compliance:
            return
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        if self._is_stable_replay():
            self.force_applied_w[:] = self._stable_force_w()
            self.force_applied_b[:] = quat_apply_inverse(root_quat, self.force_applied_w)
            self.force_apply_buffer[:, self.force_apply_idx_asset] = self.force_applied_w
            # Keep the requested load exact. It acts at each selected rigid
            # body's center of mass; off-center forces naturally induce torque.
            self.asset.has_external_wrench = False
            return
        hand_pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        current_b = quat_apply_inverse(root_quat, hand_pos - self.asset.data.root_pos_w.unsqueeze(1))
        self.force_applied_w[:] = self._physical_load_w(current_b)
        self.force_applied_b[:] = quat_apply_inverse(root_quat, self.force_applied_w)
        self.force_apply_buffer[:, self.force_apply_idx_asset] = self.force_applied_w
        hand_quat = self.asset.data.body_quat_w[:, self.force_apply_idx_asset]
        delta_w = quat_apply(hand_quat, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset] = torch.cross(delta_w, self.force_applied_w, dim=-1)
        if self.threshold_following:
            # Train on the same uncompensated physical wrench used by stable
            # replay. A torso counter-wrench would conceal single-hand torque.
            constant = self._constant_mask()[:, None, None]
            self.torque_apply_buffer[:, self.force_apply_idx_asset] *= ~constant
            self.asset.has_external_wrench = False
        else:
            self._limit_net_wrench()

    def _limit_net_wrench(self):
        pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        torso = self.asset.data.body_pos_w[:, self.torso_idx_asset].unsqueeze(1)
        f = self.force_apply_buffer[:, self.force_apply_idx_asset]
        tau = self.torque_apply_buffer[:, self.force_apply_idx_asset]
        net_f = f.sum(1)
        net_tau = torch.cross(pos - torso, f, dim=-1).sum(1) + tau.sum(1)
        f_allowed = net_f * torch.clamp(self.net_force_limit / net_f.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        tau_allowed = net_tau * torch.clamp(self.net_torque_limit / net_tau.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        self.force_apply_buffer[:, self.torso_idx_asset] = f_allowed - net_f
        self.torque_apply_buffer[:, self.torso_idx_asset] = tau_allowed - net_tau
        self.asset.has_external_wrench = False

    def debug_draw(self):
        if not self.env.sim.has_gui():
            return
        pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        self.env.debug_draw.vector(pos.reshape(-1, 3), (self.force_applied_w * 0.02).reshape(-1, 3), color=(0.0, 0.3, 1.0, 1.0))
        self.env.debug_draw.vector(self.asset.data.root_pos_w, self.root_vel_ref_w * 0.3, color=(1.0, 0.2, 0.1, 1.0))

    @observation
    def command(self):
        if self.threshold_following:
            # Both actors consume this group. Unknown-load-derived targets
            # belong only in priv; adapt cannot remove ordinary actor inputs.
            return self.safe_force_limit
        keypoint_b = self.force_keypoint_b.reshape(self.num_envs, -1)
        return torch.cat([keypoint_b, self.safe_force_limit], dim=-1)

    def command_sym(self):
        if self.threshold_following:
            return sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1])
        return sym_utils.SymmetryTransform.cat([
            sym_utils.cartesian_space_symmetry(self.asset, [self.asset.body_names[i] for i in self.force_apply_idx_asset], sign=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
        ])

    @observation
    def yield_command(self):
        """Known task command, not a sensorless force estimate.

        Include heading error so the policy can correct accumulated yaw, not
        merely stop yaw rate after it has already turned away.
        """
        velocity_b = quat_apply_inverse(self.asset.data.root_quat_w, self.root_vel_ref_w)
        heading = self._heading_error()
        return torch.cat([velocity_b, torch.cos(heading), torch.sin(heading)], dim=-1)

    def yield_command_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(5), signs=[1, -1, 1, 1, -1])

    @observation
    def heading_command(self):
        # Relative to the episode's initial heading; independent of unknown
        # force. Deployment can compute this from its orientation estimator.
        heading = self._heading_error()
        return torch.cat([torch.cos(heading), torch.sin(heading)], dim=-1)

    def heading_command_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(2), signs=[1, -1])

    @observation
    def force_priv(self):
        if self.threshold_following:
            root_quat = self.asset.data.root_quat_w
            return torch.cat([
                self.force_keypoint_b.reshape(self.num_envs, -1),
                self.admit.v[0].reshape(self.num_envs, -1),
                self.force_applied_b.reshape(self.num_envs, -1),
                self.force_expected_b.reshape(self.num_envs, -1),
                quat_apply_inverse(root_quat, self.root_vel_ref_w),
                quat_apply_inverse(root_quat, self.follow_input_force_w),
                quat_apply_inverse(root_quat, self.follow_excess_force_w),
            ], dim=-1)
        return torch.cat([
            self.force_keypoint_b.reshape(self.num_envs, -1),
            self.force_applied_b.reshape(self.num_envs, -1),
            self.force_expected_b.reshape(self.num_envs, -1),
            self.root_vel_ref_w,
            self.force_timer.float().unsqueeze(-1),
        ], dim=-1)

    def force_priv_sym(self):
        if self.threshold_following:
            return sym_utils.SymmetryTransform.cat([
                sym_utils.cartesian_space_symmetry(self.asset,
                    [self.asset.body_names[i] for i in self.force_apply_idx_asset], sign=[1, -1, 1]).repeat(4),
                sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]).repeat(3),
            ])
        return sym_utils.SymmetryTransform.cat([
            sym_utils.cartesian_space_symmetry(self.asset, [self.asset.body_names[i] for i in self.force_apply_idx_asset], sign=[1, -1, 1]).repeat(3),
            sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
        ])

    @observation
    def target_joint_pos_obs(self):
        return self.init_joint_pos.expand(self.num_envs, -1)

    def target_joint_pos_obs_sym(self):
        return sym_utils.joint_space_symmetry(self.asset, self.asset.joint_names)

    @observation
    def target_projected_gravity_b(self):
        return torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)

    def target_projected_gravity_b_sym(self):
        return sym_utils.SymmetryTransform(perm=torch.arange(3), signs=[1, -1, 1])

    @reward
    def force_target_tracking(self):
        target = self._reward_value("hand_position", self.force_keypoint_w)
        error = (self.asset.data.body_pos_w[:, self.force_apply_idx_asset] - target).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_error(error, 0.08)

    @reward
    def force_target_vel_tracking(self):
        target = self._reward_value("hand_velocity", self.force_keypoint_vel_w)
        error = (self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset] - target).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_error(error, 0.5)

    @reward
    def force_reward(self):
        errors = (self.force_applied_w - self._reward_value("expected", self.force_expected_w)).norm(dim=-1)
        if self.following_rewards:
            error = self._active_force_mean(errors)
            valid = self._reward_value("enabled", self.force_enable).any(dim=1)
            if self.threshold_following:
                valid = valid & ~self._reward_value("constant", self._constant_mask()).unsqueeze(-1)
            return _exp_error(error, 4.0), valid
        error = errors.mean(dim=-1, keepdim=True)
        return _exp_error(error, 8.0)

    @reward
    def force_exd_penalty(self):
        excess = (self.force_applied_w.norm(dim=-1) - self._reward_value("safe", self.safe_force_limit)).clamp_min(0.0)
        if self.following_rewards:
            valid = self._reward_value("enabled", self.force_enable).any(dim=1)
            if self.threshold_following:
                valid = valid & ~self._reward_value("constant", self._constant_mask()).unsqueeze(-1)
            return -self._active_force_mean(excess) / 10.0, valid
        return -excess.mean(dim=-1, keepdim=True) / 10.0

    @reward
    def stand_height(self):
        error = (self.asset.data.root_pos_w[:, 2:3] - self.root_height_ref).abs()
        return _exp_error(error, 0.08)

    @reward
    def stand_upright(self):
        error = self.asset.data.projected_gravity_b[:, :2].norm(dim=-1, keepdim=True)
        return _exp_error(error, 0.25)

    @reward
    def stand_lower_body(self):
        if self.lower_joint_ids.numel() == 0:
            return torch.ones(self.num_envs, 1, device=self.device)
        error = (self.asset.data.joint_pos[:, self.lower_joint_ids] - self.init_joint_pos[:, self.lower_joint_ids]).abs().mean(dim=-1, keepdim=True)
        score = _exp_error(error, 0.35)
        # Keep nominal posture as an idle objective; let the legs articulate
        # while a nonzero translation command is being tracked.
        return score * (1.0 - 0.9 * self._moving().float()) if self.following_rewards else score

    @reward
    def body_yield_velocity(self):
        speed = self.asset.data.root_lin_vel_w[:, :2]
        target = self._reward_value("velocity", self.root_vel_ref_w)[:, :2]
        error = (speed - target).norm(dim=-1, keepdim=True)
        if self.following_rewards:
            score = torch.exp(-(error / self.tracking_sigma).square())
            if self.threshold_following:
                score *= self._balance_factor()
            return score, self._moving()
        force = self.force_expected_w.sum(dim=1)
        force_xy = force[:, :2]
        force_active = (force_xy.norm(dim=-1, keepdim=True) > 1.0).float()
        return force_active * _exp_error(error, 0.35) + (1.0 - force_active) * _exp_error(speed.norm(dim=-1, keepdim=True), 0.20)

    @reward
    def body_rest_velocity(self):
        speed = self.asset.data.root_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
        return torch.exp(-(speed / 0.08).square()), ~self._moving()

    def _balance_factor(self):
        height_error = (self.asset.data.root_pos_w[:, 2:3] - self.root_height_ref).abs()
        tilt = self.asset.data.projected_gravity_b[:, :2].norm(dim=-1, keepdim=True)
        return torch.exp(-(height_error / 0.10).square() - (tilt / 0.35).square())

    @reward
    def body_velocity_error(self):
        # A dense Huber cost still distinguishes progress when the exponential
        # tracking score is near zero. Applies during braking and at rest too.
        target = self._reward_value("velocity", self.root_vel_ref_w)[:, :2]
        error = (self.asset.data.root_lin_vel_w[:, :2] - target).norm(dim=-1, keepdim=True) / 0.20
        return -torch.where(error < 1.0, 0.5 * error.square(), error - 0.5)

    @reward
    def body_yield_stall(self):
        target = self._reward_value("velocity", self.root_vel_ref_w)[:, :2]
        progress = (self.asset.data.root_lin_vel_w[:, :2] * target).sum(-1, keepdim=True)
        progress = progress / target.square().sum(-1, keepdim=True).clamp_min(1e-6)
        # Zero speed costs one; reverse motion costs up to two. Overspeed is
        # handled by the vector tracking score rather than rewarded here.
        return -(1.0 - progress).clamp(0.0, 2.0), self._moving()

    @reward
    def body_heading_tracking(self):
        return torch.exp(-(self._heading_error() / 0.25).square())

    @reward
    def body_yaw_rate(self):
        return -self.asset.data.root_ang_vel_w[:, 2:3].square().clamp_max(9.0)

    @reward
    def yield_step_air_time(self):
        touchdown = self.contact_sensor.compute_first_contact(self.env.step_dt)[:, self.foot_sensor_ids]
        air_time = self.contact_sensor.data.last_air_time[:, self.foot_sensor_ids]
        target = self._reward_value("velocity", self.root_vel_ref_w)[:, :2]
        progress = (self.asset.data.root_lin_vel_w[:, :2] * target).sum(-1, keepdim=True)
        progress = progress / target.square().sum(-1, keepdim=True).clamp_min(1e-6)
        # Reward short steps only when translating in the commanded direction.
        # Merely hopping or standing on one leg does not earn this term.
        score = ((air_time - 0.12).clamp(0.0, 0.25) * touchdown).sum(-1, keepdim=True)
        score = score * progress.clamp(0.0, 1.0)
        if self.threshold_following:
            score *= self._balance_factor()
        return score, self._moving()

