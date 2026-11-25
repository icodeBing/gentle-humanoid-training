"""Stationary whole-body compliance task.

This command keeps the useful part of GentleHumanoid's interaction model
(spring force -> hand admittance target) without requiring a motion dataset.
It also integrates a separate, bounded body-level velocity reference from the
net interaction force.  The latter is a task reference, not a force sensor.
"""

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
        self.hand_stiffness_range = tuple(float(x) for x in hand_stiffness_range)
        self.safe_force_range = tuple(float(x) for x in safe_force_range)
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
        self.force_kp_nominal = torch.zeros(self.num_envs, self.num_force_bodies, 1, device=self.device)
        self.force_kp = torch.zeros(self.num_envs, self.num_force_bodies, 1, device=self.device)
        self.force_enable = torch.zeros(self.num_envs, self.num_force_bodies, 1, dtype=torch.bool, device=self.device)
        self.force_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramp_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramp_total = torch.ones(self.num_envs, dtype=torch.int32, device=self.device)
        self.force_ramping_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_scale = torch.zeros(self.num_envs, 1, device=self.device)
        self.safe_force_limit = torch.zeros(self.num_envs, 1, device=self.device)

        self.force_keypoint_b = torch.zeros(shape, device=self.device)
        self.force_keypoint_w = torch.zeros(shape, device=self.device)
        self.force_keypoint_vel_w = torch.zeros(shape, device=self.device)
        self.force_keypoint_w_prev = torch.zeros(shape, device=self.device)
        self.force_applied_w = torch.zeros(shape, device=self.device)
        self.force_applied_b = torch.zeros(shape, device=self.device)
        self.force_expected_w = torch.zeros(shape, device=self.device)
        self.force_expected_b = torch.zeros(shape, device=self.device)
        self.force_pos_delta = torch.zeros(shape, device=self.device)
        self.force_apply_buffer = torch.zeros(self.num_envs, self.asset.num_bodies, 3, device=self.device)
        self.torque_apply_buffer = torch.zeros_like(self.force_apply_buffer)
        self.force_apply_world = True

        self.root_vel_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_pos_ref_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.root_height_ref = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_pos_boot_protect = self.init_joint_pos.clone()
        self._cum_error = torch.zeros(self.num_envs, 3, device=self.device)
        self.skip_ref = True
        # Optional deterministic schedule used only by evaluation replay.
        self.eval_force_schedule = None
        self.eval_force_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.eval_spring_k = None
        self.eval_spring_offset = None
        self.eval_max_force = None
        self.eval_safe_force = None
        self.eval_direction_b = None
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
        offset = self.hand_offset if self.eval_spring_offset is None else self.eval_spring_offset
        self.force_origin_b[env_ids] = self.nominal_hand_b[env_ids] + direction * offset
        if self.eval_spring_k is None:
            self.force_kp[env_ids] = torch.empty(n, self.num_force_bodies, 1, device=self.device).uniform_(*self.hand_stiffness_range)
        else:
            self.force_kp[env_ids] = self.eval_spring_k
        # Mask stiffness at reset as well, before the first physics step.
        self.force_kp[env_ids] *= self.eval_force_body_mask.reshape(1, -1, 1)
        self.force_kp_nominal[env_ids] = self.force_kp[env_ids]
        active = torch.rand(n, self.num_force_bodies, 1, device=self.device) > 0.25
        active &= self.eval_force_body_mask.reshape(1, -1, 1)
        idle = torch.rand(n, 1, 1, device=self.device) < 0.25
        active &= ~idle
        self.force_enable[env_ids] = active
        self.force_timer[env_ids] = torch.randint(
            self.force_time_range[0], self.force_time_range[1] + 1, (n,),
            device=self.device, dtype=self.force_timer.dtype,
        )
        self.force_scale[env_ids] = 1.0
        if self.eval_safe_force is None:
            self.safe_force_limit[env_ids] = torch.empty(n, 1, device=self.device).uniform_(*self.safe_force_range)
        else:
            self.safe_force_limit[env_ids] = self.eval_safe_force

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
        self.root_vel_ref_w[env_ids] = 0.0
        self.root_pos_ref_w[env_ids] = root_state[:, :3]
        self.root_height_ref[env_ids] = root_state[:, 2:3]
        self.force_reset(env_ids)

    def force_reset(self, env_ids: torch.Tensor):
        self.force_applied_w[env_ids] = 0.0
        self.force_applied_b[env_ids] = 0.0
        self.force_expected_w[env_ids] = 0.0
        self.force_expected_b[env_ids] = 0.0
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
        if mode not in {"random", "off", "spring", "on", "pulse"}:
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
            if end_s is None:
                raise ValueError("Pulse evaluation schedule requires end_s")
            active = (time_s >= schedule["start_s"]) & (time_s < end_s)
            scale = active.float()
            ramp_s = schedule["ramp_s"]
            if ramp_s > 0.0:
                ramp_in = ((time_s - schedule["start_s"]) / ramp_s).clamp(0.0, 1.0)
                ramp_out = ((end_s - time_s) / ramp_s).clamp(0.0, 1.0)
                scale = scale * torch.minimum(ramp_in, ramp_out)

        self.force_scale[:] = scale
        self.force_enable[:] = scale.unsqueeze(1) > 0.0
        self.force_enable[:] &= self.eval_force_body_mask.reshape(1, -1, 1)
        self.force_kp = self.force_kp_nominal * self.force_enable * scale.unsqueeze(1)
        self.eval_force_step += 1

    def _update_schedule(self):
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

    def before_update(self):
        new_ids = None
        if (~self.nominal_initialized).any():
            ids = (~self.nominal_initialized).nonzero(as_tuple=False).squeeze(-1)
            new_ids = ids
            current = self._current_hand_b().detach()
            self.nominal_hand_b[ids] = current[ids]
            self.nominal_initialized[ids] = True
            offset = self.hand_offset if self.eval_spring_offset is None else self.eval_spring_offset
            self.force_origin_b[ids] = self.nominal_hand_b[ids] + self.force_dir_b[ids] * offset
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
        for _ in range(4):
            drive = drive_kp * (self.nominal_hand_b - self.admit.x[0]) - drive_kd * self.admit.v[0]
            drive = clamp_norm(drive, force_limit)
            delta = self.force_origin_b - self.admit.x[0]
            ext = self.force_kp * delta
            max_force = self.max_force if self.eval_max_force is None else self.eval_max_force
            ext = clamp_norm(ext, max_force)
            self.admit.step(drive, ext)

        self.force_keypoint_b[:] = self.admit.x[0]
        self.force_keypoint_w[:] = quat_apply(root_quat, self.force_keypoint_b) + root_pos
        self.force_keypoint_vel_w[:] = (self.force_keypoint_w - self.force_keypoint_w_prev) / self.env.step_dt
        if new_ids is not None:
            self.force_keypoint_w_prev[new_ids] = self.force_keypoint_w[new_ids]
            self.force_keypoint_vel_w[new_ids] = 0.0
        self.force_keypoint_w_prev[:] = self.force_keypoint_w
        max_force = self.max_force if self.eval_max_force is None else self.eval_max_force
        expected_b = clamp_norm(self.force_kp * (self.force_origin_b - self.force_keypoint_b), max_force)
        self.force_expected_b[:] = expected_b
        self.force_expected_w[:] = quat_apply(root_quat, expected_b)

        # Body-level admittance: the reference is explicitly bounded and is
        # independent from the hand admittance mass used above.
        total_force = self.force_expected_w.sum(dim=1)
        total_force = clamp_norm(total_force, self.net_force_limit)
        root_acc = (total_force - self.body_damping * self.root_vel_ref_w) / self.body_mass
        root_acc = clamp_norm(root_acc, self.body_acc_limit)
        self.root_vel_ref_w[:] = clamp_norm(self.root_vel_ref_w + root_acc * self.env.step_dt, self.body_vel_limit)
        self.root_pos_ref_w[:] = self.asset.data.root_pos_w + self.root_vel_ref_w * self.env.step_dt

        self._cum_error[:, 0] = (self.asset.data.root_pos_w[:, 2] - self.root_height_ref[:, 0]).abs()
        self._cum_error[:, 1] = self.asset.data.projected_gravity_b[:, :2].norm(dim=-1)
        self._cum_error[:, 2] = (self.force_keypoint_w - self.asset.data.body_pos_w[:, self.force_apply_idx_asset]).norm(dim=-1).mean(dim=-1)

    def update(self):
        pass

    def step(self, substep: int):
        if substep == 0:
            self.force_apply_buffer.zero_()
            self.torque_apply_buffer.zero_()
        if not self.compliance:
            return
        root_quat = self.asset.data.root_quat_w.unsqueeze(1)
        hand_pos = self.asset.data.body_pos_w[:, self.force_apply_idx_asset]
        hand_delta = self.force_origin_b
        current_b = quat_apply_inverse(root_quat, hand_pos - self.asset.data.root_pos_w.unsqueeze(1))
        max_force = self.max_force if self.eval_max_force is None else self.eval_max_force
        force_b = clamp_norm(self.force_kp * (hand_delta - current_b), max_force)
        self.force_applied_w[:] = quat_apply(root_quat, force_b)
        self.force_applied_b[:] = force_b
        self.force_apply_buffer[:, self.force_apply_idx_asset] = self.force_applied_w
        hand_quat = self.asset.data.body_quat_w[:, self.force_apply_idx_asset]
        delta_w = quat_apply(hand_quat, self.force_pos_delta)
        self.torque_apply_buffer[:, self.force_apply_idx_asset] = torch.cross(delta_w, self.force_applied_w, dim=-1)
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
        keypoint_b = self.force_keypoint_b.reshape(self.num_envs, -1)
        return torch.cat([keypoint_b, self.safe_force_limit], dim=-1)

    def command_sym(self):
        return sym_utils.SymmetryTransform.cat([
            sym_utils.cartesian_space_symmetry(self.asset, [self.asset.body_names[i] for i in self.force_apply_idx_asset], sign=[1, -1, 1]),
            sym_utils.SymmetryTransform(perm=torch.arange(1), signs=[1]),
        ])

    @observation
    def force_priv(self):
        return torch.cat([
            self.force_keypoint_b.reshape(self.num_envs, -1),
            self.force_applied_b.reshape(self.num_envs, -1),
            self.force_expected_b.reshape(self.num_envs, -1),
            self.root_vel_ref_w,
            self.force_timer.float().unsqueeze(-1),
        ], dim=-1)

    def force_priv_sym(self):
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
        error = (self.asset.data.body_pos_w[:, self.force_apply_idx_asset] - self.force_keypoint_w).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_error(error, 0.08)

    @reward
    def force_target_vel_tracking(self):
        error = (self.asset.data.body_lin_vel_w[:, self.force_apply_idx_asset] - self.force_keypoint_vel_w).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_error(error, 0.5)

    @reward
    def force_reward(self):
        error = (self.force_applied_w - self.force_expected_w).norm(dim=-1).mean(dim=-1, keepdim=True)
        return _exp_error(error, 8.0)

    @reward
    def force_exd_penalty(self):
        excess = (self.force_applied_w.norm(dim=-1) - self.safe_force_limit).clamp_min(0.0)
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
        return _exp_error(error, 0.35)

    @reward
    def body_yield_velocity(self):
        force = self.force_expected_w.sum(dim=1)
        force_xy = force[:, :2]
        speed = self.asset.data.root_lin_vel_w[:, :2]
        target = self.root_vel_ref_w[:, :2]
        force_active = (force_xy.norm(dim=-1, keepdim=True) > 1.0).float()
        error = (speed - target).norm(dim=-1, keepdim=True)
        return force_active * _exp_error(error, 0.35) + (1.0 - force_active) * _exp_error(speed.norm(dim=-1, keepdim=True), 0.20)

