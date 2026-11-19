"""Controlled, single-policy interactions, enabled only by the evaluation CLI."""

import torch


def pulse_scale(time_s, start, end, ramp):
    """Continuous trapezoidal pulse, including a ramp before release."""
    if time_s < start or time_s >= end:
        return 0.0
    if ramp == 0:
        return 1.0
    return max(0.0, min(1.0, (time_s - start) / ramp, (end - time_s) / ramp))


class EvaluationInteraction:
    def __init__(self, command, config):
        self.command = command
        self.config = dict(config)
        self.config["direction"] = [float(value) for value in self.config["direction"]]
        if not command.compliance:
            raise ValueError("Controlled interactions require a Gentle/compliant checkpoint.")
        if command.num_envs != 1:
            raise ValueError("Controlled interactions require num_envs=1.")
        names = [command.asset.body_names[i] for i in command.force_apply_idx_asset.tolist()]
        self.body = names.index(self.config["body"])
        self.direction = torch.tensor(self.config["direction"], device=command.device, dtype=torch.float32)
        self.direction /= self.direction.norm()
        force_vector = self.config.get("force_vector", [0., 0., 0.])
        self.force_vector = torch.tensor(force_vector, device=command.device, dtype=torch.float32)
        if self.config["mode"] == "force":
            self.direction = self.force_vector / self.force_vector.norm().clamp_min(1e-6)
        self.elapsed = 0.0
        self.applied_scale = 0.0
        self.load_w = torch.zeros_like(command.force_applied_w)
        self.total_w = torch.zeros_like(command.force_applied_w)
        self.reference_force_w = torch.zeros_like(command.force_applied_w)
        self.reference_w = torch.zeros_like(command.force_applied_w)
        self.applied_reference_w = torch.zeros_like(command.force_applied_w)
        self.applied_origin_w = torch.zeros_like(command.force_applied_w)
        self.safe_force = self.config["safe_force"]
        if self.safe_force is not None:
            lo, hi = command.force_safe_bounds
            if not lo <= self.safe_force <= hi:
                raise ValueError(f"--safe-force must be within the trained range [{lo}, {hi}] N")
        self.reset(torch.arange(command.num_envs, device=command.device))

    def reset(self, env_ids):
        self.elapsed = 0.0
        self.applied_scale = 0.0
        self.load_w[env_ids] = 0
        self.total_w[env_ids] = 0
        self.reference_force_w[env_ids] = 0
        # Even without an override, controlled tests use a fixed 10 N default.
        value = self.command.force_safe_default if self.safe_force is None else self.safe_force
        self.command.force_safe_limit_tl.reset(env_ids, value=value)
        self.command.force_apply_buffer[env_ids] = 0
        self.command.torque_apply_buffer[env_ids] = 0

    def prepare_targets(self):
        c = self.command
        p = self.config
        scale = pulse_scale(self.elapsed, p["start"], p["end"], p["ramp"])
        c.force_enable.zero_()
        c.force_pos_delta.zero_()
        c.force_kp_scaled.zero_()
        c.force_expected_w.zero_()
        c.force_expected_b.zero_()
        self.reference_force_w.zero_()
        # A trajectory-following world-frame anchor is identical across thresholds.
        self.reference_w[:] = c._motion.body_pos_w[:, 0, c.force_apply_idx_motion] + c.env.scene.env_origins[:, None]
        origin = self.reference_w.clone()
        origin[:, self.body] += self.direction * p["offset"]
        if p["mode"] == "spring":
            c.force_kp_scaled[:, self.body] = p["stiffness"] * scale
            c.force_enable[:, self.body] = scale > 0
            interaction_force = None
        elif p["mode"] == "load":
            # Keep legacy mass-based tests while synchronizing their reference dynamics.
            self.reference_force_w[:, self.body, 2] = -p["load_kg"] * 9.81 * scale
            interaction_force = self.reference_force_w
        elif p["mode"] == "force":
            self.reference_force_w[:, self.body] = self.force_vector * scale
            interaction_force = self.reference_force_w
        else:
            interaction_force = None
        # Preserve the policy's native admittance model for controlled comparisons.
        directions = -self.direction.expand_as(origin)
        c.force_update_origin_and_target(
            origin_w=origin, direction_w=directions, interaction_force_w=interaction_force
        )

    def apply(self, substep):
        c = self.command
        p = self.config
        if substep == 0:
            self.applied_scale = pulse_scale(self.elapsed, p["start"], p["end"], p["ramp"])
            self.applied_reference_w.copy_(self.reference_w)
            self.applied_origin_w.copy_(c.force_origin_w)
            self.elapsed += c.env.step_dt
        # Native spring calculation updates the applied force at each physics substep.
        c.force_apply(substep)
        self.load_w.zero_()
        if p["mode"] == "load":
            self.load_w[:, self.body, 2] = -p["load_kg"] * 9.81 * self.applied_scale
        elif p["mode"] == "force":
            self.load_w[:, self.body] = self.force_vector * self.applied_scale
        self.total_w.copy_(c.force_applied_w + self.load_w)
        c.force_apply_buffer[:, c.force_apply_idx_asset] = self.total_w
        # No torso counter-force: in particular, do not silently compensate mg.
        c.torque_apply_buffer.zero_()
