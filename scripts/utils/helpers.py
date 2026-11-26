import torch
import torch.nn as nn
import hydra
import numpy as np
import time
import wandb
import logging
import os
import csv
import datetime

from typing import Sequence
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.envs.transforms import VecNorm

from termcolor import colored
from collections import OrderedDict
import imageio
from omegaconf import OmegaConf, DictConfig
import active_adaptation.learning
from active_adaptation.utils.wandb import parse_checkpoint_path
import active_adaptation

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


class ObsNorm(ModBase):
    def __init__(self, in_keys, out_keys, locs, scales):
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        
        self.loc = nn.ParameterDict({k: nn.Parameter(locs[k]) for k in in_keys})
        self.scale = nn.ParameterDict({k: nn.Parameter(scales[k]) for k in out_keys})
        self.requires_grad_(False)

    def forward(self, tensordict: TensorDictBase):
        for in_key, out_key in zip(self.in_keys, self.out_keys):
            obs = tensordict.get(in_key, None)
            if obs is not None:
                loc = self.loc[in_key]
                scale = self.scale[out_key]
                tensordict.set(out_key, (obs - loc) / scale)
        return tensordict
    
    @classmethod
    def from_vecnorm(cls, vecnorm: VecNorm, keys):
        in_keys = []
        out_keys = []
        for in_key, out_key in zip(vecnorm.in_keys, vecnorm.out_keys):
            if in_key in keys:
                in_keys.append(in_key)
                out_keys.append(out_key)
        return cls(
            in_keys=in_keys,
            out_keys=out_keys,
            locs=vecnorm.loc,
            scales=vecnorm.scale
        )


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str], device: torch.device):
        self.in_keys = in_keys
        self.device = device
        self._stats = TensorDict({key: torch.tensor([0.], device=device) for key in in_keys}, [1])
        self._episodes = torch.tensor(0, device=device)

    def add(self, tensordict: TensorDictBase) -> int:
        next_tensordict = tensordict["next"]
        done = next_tensordict["done"]
        if done.any():
            done = done.squeeze(-1)
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats = self._stats + next_tensordict[done].sum(dim=0)
            self._episodes += done.sum()
        return len(self)
    
    def pop(self):
        stats = self._stats / self._episodes
        self._stats.zero_()
        self._episodes.zero_()
        return stats.cpu()

    def __len__(self):
        return self._episodes.item()

import torch.distributed as dist
from torchrl._utils import _append_last
from torchrl.envs.transforms.transforms import _sum_left

import active_adaptation as aa

def make_env_policy(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    from active_adaptation.envs import SimpleEnv
    from torchrl.envs.transforms import TransformedEnv, Compose, InitTracker, VecNorm, StepCounter
    aa.print("import SimpleEnv done")
    base_env = SimpleEnv(cfg.task)
    aa.print("SimpleEnv done")

    if cfg.checkpoint_path is not None and aa.is_main_process():
        checkpoint_path = parse_checkpoint_path(cfg.checkpoint_path)
        aa.print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=False)
    else:
        state_dict = {}

    if aa.is_distributed():
        state_list = [state_dict]
        dist.broadcast_object_list(state_list, src=0)
        state_dict = state_list[0] or {}
    aa.print("load checkpoint done")
    
    policy_in_keys = cfg.algo.get("in_keys", ["policy", "priv"])

    for obs_group_key in list(cfg.task.observation.keys()):
        if (
            obs_group_key not in policy_in_keys
            and not obs_group_key.endswith("_")
        ):
            print(colored(f"[Warn] Obs group '{obs_group_key}' not used by policy in_keys; keeping config unchanged.", "yellow"))
    
    obs_keys = [
        key for key, spec in base_env.observation_spec.items(True, True) 
        if not (spec.dtype == bool or key.endswith("_"))
    ]
    transform = Compose(InitTracker(), StepCounter())

    assert cfg.vecnorm in ("train", "eval", None)
    print(colored(f"[Info]: create VecNorm for keys: {obs_keys}", "green"))
    vecnorm = VecNorm(obs_keys, decay=0.9999)
    vecnorm(base_env.fake_tensordict())

    if "vecnorm" in state_dict.keys():
        print(colored("[Info]: Load VecNorm from checkpoint.", "green"))
        vecnorm.load_state_dict(state_dict["vecnorm"])
    if cfg.vecnorm == "train":
        print(colored("[Info]: Updating obervation normalizer.", "green"))
        transform.append(vecnorm)
    elif cfg.vecnorm == "eval":
        print(colored("[Info]: Not updating obervation normalizer.", "green"))
        transform.append(vecnorm.to_observation_norm())
    elif cfg.vecnorm is not None:
        raise ValueError
    aa.print("create VecNorm done")

    env = TransformedEnv(base_env, transform)
    env.set_seed(cfg.seed)
    aa.print("TransformedEnv done")
    
    # setup policy
    policy_cls = hydra.utils.get_class(cfg.algo._target_)
    active_adaptation.print(f"Creating policy {policy_cls} on device {base_env.device}")
    policy = policy_cls(
        cfg.algo,
        env.observation_spec, 
        env.action_spec, 
        env.reward_spec,
        device=base_env.device,
        env=env
    )
    aa.print("policy done")
    
    if "policy" in state_dict.keys():
        print(colored("[Info]: Load policy from checkpoint.", "green"))
        policy.load_state_dict(state_dict["policy"])
    
    if cfg.checkpoint_path is not None:
        policy.broadcast_parameters([vecnorm])

    primer = policy.make_tensordict_primer()

    if primer is not None:
        print(colored(f"[Info]: Add TensorDictPrimer {primer}.", "green"))
        transform.append(primer)
    env = TransformedEnv(env.base_env, transform)

    return env, policy, vecnorm, primer


from torchrl.envs import TransformedEnv, ExplorationType, set_exploration_type
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# External force visualization
#
# `MotionTrackingCommand_impedance` pushes the robot hands/arms with a virtual spring at
# every physics substep (see `force_apply`). The force that is actually handed to PhysX is
# stored in `command_manager.force_applied_w`
# ([num_envs, num_force_bodies, 3], world frame, Newton).
#
# The helpers below make that quantity visible in a headless `--video` replay:
#   * projected per-body force arrows, a HUD with force components / magnitude,
#   * a rolling |F| curve per body at the bottom of the frame,
#   * a csv next to the video for offline plotting.
# --------------------------------------------------------------------------------------

_FORCE_COLORS = [
    (255, 90, 90),
    (90, 170, 255),
    (90, 230, 140),
    (255, 200, 70),
    (210, 130, 255),
    (80, 220, 220),
]


def _force_reference_snapshot(base_env, env_index: int | None = None):
    """Copy the prepared reference BEFORE physics can advance its schedule."""
    command = base_env.command_manager
    if env_index is None:
        env_index = int(getattr(base_env, "lookat_env_i", 0))
    result = {}
    for key, attr in [("expected", "force_expected_w"), ("interact", "force_interact_w")]:
        value = getattr(command, attr, None)
        result[key] = None if value is None else value[env_index].detach().float().cpu().numpy().copy()
    if hasattr(command, "force_safe_limit_tl"):
        result["limit"] = float(command.force_safe_limit_tl.current[env_index].reshape(-1)[0])
    elif hasattr(command, "safe_force_limit"):
        result["limit"] = float(command.safe_force_limit[env_index].reshape(-1)[0])
    else:
        result["limit"] = float("nan")
    return result


def _force_snapshot(base_env, env_index: int | None = None, reference=None):
    """Read the force the environment currently applies to the robot.

    Returns ``None`` when the task does not apply external forces (e.g. locomotion).
    """
    command = getattr(base_env, "command_manager", None)
    if command is None or not hasattr(command, "force_applied_w"):
        return None

    if env_index is None:
        env_index = int(getattr(base_env, "lookat_env_i", 0))

    applied = command.force_applied_w[env_index].detach().float().cpu().numpy()
    interaction = getattr(command, "eval_interaction", None)
    if interaction is not None:
        applied = interaction.total_w[env_index].detach().float().cpu().numpy()

    expected = None
    if hasattr(command, "force_expected_w"):
        expected = command.force_expected_w[env_index].detach().float().cpu().numpy()
    if interaction is not None and interaction.config["mode"] == "load":
        expected = None

    limit = float("nan")
    if hasattr(command, "force_safe_limit_tl"):
        limit = float(command.force_safe_limit_tl.current[env_index].reshape(-1)[0])
    elif hasattr(command, "safe_force_limit"):
        limit = float(command.safe_force_limit[env_index].reshape(-1)[0])

    # Standing before_update prepares the NEXT interval after physics. Pair the
    # force just applied with the saved reference from ITS interval, particularly
    # at load onset/release. Do not pair it with the next interval's reference.
    interact = None
    if reference is not None:
        expected = reference["expected"]
        interact = reference["interact"]
        limit = reference["limit"]

    indices = command.force_apply_idx_asset
    indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    names = [str(command.asset.body_names[i]).replace("_link", "") for i in indices]
    positions = command.asset.data.body_pos_w[env_index, indices].detach().float().cpu().numpy()

    snap = {
        "applied": applied,                                                  # [M, 3]
        "positions": positions,                                              # [M, 3], world frame
        "expected": expected,                                                # [M, 3] | None
        "interact": interact,                                                # last virtual substep, world frame
        "magnitudes": np.linalg.norm(applied, axis=-1),                      # [M]
        "limit": limit,                                                      # scalar
        "names": names,
    }
    if getattr(command, "eval_force_schedule", None) is not None:
        snap["root_velocity"] = command.asset.data.root_lin_vel_w[env_index].detach().float().cpu().numpy()
        snap["root_velocity_ref"] = command.root_vel_ref_w[env_index].detach().float().cpu().numpy()
    return snap


def _draw_force_chart(frame: np.ndarray, history, height: int = 170, pad: int = 12):
    """Rolling |F| curve per force body, drawn at the bottom of the frame."""
    frame = np.array(frame, copy=True)
    mag = np.stack(history["mag"], axis=0)                                   # [T, M]
    if mag.shape[0] < 2:
        return frame, max(1.0, float(mag.max()))

    limit = np.asarray(history["limit"], dtype=np.float32)                   # [T]
    has_limit = np.isfinite(limit).any()

    limit_values = limit[np.isfinite(limit)]
    ymax = max(1.0, float(mag.max()) * 1.15)
    if limit_values.size:
        ymax = max(ymax, float(limit_values.max()) * 1.15)

    h, w = frame.shape[:2]
    height = int(min(height, h // 3))
    top = h - height
    frame[top:h] = (frame[top:h].astype(np.float32) * 0.35).astype(np.uint8)
    frame[top, pad:w - pad] = (90, 90, 90)

    plot_top, plot_bottom = top + pad, h - pad - 1
    xs = np.linspace(pad, w - pad - 1, mag.shape[0]).astype(np.int32)
    cols = np.arange(pad, w - pad - 1, dtype=np.int32)

    def value_to_row(values):
        values = np.clip(np.nan_to_num(values) / ymax, 0.0, 1.0)
        return (plot_bottom - values * (plot_bottom - plot_top)).round().astype(np.int32)

    def draw_curve(rows, color, thickness: int = 1):
        rows = np.interp(cols, xs, rows).round().astype(np.int32)
        for dy in range(-thickness, thickness + 1):
            frame[np.clip(rows + dy, top + 1, h - 1), cols] = color

    if has_limit:
        limit_rows = value_to_row(limit)
        for i in range(0, len(cols) - 4, 8):
            frame[np.clip(np.interp(cols[i:i + 4], xs, limit_rows).round().astype(np.int32), top + 1, h - 1),
                  cols[i:i + 4]] = (130, 130, 130)

    for m in range(mag.shape[1]):
        draw_curve(value_to_row(mag[:, m]), _FORCE_COLORS[m % len(_FORCE_COLORS)], thickness=1)

    return frame, ymax


def _draw_force_arrows(frame: np.ndarray, snap):
    """Project applied world-frame forces at their body locations into RGB."""
    try:
        from PIL import Image, ImageDraw
        from pxr import UsdGeom
        import omni.usd
    except ImportError:
        return frame

    camera = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/OmniverseKit_Persp"))
    frustum = camera.GetCamera().frustum
    view = frustum.ComputeViewMatrix()
    matrix = view * frustum.ComputeProjectionMatrix()
    h, w = frame.shape[:2]

    def project(point):
        from pxr import Gf
        p = Gf.Vec3d(*(float(v) for v in point))
        if view.Transform(p)[2] >= 0:
            return None
        ndc = matrix.Transform(p)
        if not all(np.isfinite(v) for v in ndc) or max(abs(ndc[0]), abs(ndc[1])) > 3:
            return None
        return np.array(((ndc[0] + 1) * w / 2, (1 - ndc[1]) * h / 2), dtype=np.float32)

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    for m, (origin, force) in enumerate(zip(snap["positions"], snap["applied"])):
        magnitude = float(np.linalg.norm(force))
        if magnitude < 0.1:
            continue
        direction = force / magnitude
        # Keep arrows legible across the configured force range without letting
        # a saturated spring obscure the robot.
        arrow_length_m = min(0.012 * magnitude, 0.40)
        start = project(origin)
        end = project(origin + direction * arrow_length_m)
        if start is None or end is None:
            continue
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 3.0:
            continue
        color = _FORCE_COLORS[m % len(_FORCE_COLORS)]
        unit = delta / length
        head_length = min(11.0, max(6.0, length * 0.22))
        normal = np.array((-unit[1], unit[0]), dtype=np.float32)
        base = end - unit * head_length
        left = base + normal * head_length * 0.55
        right = base - normal * head_length * 0.55
        p0, p1, p2 = [tuple(np.round(p).astype(int)) for p in (end, left, right)]
        draw.line([tuple(np.round(start).astype(int)), tuple(np.round(base).astype(int))], fill=color, width=4)
        draw.polygon([p0, p1, p2], fill=color)

    return np.array(image, copy=True)


def _overlay_force_text(frame: np.ndarray, snap, step: int, time_s: float, ymax: float):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    lines, colors = [], []

    def add(text, color=(255, 255, 255)):
        lines.append(text)
        colors.append(tuple(int(c) for c in color))

    add(f"external force  t={time_s:6.2f}s  step={step:4d}  chart ymax={ymax:5.1f}N")
    for m, name in enumerate(snap["names"]):
        fx, fy, fz = snap["applied"][m]
        text = f"{name:<22s} ({fx:+6.2f},{fy:+6.2f},{fz:+6.2f})N  |F|={snap['magnitudes'][m]:6.2f}N"
        if snap["expected"] is not None:
            text += f"  |Fexp|={float(np.linalg.norm(snap['expected'][m])):6.2f}N"
        add(text, _FORCE_COLORS[m % len(_FORCE_COLORS)])
    add(f"policy safety input = {snap['limit']:5.2f}N (not a hard force clamp)", (200, 200, 200))
    if "root_velocity" in snap:
        actual = snap["root_velocity"]
        target = snap["root_velocity_ref"]
        add(f"root velocity WORLD xy: actual=({actual[0]:+.3f},{actual[1]:+.3f})  "
            f"reference=({target[0]:+.3f},{target[1]:+.3f}) m/s")
    add("arrows: applied-force direction; length = 0.012 m/N (capped at 0.40 m)", (220, 220, 220))

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    line_height = 17
    try:
        widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    except Exception:
        widths = [len(line) * 7 for line in lines]
    panel_w = max(widths) + 20
    panel_h = len(lines) * line_height + 12
    draw.rectangle([6, 6, 6 + panel_w, 6 + panel_h], fill=(0, 0, 0, 150))

    y = 12
    for line, color in zip(lines, colors):
        draw.text((14, y), line, font=font, fill=color + (255,))
        y += line_height

    return np.array(image, copy=True)


def _overlay_force(frame: np.ndarray, snap, step: int, time_s: float, history):
    frame = np.array(frame, copy=True)
    frame = _draw_force_arrows(frame, snap)
    frame, ymax = _draw_force_chart(frame, history)
    return _overlay_force_text(frame, snap, step, time_s, ymax)


def _force_record_rows(snap, step: int, step_dt: float, physics_dt: float):
    """Log world vectors; reference timestamps denote virtual integration time.

    Applied force is from the final physics substep, interact is the input to
    the final virtual substep, expected is evaluated at its final state. These
    are paired within one control interval, not claimed to be sensor samples
    from an identical physical instant. Standing uses four virtual substeps.
    """
    interval_start = step * step_dt
    interval_end = (step + 1) * step_dt
    rows = []
    for m, name in enumerate(snap["names"]):
        applied = snap["applied"][m]
        expected = (snap["expected"][m] if snap["expected"] is not None else np.full(3, np.nan))
        interact = (snap["interact"][m] if snap["interact"] is not None else np.full(3, np.nan))
        row = dict(step=step, time_s=round(interval_end, 6), body=name,
                   fx_N=float(applied[0]), fy_N=float(applied[1]), fz_N=float(applied[2]),
                   f_mag_N=float(np.linalg.norm(applied)),
                   f_expected_mag_N=float(np.linalg.norm(expected)), safe_limit_N=snap["limit"])
        for label, vector in [("f_applied", applied), ("f_interact", interact), ("f_expected", expected)]:
            row.update({f"{label}_{axis}_N": float(value) for axis, value in zip("xyz", vector)})
        row.update(f_applied_mag_N=float(np.linalg.norm(applied)),
                   f_interact_mag_N=float(np.linalg.norm(interact)),
                   applied_expected_error_N=float(np.linalg.norm(applied - expected)),
                   applied_interact_error_N=float(np.linalg.norm(applied - interact)),
                   interval_start_s=round(interval_start, 6),
                   applied_sample_time_s=round(interval_end - physics_dt, 6),
                   expected_reference_time_s=(round(interval_end, 6)
                                              if snap["interact"] is not None else float("nan")),
                   interact_reference_time_s=(round(interval_start + 3 * physics_dt, 6)
                                              if snap["interact"] is not None else float("nan")))
        rows.append(row)
    return rows


def _write_force_csv(path: str, records):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@torch.inference_mode()
def evaluate(
    env: TransformedEnv,
    policy: torch.nn.Module,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MODE,
    render=False,
    keys=[("next", "stats")],
    show_force: bool = False,
    output_dir: str | None = None,
    experiment_metadata: dict | None = None,
):
    """
    Evaluate the policy on the environment, selecting `keys` from the trajectory.
    If `render` is True, record and save the video.
    If `show_force` is True, draw the external force applied to the robot on the video
    and dump it to a csv next to the video.
    """
    keys = set(keys)
    keys.add(("next", "done"))

    env.eval()
    env.set_seed(seed)

    tensordict_ = env.reset()
    command = env.base_env.command_manager
    interaction = getattr(command, "eval_interaction", None)
    settle_replay = getattr(command, "eval_force_schedule", None) is not None
    single_episode = interaction is not None or settle_replay
    if settle_replay:
        # Training reset randomizes episode ages and leaves the command awaiting
        # its first update. Replay needs a full episode and initialized springs
        # before the first physics step (including when the schedule is off).
        env.base_env.episode_length_buf.zero_()
        command.eval_force_step.zero_()
        command.before_update()
        command.update()
        print("[Replay] Starting at episode step zero; recording stops at the first episode end.")
    recorder = None
    if interaction is not None:
        from scripts.utils.interaction_metrics import InteractionRecorder
        if output_dir is None:
            raise ValueError("Controlled evaluation requires an output directory")
        recorder = InteractionRecorder(env.base_env, output_dir, experiment_metadata)
        # BaseEnv normally randomizes episode ages for training; start this test at zero.
        env.base_env.episode_length_buf.zero_()
        command.t.sub_(1)
        command.before_update()
        command.update()
        available = (int(command.lengths[0]) - int(command.t[0]) - 1) * env.step_dt
        print(f"[Experiment] Available motion duration: {available:.2f}s; recording stops at the first episode end.")
    trajs = []
    frames = []

    force_records = []
    motion_records = []
    force_history = {"mag": [], "limit": []}
    force_max_history = 400

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            force_reference = (_force_reference_snapshot(env.base_env)
                               if settle_replay and render and show_force else None)
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)
            if not single_episode:
                tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            else:
                from torchrl.envs.utils import step_mdp
                tensordict = env.step(tensordict_)
                tensordict_ = step_mdp(tensordict, keep_other=True)
                if recorder is not None:
                    recorder.capture(i, bool(tensordict["next", "terminated"].any()),
                                     bool(tensordict["next", "truncated"].any()))
            if settle_replay:
                index = int(getattr(env.base_env, "lookat_env_i", 0))
                data = command.asset.data
                row = dict(step=i, time_s=round((i + 1) * env.step_dt, 4),
                           terminated=int(tensordict["next", "terminated"][index].any()),
                           truncated=int(tensordict["next", "truncated"][index].any()))
                for label, tensor, unit in [
                    ("root_pos", data.root_pos_w, "m"),
                    ("root_vel", data.root_lin_vel_w, "m_s"),
                    ("root_vel_ref", command.root_vel_ref_w, "m_s"),
                    ("root_ang_vel", data.root_ang_vel_w, "rad_s"),
                ]:
                    values = tensor[index].detach().float().cpu().tolist()
                    row.update({f"{label}_{axis}_{unit}": float(value)
                                for axis, value in zip("xyz", values)})
                q = data.root_quat_w[index].detach().float().cpu().numpy()
                row["root_yaw_rad"] = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                                        1 - 2 * (q[2] ** 2 + q[3] ** 2)))
                motion_records.append(row)
            trajs.append(tensordict.select(*keys, strict=False).cpu())
            if render:
                frame = env.render("rgb_array")
                snap = _force_snapshot(env.base_env, reference=force_reference) if show_force else None
                if snap is not None:
                    time_s = (i + 1) * env.step_dt
                    force_history["mag"].append(snap["magnitudes"].copy())
                    force_history["limit"].append(snap["limit"])
                    for key in ("mag", "limit"):
                        if len(force_history[key]) > force_max_history:
                            force_history[key].pop(0)
                    frame = _overlay_force(frame, snap, i, time_s, force_history)
                    force_records.extend(_force_record_rows(
                        snap, i, env.step_dt, env.base_env.physics_dt))
                if recorder is not None:
                    frame = recorder.overlay(frame)
                frames.append(frame)
            if single_episode and tensordict["next", "done"].any():
                if settle_replay:
                    print(f"[Replay] Stopped at {(i + 1) * env.step_dt:.2f}s: "
                          f"terminated={bool(tensordict['next', 'terminated'].any())}, "
                          f"truncated={bool(tensordict['next', 'truncated'].any())}")
                break
    if recorder is not None:
        recorder.save()
    inference_time = np.mean(inference_time[5:] or inference_time)
    print(f"Average inference time: {inference_time:.4f} s")

    trajs: TensorDictBase = torch.stack(trajs, dim=1)
    done = trajs.get(("next", "done"))
    episode_cnt = len(done.nonzero())
    first_done = torch.argmax(done.long(), dim=1).cpu()
    first_done = torch.where(done.any(dim=1).cpu(), first_done, torch.full_like(first_done, done.shape[1]-1))

    def take_first_episode(tensor: torch.Tensor):
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    info = {}
    stats = {}
    compute_std_for = ["return", "survival"]
    for k, v in trajs["next", "stats"].items(True, True):
        v = take_first_episode(v)
        key = "eval/" + ("/".join(k) if isinstance(k, tuple) else k)
        stats[key] = v
        info[key] = torch.mean(v.float()).item()
        if k in compute_std_for:
            info[key + "_std"] = torch.std(v.float(), unbiased=False).item()

    # log video
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S-%f")
    video_dir = output_dir or os.path.join(os.path.dirname(__file__), "..", "videos")
    if len(frames):
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"recording-{time_str}.mp4")
        fps = int(1 / env.step_dt)
        try:
            imageio.mimwrite(video_path, frames, fps=fps)
        except Exception:
            # Fallback: ensure frames are numpy arrays
            video_array = np.stack(frames)
            imageio.mimwrite(video_path, list(video_array), fps=fps)
        print(f"video saved to {os.path.abspath(video_path)}")

    if len(force_records):
        os.makedirs(video_dir, exist_ok=True)
        csv_path = os.path.join(video_dir, f"force-{time_str}.csv")
        _write_force_csv(csv_path, force_records)
        peak = max(row["f_mag_N"] for row in force_records)
        info["force/peak"] = peak
        print(f"force log saved to {os.path.abspath(csv_path)} (peak |F| = {peak:.2f} N)")

    if motion_records:
        os.makedirs(video_dir, exist_ok=True)
        motion_path = os.path.join(video_dir, f"motion-{time_str}.csv")
        with open(motion_path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(motion_records[0]))
            writer.writeheader()
            writer.writerows(motion_records)
        print(f"root motion log saved to {os.path.abspath(motion_path)}")

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats
