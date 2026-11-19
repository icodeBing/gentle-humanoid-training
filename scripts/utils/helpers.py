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
#   * a HUD with per-body force components / magnitude / expected value,
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


def _force_snapshot(base_env, env_index: int | None = None):
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

    indices = command.force_apply_idx_asset
    indices = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    names = [str(command.asset.body_names[i]).replace("_link", "") for i in indices]

    return {
        "applied": applied,                                                  # [M, 3]
        "expected": expected,                                                # [M, 3] | None
        "magnitudes": np.linalg.norm(applied, axis=-1),                      # [M]
        "limit": limit,                                                      # scalar
        "names": names,
    }


def _draw_force_chart(frame: np.ndarray, history, height: int = 170, pad: int = 12):
    """Rolling |F| curve per force body, drawn at the bottom of the frame."""
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

    return np.array(image)


def _overlay_force(frame: np.ndarray, snap, step: int, time_s: float, history):
    frame = np.array(frame, copy=True)
    frame, ymax = _draw_force_chart(frame, history)
    return _overlay_force_text(frame, snap, step, time_s, ymax)


def _write_force_csv(path: str, records):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "time_s", "body",
            "fx_N", "fy_N", "fz_N", "f_mag_N", "f_expected_mag_N", "safe_limit_N",
        ])
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
    force_history = {"mag": [], "limit": []}
    force_max_history = 400

    inference_time = []
    torch.compiler.cudagraph_mark_step_begin()
    with set_exploration_type(exploration_type):
        for i in tqdm(range(env.max_episode_length), miniters=10):
            s = time.perf_counter()
            tensordict_ = policy(tensordict_)
            e = time.perf_counter()
            inference_time.append(e - s)
            if recorder is None:
                tensordict, tensordict_ = env.step_and_maybe_reset(tensordict_)
            else:
                from torchrl.envs.utils import step_mdp
                tensordict = env.step(tensordict_)
                tensordict_ = step_mdp(tensordict, keep_other=True)
                recorder.capture(i, bool(tensordict["next", "terminated"].any()),
                                 bool(tensordict["next", "truncated"].any()))
            trajs.append(tensordict.select(*keys, strict=False).cpu())
            if render:
                frame = env.render("rgb_array")
                snap = _force_snapshot(env.base_env) if show_force else None
                if snap is not None:
                    time_s = (i + 1) * env.step_dt
                    force_history["mag"].append(snap["magnitudes"].copy())
                    force_history["limit"].append(snap["limit"])
                    for key in ("mag", "limit"):
                        if len(force_history[key]) > force_max_history:
                            force_history[key].pop(0)
                    frame = _overlay_force(frame, snap, i, time_s, force_history)
                    for m, name in enumerate(snap["names"]):
                        fx, fy, fz = (float(v) for v in snap["applied"][m])
                        exp = float(np.linalg.norm(snap["expected"][m])) if snap["expected"] is not None else float("nan")
                        force_records.append([
                            i, round(time_s, 4), name,
                            round(fx, 4), round(fy, 4), round(fz, 4),
                            round(float(snap["magnitudes"][m]), 4),
                            round(exp, 4),
                            round(float(snap["limit"]), 4),
                        ])
                if recorder is not None:
                    frame = recorder.overlay(frame)
                frames.append(frame)
            if recorder is not None and tensordict["next", "done"].any():
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
        peak = max(row[6] for row in force_records)
        info["force/peak"] = peak
        print(f"force log saved to {os.path.abspath(csv_path)} (peak |F| = {peak:.2f} N)")

    info["episode_cnt"] = episode_cnt
    return dict(sorted(info.items())), trajs, stats
