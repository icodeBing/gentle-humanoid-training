import torch
import wandb
import os
import sys
import hydra
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from scripts.utils.play import play
from scripts.utils.eval import eval
from scripts.utils.interaction_cli import add_interaction_arguments, configure_interaction

FILE_PATH = os.path.dirname(__file__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_path", type=str)
    parser.add_argument("--checkpoint", type=str, help="Local checkpoint path; loads cfg.yaml from the same directory")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("-p", "--play", action="store_true", default=False)
    # whether to override terrain and command
    parser.add_argument("-t", "--terrain", action="store_true", default=False)
    parser.add_argument("-c", "--command", action="store_true", default=False)
    parser.add_argument("-o", "--teleop", action="store_true", default=False)
    
    parser.add_argument("-e", "--export", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("--show-force", action="store_true", default=False,
                        help="Overlay the external force applied to the robot on the recorded video "
                             "and dump it to a csv (only used together with --video).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Evaluation and recording duration in seconds; defaults to the task episode length.")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Override the number of parallel environments for evaluation or playback.")
    parser.add_argument("--settle-force-mode", choices=["random", "off", "spring", "on", "pulse"], default=None,
                        help="Deterministic force schedule for StandingImpedance replay.")
    parser.add_argument("--settle-force-start", type=float, default=0.0,
                        help="Force onset time in seconds for --settle-force-mode pulse.")
    parser.add_argument("--settle-force-end", type=float, default=None,
                        help="Force release time in seconds for --settle-force-mode pulse.")
    parser.add_argument("--settle-force-ramp", type=float, default=0.0,
                        help="Force ramp time in seconds for --settle-force-mode pulse.")
    parser.add_argument("--settle-spring-k", type=float, default=None,
                        help="Fixed replay spring stiffness in N/m; otherwise use the task sampler.")
    parser.add_argument("--settle-spring-offset", type=float, default=None,
                        help="Fixed replay anchor offset in meters; otherwise use the task value.")
    parser.add_argument("--settle-max-force", type=float, default=None,
                        help="Per-interaction-body spring force cap in N; task default is 60 N.")
    parser.add_argument("--settle-safe-force", type=float, default=None,
                        help="Policy safety input for replay, 5..15 N; not a hard force clamp.")
    parser.add_argument("--settle-spring-direction", nargs=3, type=float, default=None,
                        metavar=("DX", "DY", "DZ"),
                        help="Fixed spring direction in nominal body coordinates.")
    parser.add_argument("--settle-force-bodies", nargs="+", default=None,
                        choices=[
                            "left_shoulder_yaw_link", "right_shoulder_yaw_link",
                            "left_wrist_roll_link", "right_wrist_roll_link",
                            "left_hand_mimic", "right_hand_mimic",
                        ],
                        help="Interaction bodies that receive the replay spring; default: all six.")
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=None)
    parser.add_argument("-s", "--success", action="store_true", default=False)  # test success rate
    add_interaction_arguments(parser)
    args = parser.parse_args()
    if args.settle_force_bodies is not None and args.settle_force_mode is None:
        parser.error("--settle-force-bodies requires --settle-force-mode")

    if args.checkpoint is not None:
        checkpoint_path = os.path.abspath(args.checkpoint)
        if not os.path.isfile(checkpoint_path):
            parser.error(f"Checkpoint not found: {checkpoint_path}")
        root = os.path.dirname(checkpoint_path)
        cfg_path = os.path.join(root, "cfg.yaml")
        if not os.path.isfile(cfg_path):
            parser.error(f"cfg.yaml not found beside checkpoint: {cfg_path}")
        checkpoint_name = os.path.basename(checkpoint_path)
    else:
        if args.run_path is None:
            parser.error("provide either --checkpoint for a local run or --run_path for a W&B run")

        api = wandb.Api()
        run = api.run(args.run_path)
        print(f"Loading run {run.name}")

        root = os.path.join(os.path.dirname(__file__), "wandb", run.name)
        os.makedirs(root, exist_ok=True)

        checkpoints = []
        for file in run.files():
            print(file.name)
            if "checkpoint" in file.name:
                checkpoints.append(file)
            elif file.name == "cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "files/cfg.yaml":
                file.download(root, replace=True)
            elif file.name == "config.yaml":
                file.download(root, replace=True)

        if args.iterations is None:
            def sort_by_time(file):
                number_str = file.name[:-3].split("_")[-1]
                return 100000 if number_str == "final" else int(number_str)

            checkpoints.sort(key=sort_by_time)
            if not checkpoints:
                parser.error(f"No checkpoints found in W&B run {args.run_path}")
            checkpoint = checkpoints[-1]
        else:
            checkpoint = next(
                (file for file in checkpoints if file.name == f"checkpoint_{args.iterations}.pt"),
                None,
            )
            if checkpoint is None:
                parser.error(f"Checkpoint iteration {args.iterations} not found in W&B run {args.run_path}")

        print(f"Downloading {checkpoint.name}")
        checkpoint.download(root, replace=True)
        checkpoint_name = checkpoint.name

    cfg_path = os.path.join(root, "cfg.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(root, "files", "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)

    cfg["checkpoint_path"] = os.path.join(root, checkpoint_name)
    cfg["vecnorm"] = "eval"

    if args.teleop:
        cfg["task"]["command"]["teleop"] = True

    if args.task is not None:
        with hydra.initialize(config_path="../cfg", job_name="eval", version_base=None):
            _cfg = hydra.compose(config_name="eval", overrides=[f"task={args.task}"])
        # cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["reward"] = _cfg.task.reward
        cfg["task"]["termination"] = _cfg.task.termination
        cfg["task"]["observation"] = _cfg.task.observation
        cfg["task"]["action"] = _cfg.task.action
        cfg["task"]["randomization"] = _cfg.task.randomization
        cfg["task"]["robot"] = _cfg.task.robot
        if args.terrain:
            cfg["task"]["terrain"] = _cfg.task.terrain
        if args.command:
            cfg["task"]["command"] = _cfg.task.command
        cfg["task"]["flags"] = _cfg.task.flags

    if args.duration is not None:
        import math
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("--duration must be greater than zero")
        step_dt = float(cfg.task.sim.step_dt)
        cfg["task"]["max_episode_length"] = max(1, int(round(args.duration / step_dt)))
        print(
            f"[Info] Evaluation duration: {cfg['task']['max_episode_length'] * step_dt:.3f}s "
            f"({cfg['task']['max_episode_length']} steps at {step_dt:.4f}s/step)."
        )

    if args.num_envs is not None:
        if args.num_envs < 1:
            parser.error("--num-envs must be at least 1")
        cfg.task.num_envs = args.num_envs
        print(f"[Info] Evaluation environments: {args.num_envs}")

    if args.settle_force_mode is not None:
        if args.settle_force_mode == "pulse" and args.settle_force_end is None:
            parser.error("--settle-force-mode pulse requires --settle-force-end")
        if args.settle_force_start < 0 or args.settle_force_ramp < 0:
            parser.error("settle force times must be non-negative")
        if args.settle_force_end is not None and args.settle_force_end <= args.settle_force_start:
            parser.error("--settle-force-end must be greater than --settle-force-start")
        target = str(cfg.task.command.get("_target_", ""))
        if not target.endswith("StandingImpedanceCommand"):
            parser.error("--settle-force-mode requires a G1SETTLE/StandingImpedance checkpoint")
        cfg.eval_settle_force_schedule = dict(
            mode=args.settle_force_mode,
            start_s=args.settle_force_start,
            end_s=args.settle_force_end,
            ramp_s=args.settle_force_ramp,
        )
        if args.settle_spring_k is not None and args.settle_spring_k < 0:
            parser.error("--settle-spring-k must be non-negative")
        if args.settle_spring_offset is not None and args.settle_spring_offset < 0:
            parser.error("--settle-spring-offset must be non-negative")
        if args.settle_max_force is not None and args.settle_max_force <= 0:
            parser.error("--settle-max-force must be positive")
        if args.settle_safe_force is not None and not 5 <= args.settle_safe_force <= 15:
            parser.error("--settle-safe-force must be between 5 and 15 N")
        if args.settle_spring_direction is not None and sum(v * v for v in args.settle_spring_direction) == 0:
            parser.error("--settle-spring-direction must be nonzero")
        cfg.eval_settle_spring = dict(
            stiffness=args.settle_spring_k,
            offset=args.settle_spring_offset,
            max_force=args.settle_max_force,
            direction=args.settle_spring_direction,
            safe_force=args.settle_safe_force,
            body_names=args.settle_force_bodies,
        )
    
    configure_interaction(args, cfg, parser)

    if args.play:
        if not args.success:
            cfg["app"]["headless"] = False
            cfg["task"]["num_envs"] = 16
        cfg["export_policy"] = args.export
        cfg["perf_test"] = False
        play(cfg)
    else:
        if args.video:
            cfg["task"]["num_envs"] = 1
            cfg["eval_render"] = True
            cfg["app"]["enable_cameras"] = True
            cfg["app"]["headless"] = True
            cfg["eval_show_force"] = args.show_force
        elif args.show_force:
            print("[Warn] --show-force only has an effect together with --video.")
        eval(cfg)
    exit(0)

if __name__ == "__main__":
    main()
