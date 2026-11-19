"""Evaluation arguments and configuration, without simulator imports."""

import math
import os


def add_interaction_arguments(parser):
    group = parser.add_argument_group("controlled single-policy experiments")
    group.add_argument("--experiment", choices=["spring", "load", "none"], default=None)
    group.add_argument("--safe-force", type=float, default=None, help="Fixed policy safety input, 5..15 N (default: 10 in experiments).")
    group.add_argument("--load-kg", type=float, default=1.0, help="Equivalent payload mass on ONE hand; load experiment only.")
    group.add_argument("--body", default="right_hand_mimic", choices=["left_hand_mimic", "right_hand_mimic", "left_wrist_roll_link", "right_wrist_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link"])
    group.add_argument("--direction", nargs=3, type=float, default=[1., 0., 0.], help="Spring pull direction in WORLD coordinates; normalized automatically.")
    group.add_argument("--spring-k", type=float, default=100., help="Virtual interaction stiffness in N/m; NOT a robot joint gain.")
    group.add_argument("--spring-offset", type=float, default=0.2, help="Anchor offset from the original motion in meters.")
    group.add_argument("--force-start", type=float, default=3.)
    group.add_argument("--force-end", type=float, default=12.)
    group.add_argument("--force-ramp", type=float, default=1., help="Ramp up/down time, in seconds.")
    group.add_argument("--dataset", help="One prepared dataset name/path, e.g. interx_small; default: first in checkpoint config.")
    group.add_argument("--motion-id", type=int, default=0, help="Fixed zero-based motion ID in the chosen dataset.")
    group.add_argument("--motion-start", type=float, default=0., help="Starting offset within the motion, in seconds.")
    group.add_argument("--seed", type=int, default=None)
    group.add_argument("--output-dir", help="Experiment output directory; use a different directory for each condition.")
    group.add_argument("--camera-close", action="store_true", help="Use a closer upper-body camera for videos.")


def configure_interaction(args, cfg, parser):
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.eval_output_dir = os.path.abspath(args.output_dir)
    if args.camera_close:
        cfg.task.viewer.eye = [1.2, -1.8, 0.7]
        cfg.task.viewer.lookat = [0.1, 0., 0.35]
    if args.experiment is None:
        if args.safe_force is not None or args.dataset is not None:
            parser.error("--safe-force / --dataset require --experiment spring, load or none")
        return
    if args.play or args.success:
        parser.error("--experiment uses the offline evaluation path; omit --play/--success")
    target = str(cfg.task.command.get("_target_", ""))
    if not target.endswith("MotionTrackingCommand_impedance") or not cfg.task.command.get("compliance", True):
        parser.error("--experiment requires a Gentle/compliant checkpoint and its matching cfg.yaml")
    values = [args.load_kg, args.spring_k, args.spring_offset, args.force_start, args.force_end, args.force_ramp, args.motion_start, *args.direction]
    if not all(math.isfinite(v) for v in values):
        parser.error("experiment numbers must be finite")
    if args.safe_force is not None and not 5 <= args.safe_force <= 15:
        parser.error("--safe-force must be between 5 and 15 N")
    if min(args.load_kg, args.spring_k, args.spring_offset, args.force_ramp, args.motion_start, args.force_start) < 0 or args.motion_id < 0:
        parser.error("mass, stiffness, offset, times and motion ID must be nonnegative")
    if sum(v*v for v in args.direction) == 0:
        parser.error("--direction must be a nonzero vector")
    if args.force_end <= args.force_start or 2 * args.force_ramp > args.force_end - args.force_start:
        parser.error("require force-end > force-start and 2*force-ramp <= force-end-force-start")
    if args.experiment == "load" and not args.body.endswith("hand_mimic"):
        parser.error("--experiment load requires left_hand_mimic or right_hand_mimic")
    duration = cfg.task.max_episode_length * cfg.task.sim.step_dt
    if args.experiment != "none" and args.force_start >= duration:
        parser.error("--force-start must be shorter than --duration / episode duration")
    if args.force_end >= duration:
        print("[Warn] No full recovery window: use --force-end shorter than --duration.")
    command = cfg.task.command
    selected_dataset = args.dataset or command.dataset.mem_paths[0]
    command.dataset.mem_paths = [selected_dataset]
    command.dataset.path_weights = [1.]
    command.dataset.fix_ds = 0
    command.dataset.fix_motion_id = args.motion_id
    command.motion_start_frame = int(round(args.motion_start / cfg.task.sim.step_dt))
    command.motion_buffer_steps = max(1000, command.motion_start_frame + cfg.task.max_episode_length + 20)
    # Controlled comparisons use the same initial state and no domain randomization.
    cfg.task.randomization = {}
    command.init_noise = {key: 0. for key in command.init_noise}
    command.eval_interaction = dict(
        mode=args.experiment, safe_force=args.safe_force, body=args.body,
        direction=args.direction, stiffness=args.spring_k, offset=args.spring_offset,
        start=args.force_start, end=args.force_end, ramp=args.force_ramp, load_kg=args.load_kg,
    )
    cfg.task.num_envs = 1
    cfg.eval_experiment = True
    cfg.eval_show_force = args.show_force
    cfg.eval_render = bool(args.video)
    cfg.app.headless = True
    cfg.app.enable_cameras = bool(args.video)
    print(f"[Experiment] {args.experiment}, dataset={selected_dataset}, motion={args.motion_id}, fixed safe force={args.safe_force or 10} N")
