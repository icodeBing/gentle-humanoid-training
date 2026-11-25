import torch
import hydra
import numpy as np
import einops
import time
import sys
from tqdm import tqdm
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

from active_adaptation.learning import ALGOS

import wandb
import logging
from tqdm import tqdm
from scripts.utils.helpers import make_env_policy, evaluate

import os
import datetime
import termcolor

def eval(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    output_dir = cfg.get("eval_output_dir")
    if cfg.get("eval_experiment", False):
        if not output_dir:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs_eval", stamp))
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(output_dir, "summary.json")):
            raise FileExistsError(f"Experiment already exists in {output_dir}; choose a new --output-dir")
        OmegaConf.save(cfg, os.path.join(output_dir, "eval_cfg.yaml"))
        # Dataset sampling and startup randomization occur during environment construction.
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    env, agent, vecnorm, _ = make_env_policy(cfg)
    schedule = cfg.get("eval_settle_force_schedule")
    if schedule is not None:
        command = env.base_env.command_manager
        if not hasattr(command, "set_eval_force_schedule"):
            raise RuntimeError("The loaded command does not support settle force replay schedules")
        command.set_eval_force_schedule(**dict(schedule))
        print(f"[Replay] settle force schedule: {dict(schedule)}")
    spring = cfg.get("eval_settle_spring")
    if spring is not None:
        command = env.base_env.command_manager
        if not hasattr(command, "set_eval_spring"):
            raise RuntimeError("The loaded command does not support settle spring replay overrides")
        command.set_eval_spring(**dict(spring))
        print(f"[Replay] settle spring override: {dict(spring)}")
    
    keys = [
        ("next", "stats"),
        ("next", "done"), 
        ("next", "reward"),
        "value_obs",
        "value_priv",
        "value_adapt",
        "context_expert",
        "context_scale",
        "context_adapt",
        "context_adapt_scale",
        "action_kl",
    ]
    
    policy_eval = agent.get_rollout_policy("eval")
    info, trajs, stats = evaluate(
        env, policy_eval,
        render=cfg.eval_render,
        seed=cfg.seed,
        keys=keys,
        show_force=cfg.get("eval_show_force", False),
        output_dir=output_dir,
        experiment_metadata=dict(checkpoint_path=cfg.checkpoint_path, seed=cfg.seed),
    )
    
    print(termcolor.colored(trajs, "light_yellow"))
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")

    info["task"] = cfg.task.name
    info["algo"] = cfg.algo.name
    info["checkpoint_path"] = cfg.checkpoint_path
    info["argv"] = sys.argv
    print(OmegaConf.to_yaml(info))

    env.close()
    simulation_app.close()
