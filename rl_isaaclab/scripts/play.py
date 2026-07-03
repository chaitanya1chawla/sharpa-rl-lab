# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil

from isaaclab.app import AppLauncher

# Isaac Lab ViewerCfg defaults (wide shot)
VIEWER_ENV_INDEX_DEFAULT = 0

# Close-up defaults for play / video recording
VIEWER_EYE_CLOSE = (0.5, 0.5, 0.85)
VIEWER_LOOKAT_CLOSE = (0.0, 0.0, 0.58)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--cache", type=str, default=None, help="Cache path.")
parser.add_argument("--load_path", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record an MP4 of the rollout.")
parser.add_argument("--video_length", type=int, default=300, help="Number of simulation steps to record.")
parser.add_argument(
    "--viewer_eye",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=VIEWER_EYE_CLOSE,
    help=f"Camera eye offset (m) relative to viewer origin. Default: {VIEWER_EYE_CLOSE}.",
)
parser.add_argument(
    "--viewer_lookat",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=VIEWER_LOOKAT_CLOSE,
    help=f"Camera look-at offset (m) relative to viewer origin. Default: {VIEWER_LOOKAT_CLOSE}.",
)
parser.add_argument(
    "--viewer_env_index",
    type=int,
    default=VIEWER_ENV_INDEX_DEFAULT,
    help=f"Environment index for viewer origin. Default: {VIEWER_ENV_INDEX_DEFAULT}.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# enable offscreen rendering for headless video capture
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rl_isaaclab.algo.ppo.ppo import PPO
from rl_isaaclab.algo.padapt.padapt import ProprioAdapt
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from rl_isaaclab.wrapper.config_wrapper import ConfigWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/')
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["algorithm"]["max_agent_steps"] = args_cli.max_agent_steps if args_cli.max_agent_steps is not None else agent_cfg["algorithm"]["max_agent_steps"]
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["device"] = args_cli.device if args_cli.device is not None else agent_cfg["device"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["algo"] = args_cli.algorithm if args_cli.algorithm is not None else agent_cfg["algo"]
    agent_cfg["load_path"] = args_cli.load_path if args_cli.load_path is not None else agent_cfg["load_path"]
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, -9.81)
    env_cfg.gravity_curriculum = False
    env_cfg.grasp_cache_path = args_cli.cache if args_cli.cache is not None else env_cfg.grasp_cache_path
    env_cfg.viewer.eye = tuple(args_cli.viewer_eye)
    env_cfg.viewer.lookat = tuple(args_cli.viewer_lookat)
    env_cfg.viewer.env_index = args_cli.viewer_env_index
    print(
        f"[INFO] Viewer: env_index={env_cfg.viewer.env_index}, "
        f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}"
    )
    config = ConfigWrapper(agent_cfg, env_cfg, test=True)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg["algorithm"]["experiment_name"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "play")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_dir}")
        print(f"[INFO] Video kwargs: {video_kwargs}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    agent = eval(agent_cfg["algo"])(env, output_dir=log_dir, full_config=config, create_output_dir=False)
    
    # load the checkpoint
    resume_path = agent_cfg["load_path"]
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    agent.restore_test(resume_path)
    max_steps = args_cli.video_length if args_cli.video else None
    agent.test(max_steps=max_steps)
    if args_cli.video:
        print(f"[INFO] Saved video under: {os.path.join(log_dir, 'videos', 'play')}")

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
