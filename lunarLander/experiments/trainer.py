


import argparse
import os
import logging
import gymnasium as gym
import torch
import time
import sys

# Add the submission directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import CustomEnvs



from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

# Mapping of algorithm name to class and device preference
# Algorithm mapping to load the correct class
ALGORITHMS = {
    "ppo": {"class": PPO, "device": "cpu"},
    "dqn": {"class": DQN, "device": "cuda"},
    "a2c": {"class": A2C, "device": "cuda"},
    "sac": {"class": SAC, "device": "cuda"},
    "td3": {"class": TD3, "device": "cuda"},
}

def setup_logging(env_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    safe_env_name = env_name.replace("/", "_")
    log_file = os.path.join(log_dir, f"{safe_env_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_and_record(env_name, model_dir, video_dir, alg_name, timesteps=100_000, seed=42):
    setup_logging(env_name, model_dir)
    logging.info(f"Training environment: {env_name} with algorithm: {alg_name.upper()}")

    # Validate algorithm
    alg_name = alg_name.lower()
    if alg_name not in ALGORITHMS:
        logging.error(f"Unsupported algorithm: {alg_name}. Falling back to PPO.")
        alg_name = "ppo"
    
    model_cls = ALGORITHMS[alg_name]["class"]
    preferred_device = ALGORITHMS[alg_name]["device"]
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda and preferred_device == "cuda" else "cpu"

    logging.info(f"Using device: {device}")


    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_name)
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    model_path = os.path.join(model_dir, f"{env_name}.zip")
    model.save(model_path)
    logging.info(f"Saved model to {model_path}")

    video_name = f"{env_name}_seed{seed}"
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, 
        video_dir, 
        name_prefix=video_name, 
        episode_trigger=lambda x: True)
    obs, _ = env.reset(seed=seed)
    logging.info(f"Recording video to {video_dir}")
    
    done = False
    steps = 0
    while not done:
        logging.info("Taking action in the environment")
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
        if steps == 100_000:
            logging.info("Reached maximum steps, stopping the episode")
            done = True
    logging.info("Episode finished, closing environment")

    env.close()
    logging.info(f"Saved video to {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, help="Gymnasium environment name")
    parser.add_argument("--mod-dir", help="Directory to save model")
    parser.add_argument("--vid-dir", help="Directory to save video")
    parser.add_argument("--tag", help="Tag for the experiment (optional)")
    parser.add_argument("--alg", default="ppo", help="RL algorithm (ppo, dqn, a2c, sac, td3)")
    parser.add_argument("--trn-time", type=int, default=100_000, help="Number of timesteps for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.tag:
        # If a tag is provided, find the directory by tag
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        base_dir = os.path.join("results", f"{timestamp}_{args.tag}")
        model_dir = os.path.join(base_dir, "models")
        video_dir = os.path.join(base_dir, "videos")
    else:
        # If no tag is provided, use the specified directories directly
        model_dir = args.mod_dir
        video_dir = args.vid_dir

    train_and_record(args.env, model_dir, video_dir, args.alg, args.trn_time, args.seed)
    logging.info("Training and recording completed successfully.")
    logging.shutdown()

# Sample command
# python3 experiments/trainer.py --env CustomLunarLander-v0 --tag test --alg ppo --trn-time 100000 --seed 42