"""
Model Runner for Reinforcement Learning with Safety Shields

This module provides functionality to run trained RL models and record videos
of their performance, with optional safety shielding using formal verification
techniques. The shield ensures that the agent's actions satisfy safety properties
while maintaining task performance.
"""

import os
import argparse
import logging
import gymnasium as gym
import sys

# Add the submission directory to the Python path for marg
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# Add the lunarLander directory to the Python path for CustomEnvs
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import CustomEnvs
import torch
import pprint as pp
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
import subprocess

# Local imports for graph-based shield generation and MARG framework
import landerToGraph as l2g
from marg.shield.shield import Shield
from marg.shield.solver.Parity import compute_template_in_one_player_graph as compute_template


# Algorithm mapping to load the correct reinforcement learning class
# Each algorithm specifies its class and preferred device (CPU/GPU)
ALGORITHMS = {
    "ppo": {"class": PPO, "device": "cpu"},
    "dqn": {"class": DQN, "device": "cuda"},
    "a2c": {"class": A2C, "device": "cuda"},
    "sac": {"class": SAC, "device": "cuda"},
    "td3": {"class": TD3, "device": "cuda"},
}

# Shield enforcement parameters - control how strictly the shield intervenes
ENFORCEMENT_PARAMETER = 0.3  # Standard enforcement for liveness properties
RECOVERY_ENFORCEMENT_PARAMETER = 0.2  # Not needed for safety-only shield
LOGGING_LEVEL = logging.INFO  # Set the default logging level

def find_directory_by_tag(tag):
    """
    Find the most recent training run directory based on a tag.
    
    Searches for directories in './results' that end with the given tag
    and returns the one with the latest timestamp.
    
    Args:
        tag (str): The tag suffix to search for (e.g., 'submission', 'test')
        
    Returns:
        str: Path to the most recent directory matching the tag
        
    Raises:
        FileNotFoundError: If no directory with the given tag is found
    """
    candidates = []
    for item in os.listdir('./results'):
        item_path = os.path.join('./results', item)
        if os.path.isdir(item_path) and item.endswith(f"_{tag}"):
            # Extract timestamp (assumes format: <timestamp>_<tag>)
            parts = item.split('_')
            if len(parts) >= 2:
                timestamp = parts[0]
                candidates.append((timestamp, item_path))
    if not candidates:
        raise FileNotFoundError(f"No directory ending with '_{tag}' found.")
    # Sort by timestamp descending and return the latest
    latest = max(candidates, key=lambda x: x[0])
    return latest[1]


def setup_logging(log_dir, log_file_name):
    """
    Configure logging to write to a file in the specified directory.
    
    Clears any existing handlers and sets up file-based logging with timestamps.
    Creates necessary directories if they don't exist.
    
    Args:
        log_dir (str): Directory where log files should be stored
        log_file_name (str): Name of the log file to create
    """       
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    os.makedirs(log_dir, exist_ok=True)

    video_log_dir = os.path.join(log_dir, "videos")
    os.makedirs(video_log_dir, exist_ok=True) 

    log_file_name = os.path.join(video_log_dir, log_file_name)
    # if the log file already exists, clear it
    if os.path.exists(log_file_name):
        os.remove(log_file_name)

    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[#logging.StreamHandler(),
                  logging.FileHandler(log_file_name)]
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def load_model(model_path, env, alg_name):
    """
    Load a trained RL model from disk with the specified algorithm.
    
    Automatically selects the appropriate device (CPU/GPU) based on availability
    and algorithm preferences. Falls back to PPO if the specified algorithm
    is not supported.
    
    Args:
        model_path (str): Path to the saved model file
        env: Gymnasium environment (used for model initialization)
        alg_name (str): Name of the RL algorithm (ppo, dqn, a2c, sac, td3)
        
    Returns:
        Loaded RL model instance
        
    Raises:
        RuntimeError: If model loading fails
    """

    if alg_name not in ALGORITHMS:
        logging.error(f"Unsupported algorithm: {alg_name}. Falling back to PPO.")
        alg_name = "ppo"
    
    model_cls = ALGORITHMS[alg_name]["class"]
    preferred_device = ALGORITHMS[alg_name]["device"]
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda and preferred_device == "cuda" else "cpu"

    logging.info(f"Using device: {device}")
    try:
        model = model_cls.load(model_path, env=env, device=device)
        logging.info(f"Model loaded with algorithm class: {model_cls.__name__}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path} with known algorithms. Error: {e}")

def get_action_distribution(model, obs):
    """
    Extract action probabilities from the RL model's policy.
    
    This function works with different types of policies (PPO, A2C with action_dist,
    and DQN with Q-values) to obtain a probability distribution over actions.
    
    Args:
        model: Trained RL model with a policy
        obs: Current observation from the environment
        
    Returns:
        dict: Action distribution mapping action names to probabilities
    """
    if hasattr(model.policy, 'mlp_extractor'):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=model.device).unsqueeze(0)
            latent_pi, _ = model.policy.mlp_extractor(obs_tensor)
            action_logits = model.policy.action_net(latent_pi)
            if hasattr(model.policy, 'action_dist'):
                # For policies with action_dist (e.g., PPO, A2C)
                action_dist = model.policy.action_dist.proba_distribution(action_logits)
                if hasattr(action_dist, 'distribution') and hasattr(action_dist.distribution, 'probs'):
                    probs = action_dist.distribution.probs.cpu().numpy().flatten()
                else:
                    probs = action_dist.probs.cpu().numpy().flatten()
            else:
                # For DQN, action_logits are Q-values; use softmax to get probabilities
                probs = torch.softmax(action_logits, dim=1).cpu().numpy().flatten()
            action_distribution = {l2g.INVERSE_ACTION_MAP[i]: probs[i] for i in range(len(probs))}
            return action_distribution


def record_video(model, video_path, env_name, seed, apply_shield, log_dir, grid_size=60, universal_counter=False, enforcement_parameter=ENFORCEMENT_PARAMETER, safety_only=False):
    """
    Record a video of the trained model performing in the environment.
    
    This function can record videos with or without safety shielding. When shielding
    is enabled, it builds a graph representation of the environment, computes safety
    and liveness properties, and uses a shield to modify the agent's actions.
    
    Args:
        model: Trained RL model
        video_path (str): Directory to save the video
        env_name (str): Name of the Gymnasium environment
        seed (int): Random seed for reproducibility
        apply_shield (bool): Whether to apply safety shielding
        log_dir (str): Directory for log files
        grid_size (int): Size of the discretization grid for shielding
        universal_counter (bool): Whether to use universal counters in the shield
        enforcement_parameter (float): Shield enforcement strength
        safety_only (bool): If True, only apply safety shield (no liveness)
    """    
    
    # Generate descriptive names for video and log files
    safe_env_name = env_name.replace("/", "_")
    video_name = f"{safe_env_name}_seed{seed}"
    grid_image_name = f"{safe_env_name}_seed{seed}"
    if apply_shield:
        video_name += f"_shielded"
        if safety_only:
            video_name += "_safety"
        else:
            video_name += f"_uni_{universal_counter}"

    log_file_name = f"{video_name}.log"
    setup_logging(log_dir, log_file_name)

    # Create environment with video recording capability
    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_path,
        name_prefix=video_name,
        episode_trigger=lambda x: True,  # Record every episode
    )

    obs, _ = env.reset(seed=seed) 

    # Extract terrain information from custom environment for MARG shield generation
    terrain_info = None
    if not hasattr(env.unwrapped, "terrain_info"):
        raise ValueError("The environment does not have terrain data. Ensure you are using CustomLunarLander-v0.")
    else:
        terrain_info = env.unwrapped.terrain_info

    # Set up paths for saving grid visualization
    grid_image_path = os.path.join(video_path.replace("videos", "grids"), f"{grid_image_name}.png")
    os.makedirs(os.path.dirname(grid_image_path), exist_ok=True)

    # Initialize shield-related variables
    graph = None
    safe_buechi_graph = None
    shield = None
    shield_safety = None
    terrainGraphOutput = None
    
    if apply_shield:
        logging.info("Shielding is enabled, building graph with terrain information.")

        # Build graph representation of the environment using terrain data
        terrainGraphOutput = l2g.buildGraphWithTerrain(grid_size, terrainInfo=terrain_info, logger=logging.getLogger())

        logging.info(f"Grid image will be saved to: {grid_image_path}")

        # Create visualization elements for different regions of the grid
        visualElements = l2g.VisualElements()
        if safety_only:
            # For safety-only mode, highlight recovery zones
            visualElements.add(terrainGraphOutput.get_element("recovery_ground_safety_cells"), 'orange', alpha=0.2)
        else:
            # For full shielding, highlight target helipad
            visualElements.add(terrainGraphOutput.get_element("helipad_column"), 'blue', alpha=0.5)
        # Common visualization elements
        visualElements.add(terrainGraphOutput.get_element("ground_cells"), 'green', alpha=0.8)
        visualElements.add(terrainGraphOutput.get_element("extreme_cells"), 'red', alpha=0.3)
        
        # Generate and save the grid visualization
        l2g.visualizeGraph(terrainGraphOutput.get_element("graph"), terrainGraphOutput.get_element("grid"), visualElements, file_location=grid_image_path)

        # Convert terrain graph to data structures for shield computation
        graph, safe_buechi_graph = l2g.graphToDataStructure(terrainGraphOutput=terrainGraphOutput, logger=logging.getLogger())
        logging.debug(f"Graph created with {len(graph.vertices)} vertices and {sum(len(v) for v in graph.outgoing_edges.values())} edges.")

        # Compute safety and liveness templates using parity games
        _, safety_template, co_live_template, group_liveness_template = compute_template(graph, 0)
        logging.info("Winning region and templates computed.")

        # Compute templates for the safety-focused graph
        _, safety_template_safe_buechi, co_live_template_safe_buechi, group_liveness_template_safe_buechi = compute_template(safe_buechi_graph, 0)
        logging.info("Winning region and templates computed for safe Buechi graph.")

        # Initialize shields with computed templates
        shield = Shield(graph, safety_template, co_live_template, group_liveness_template, enforcement_parameter=enforcement_parameter, universal_counter=universal_counter, logger=logging.getLogger())
        logging.info(f"Shield initialized with enforcement parameter {enforcement_parameter}.")

        # Initialize safety-only shield 
        shield_safety = Shield(safe_buechi_graph, safety_template_safe_buechi, co_live_template_safe_buechi, group_liveness_template_safe_buechi, enforcement_parameter=RECOVERY_ENFORCEMENT_PARAMETER, universal_counter=universal_counter, logger=logging.getLogger())
    
    # Episode execution loop
    old_obs = obs.copy()
    steps = 0
    logging.info(f"Starting video recording for environment: {env_name}")
    logging.debug(f"Grid is {terrainGraphOutput.get_element('grid') if terrainGraphOutput else 'not available'}")
    done = False
    landedLongEnough = 2  # Number of steps to land on the helipad before ending
    grid = terrainGraphOutput.get_element("grid") if terrainGraphOutput else None
    
    while not done:
        logging.info(f"Step {steps}, current observation: {obs}")
        
        # Get action distribution from the trained model
        action_distribution = get_action_distribution(model, obs)

        next_action, next_state = None, None
        if apply_shield:
            # Convert continuous observation to discrete state for shield
            discrete_state = l2g.map_state_to_discrete(obs, grid)
            logging.debug(f"Discrete state {discrete_state} mapped from observation: {obs}")
            
            if shield is None or shield_safety is None:
                raise ValueError("Shield is enabled but not initialized.")
                
            # Apply appropriate shield based on mode
            if safety_only:
                next_action, next_state = shield_safety.sample_next_action_and_state(discrete_state, action_distribution)
                logging.debug(f"Expected action is {max(action_distribution, key=lambda k: action_distribution[k])} but shielded action is {next_action} for state {discrete_state} and expected next state: {next_state}")
            else:
                next_action, next_state = shield.sample_next_action_and_state(discrete_state, action_distribution)
                logging.debug(f"Expected action is {max(action_distribution, key=lambda k: action_distribution[k])} but shielded action is {next_action} for state {discrete_state} and expected next state: {next_state}")
        else:
            # Without shield, select action with highest probability
            if action_distribution is not None and len(action_distribution) > 0:
                next_action = max(action_distribution, key=lambda k: action_distribution[k])
                logging.debug(f"Action selected: {next_action} with distribution: {action_distribution}")
            else:
                logging.error("Action distribution is None or empty.")
                raise ValueError("Action distribution is None or empty.")
        # Convert action name to environment action index and execute step
        action = l2g.ACTION_MAP[next_action]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check landing status for potential early termination
        if terrain_info is not None:
            _ = l2g.landedOutsideHelipad(obs, terrain_info["helipad_x1"], terrain_info["helipad_x2"], logger=logging.getLogger())
            
        done = terminated or truncated
        obs_diff = obs - old_obs  # Track observation changes
        old_obs = obs.copy()
        
        # Validate shield predictions if shielding is active
        if apply_shield:
            new_state = l2g.map_state_to_discrete(obs, grid)
            new_state = (int(new_state[0]), int(new_state[1]))  # Convert to tuple of integers

            logging.debug(f"State {new_state} has priority {graph.priorities[new_state][0]} and priority {safe_buechi_graph.priorities[new_state][0]}.")

            # Check if the actual transition matches the shield's prediction
            if (new_state) != next_state:
                logging.warning(f"State mismatch: expected {next_state}, got {new_state}.")
            else:
                logging.debug(f"State match: {new_state} matches expected {next_state}.")

        logging.info(f"Step {steps}: \n\t\tAction: {action}, Reward: {reward}")
        
        steps += 1
        
        # Check for successful landing on helipad
        if l2g.landedOnHelipad(obs, terrain_info["helipad_x1"], terrain_info["helipad_x2"]):
            logging.info("Lander has landed on the helipad. Counting down to end episode.")
            landedLongEnough -= 1
            
        if landedLongEnough <= 0:
            logging.info("Reached maximum steps for video recording.")
            done = True
    # Clean up and finalize video recording
    logging.info(f"Video recording completed after {steps} steps.")
    logging.info("Video recording finished, closing environment.")

    env.close()
    
    # Optional: overlay grid visualization on video (currently disabled)
    # if apply_shield:
    #     overlay_image_on_video(env_name, os.path.join(video_path, f"{video_name}-episode-0.mp4"), grid_image_path, os.path.join(video_path, f"{video_name}-overlay.mp4"))

    logging.info(f"Video saved to {video_path} as {video_name}-episode-0.mp4")

def overlay_image_on_video(env_name: str, video_path: str, image_path: str, output_path: str):
    """
    Overlays an image on a video using ffmpeg, replicating a specific filter_complex pipeline.
    
    Parameters:
        video_path (str): Path to the input video file.
        image_path (str): Path to the overlay image.
        output_path (str): Path to save the output video.
    """

    # Create environment to get frame dimensions for ffmpeg scaling
    env = gym.make(env_name, render_mode="rgb_array")
    env.reset(seed=42)  # Reset the environment to ensure it is ready for rendering
    frame = env.render()
    frame_height, frame_width, _ = frame.shape  # Get the frame size from the environment
    
    # Build ffmpeg command for overlaying grid image on video
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', video_path,
        '-i', image_path,
        '-filter_complex',
        f'[1:v]scale={frame_width}:{frame_height},format=rgba,colorchannelmixer=aa=0.3[ovr];'
        '[0:v][ovr]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto',
        '-c:a', 'copy',
        output_path
    ]
    
    try:
        # Execute ffmpeg command to create overlay video
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Delete the original video file to save space
        if os.path.exists(video_path):
            os.remove(video_path)
    except subprocess.CalledProcessError as e:
        print("Error running ffmpeg command:", e)


def main():
    """
    Main function to parse command line arguments and execute video recording.
    
    Supports recording videos with various shield configurations:
    - No shield (baseline model performance)
    - Full shield with safety and liveness properties
    - Safety-only shield for recovery behaviors
    - Universal vs individual counters for liveness properties
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Tag to find the model directory")
    parser.add_argument("--env", required=True, help="Gymnasium environment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alg", default="ppo", help="RL algorithm (ppo, dqn, a2c, sac, td3)")
    parser.add_argument("--shield", action="store_true", help="Whether to use shielded environments")
    parser.add_argument("--universal", action="store_true", help="Whether to use universal counter for live groups")
    parser.add_argument("--all", action="store_true", help="Whether to generate all kinds of videos")
    parser.add_argument("--safety", action="store_true", help="Whether to generate video with only safety shield")
    parser.add_argument("--grid_size", type=int, default=50, help="Grid size for shielded environments (default: 50)")
    args = parser.parse_args()

    # Locate the training directory and set up paths
    base_dir = find_directory_by_tag(args.tag)
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_path = os.path.join(base_dir, "models", f"{args.env}.zip")
    video_dir = os.path.join(base_dir, "videos")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    os.makedirs(video_dir, exist_ok=True)

    # Load the trained model
    print(f"Loading model from: {model_path}")
    dummy_env = gym.make(args.env)  # Dummy environment needed for model loading
    alg_name = args.alg.lower()  # Normalize algorithm name
    model = load_model(model_path, dummy_env, alg_name)
    # Record videos based on command line arguments
    print(f"Recording video in: {video_dir}")
    if args.all:
        # Generate all possible video variants for comprehensive evaluation
        print("Recording video with universal counter.")
        record_video(model, video_dir, args.env, args.seed, args.shield, log_dir, args.grid_size, universal_counter=True, safety_only=False)
        print("Recording video with individual counters.")
        record_video(model, video_dir, args.env, args.seed, args.shield, log_dir, args.grid_size, universal_counter=False, safety_only=False)
        print("Recording video with safety shield.")
        record_video(model, video_dir, args.env, args.seed, True, log_dir, args.grid_size, safety_only=True)
        print("Recording video without shield.")
        record_video(model, video_dir, args.env, args.seed, False, log_dir, args.grid_size)
    else:
        # Record single video with specified configuration
        print(f"Recording video with {'universal' if args.universal else 'individual'} counter.")
        record_video(model, video_dir, args.env, args.seed, args.shield, log_dir, args.grid_size, args.universal, args.safety)

if __name__ == "__main__":
    main()

# Example usage:
# python3 model_runner.py --tag testnew --env CustomLunarLander-v0 --grid_size 60 --shield --universal --seed 77 --all