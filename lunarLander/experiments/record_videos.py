#!/usr/bin/env python3
"""
record_videos.py

A script to record videos for multiple tags, generate random seeds based on a meta-seed (or fully random if None),
and vary enforcement parameters in parallel. It also logs the generated seeds to a timestamped log file and
saves video-run logs alongside the videos to avoid cluttering the current directory.

Usage:
    python3 record_videos.py [--workers N] [--meta-seed M] [--num-seeds K]

Configure TAGS, PARAMS, ENV_NAME, ALG, GRID_SIZE below or via command-line.
"""
import os
import itertools
import argparse
import random
import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gymnasium as gym

# Import your core functions from model_runner.py
from model_runner import find_directory_by_tag, load_model, record_video

# ─── Gym’s own logger ─────────────────────────────────────────────────────
# Send all Gym INFO+ messages into record_videos.log instead of stdout/stderr
gym_logger = logging.getLogger("gym")
gym_logger.setLevel(logging.INFO)
# If it has no file handler yet, add one:
if not any(isinstance(h, logging.FileHandler) for h in gym_logger.handlers):
    fh = logging.FileHandler("record_videos.log", mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    gym_logger.addHandler(fh)


# === Configuration Defaults ===
TAGS = ["review"]
PARAMS = [i / 100 for i in range(1, 10)] + [i / 10 for i in range(1, 6)] # 0.01, 0.02, ..., 0.09, 0.1, 0.2, ..., 0.5 
ENV_NAME = "CustomLunarLander-v0"
ALG = "ppo"
GRID_SIZE = 60
DEFAULT_META_SEED = 42 # can be None
DEFAULT_NUM_SEEDS = 200
# ===============================


def process_combo(tag: str, seed: int, ep: float):
    """
    Record all four video variants for a given tag, seed, and enforcement parameter.
    Saves both videos and logs into the same out_dir to avoid writing files to CWD.
    """
    base_dir = find_directory_by_tag(tag)
    model_path = os.path.join(base_dir, "models", f"{ENV_NAME}.zip")
    env = gym.make(ENV_NAME)
    model = load_model(model_path, env, ALG)

    # Directory for this seed + parameter: videos + logs go here
    out_dir = os.path.join(base_dir, "videos", f"seed_{seed}", f"param_{ep:.2f}")
    os.makedirs(out_dir, exist_ok=True)

    # Use out_dir as the log_dir so that record_video writes logs there
    log_dir = out_dir

    # # 1) Unshielded
    record_video(
        model, out_dir, ENV_NAME, seed,
        apply_shield=False,
        log_dir=log_dir,
        grid_size=GRID_SIZE,
        universal_counter=False,
        enforcement_parameter=ep,
        safety_only=False,
    )

    # 2) Shielded: safety only
    record_video(
        model, out_dir, ENV_NAME, seed,
        apply_shield=True,
        log_dir=log_dir,
        grid_size=GRID_SIZE,
        universal_counter=False,
        enforcement_parameter=ep,
        safety_only=True,
    )

    # 4) Shielded: liveness + individual counter (universal_counter=False)
    record_video(
        model, out_dir, ENV_NAME, seed,
        apply_shield=True,
        log_dir=log_dir,
        grid_size=GRID_SIZE,
        universal_counter=False,
        enforcement_parameter=ep,
        safety_only=False,
    )

    return (tag, seed, ep)


def main(workers: int, meta_seed, num_seeds: int, tags: list):
    # Generate random seeds: reproducible if meta_seed is not None, else fully random
    if meta_seed is None:
        seeds = [random.randint(0, 2**16 - 1) for _ in range(num_seeds)]
        meta_info = "(fully random)"
    else:
        rng = random.Random(meta_seed)
        seeds = [rng.randint(0, 2**16 - 1) for _ in range(num_seeds)]
        meta_info = f"(meta-seed={meta_seed})"

    # Write out a seed log for reproducibility
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"seeds_{timestamp}.log"
    with open(log_filename, "w") as logf:
        logf.write(f"Seed log generated at {timestamp}\n")
        if meta_seed is None:
            logf.write("Meta-seed: None (fully random)\n")
        else:
            logf.write(f"Meta-seed: {meta_seed}\n")
        logf.write("Seeds:\n")
        for s in seeds:
            logf.write(f"{s}\n")
    print(f"Seed log written to {log_filename}")

    combos = list(itertools.product(tags, seeds, PARAMS))
    total = len(combos)
    print(f"Launching {total} jobs {meta_info} with up to {workers} workers...")

    with open(log_filename, "a") as logf:
        logf.write(f"0/{total}\n")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(process_combo, tag, seed, ep): (tag, seed, ep)
            for tag, seed, ep in combos
        }

        for idx, future in enumerate(tqdm(as_completed(futures), total=total, desc="Jobs", unit="job"), start=1):
            tag, seed, ep = futures[future]
            try:
                future.result()
                tqdm.write(f"[{idx}/{total}] Done: tag={tag}, seed={seed}, ep={ep:.2f}")
                with open(log_filename, "a") as logf:
                    logf.write(f"{idx}/{total} - tag={tag}, seed={seed}, ep={ep:.2f}\n")
            except Exception as e:
                tqdm.write(f"ERROR tag={tag}, seed={seed}, ep={ep:.2f}: {e}")
                with open(log_filename, "a") as logf:
                    logf.write(f"ERROR tag={tag}, seed={seed}, ep={ep:.2f}: {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel video recording for tags with random or meta-seeded seeds and parameters"
    )
    parser.add_argument(
        "--workers", type=int, default=(os.cpu_count() or 2) // 2,
        help="Max parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--meta-seed", type=int, default=DEFAULT_META_SEED,
        help="Meta seed for random seed generation (omit for fully random)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
        help="Number of random seeds to generate"
    )
    parser.add_argument(
        "--tags", nargs="+", default=TAGS,
        help="List of tags to process (default: %(default)s)"
    )
    parser.add_argument(
        "--tag", type=str, default="review",
        help="Tag suffix of the ppo run directory (default: %(default)s)"
    )
    args = parser.parse_args()
    tags = args.tags if args.tags else [args.tag]
    main(args.workers, args.meta_seed, args.num_seeds, tags)
