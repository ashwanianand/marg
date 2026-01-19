#!/usr/bin/env python3
import os
from pathlib import Path
import json, subprocess, sys, time

ROOT_DIR = Path(__file__).resolve().parent
PESTEL = ROOT_DIR / 'pestel' / 'build' / 'pestel'

# Now import the function
from lib.generate_hoa import create_hoa_file
from lib.game import OvercookedGame
from lib.generate_layout import generate_layout


def main(extra):
    print(f"### For layout = {extra} ###")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    layout = f"{extra}.layout"
    layout_filename = os.path.join(ROOT_DIR, 'layouts', layout)
    game_filename = os.path.join(ROOT_DIR, 'hoa_files', f'OvercookedGame_{extra}.hoa')
    strategy_filename = os.path.join(ROOT_DIR, 'templates', f'OvercookedGame_{extra}.txt')
    results_file = os.path.join(ROOT_DIR, 'results', f'OvercookedGame_{extra}.json')
    results = {}

    results_key = f"{extra}"
    results[results_key] = {}

    # generate layout file
    start = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    # generate_layout(layout_filename, extra_rows=extra)
    end = time.time()
    layout_time = end - start
    print(f"Generated layout with {extra} extra rows: {layout_filename} in {layout_time:.4f} seconds")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    results[results_key]['layout_time'] = layout_time

    # generate hoa file
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    start = time.time()
    game = OvercookedGame(layout_filename)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    len_states = create_hoa_file(game, game_filename)
    end = time.time()
    hoa_time = end - start
    results[results_key]['hoa_time'] = hoa_time
    results[results_key]['num_states'] = len_states

    # generate strategy template
    print(f"Generating strategy template...")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    start = time.time()
    try:
        with open(game_filename, "rb") as f:
            hoa_bytes = f.read()
        out = subprocess.check_output(
            [PESTEL],
            input=hoa_bytes
        )
        with open(strategy_filename, "wb") as f:
            f.write(out)
    except Exception as e:
        sys.exit(f"Error: {e}")
    end = time.time()
    strategy_time = end - start
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end=' - ')
    results[results_key]['strategy_time'] = strategy_time

    # Write results to file after each run
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main(str(sys.argv[1]) if len(sys.argv) > 1 else 0)  # Default to 0 extra rows if not specified
