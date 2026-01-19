import os
import re
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def find_run_dir(base_dir, tag):
    # find ppo_*_<tag> directory
    for entry in os.listdir(base_dir):
        if entry.endswith(f'_{tag}'):
            return Path(base_dir) / entry
    raise FileNotFoundError(f"Run directory with tag '{tag}' not found in {base_dir}")


def parse_log(log_path):
    status_pattern = re.compile(r"Lander landed on the helipad\.|Lander landed outside the helipad\.|Lander is not landed,.*")
    steps_pattern = re.compile(r"Video recording completed after (\d+) steps")
    status = None
    steps = None
    helipad = False
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m_stat = status_pattern.search(line)
            if m_stat and not helipad:
                text = m_stat.group(0)
                if 'on the helipad' in text:
                    status = 'Helipad'
                    helipad = True
                elif 'outside the helipad' in text:
                    status = 'Outside'
                else:
                    status = 'Unsuccessful'
            m_steps = steps_pattern.search(line)
            if m_steps:
                steps = int(m_steps.group(1))
    return status or 'Unsuccessful', steps or 0


def extract_info_from_filename(fn):
    # e.g. CustomLunarLander-v0_seed42_shielded_uni_True.log or ..._safety.log
    shield = 'None'
    counter = ''
    if 'shielded_safety' in fn:
        shield = 'Safety'
    elif 'shielded_uni' in fn:
        shield = 'Liveness'
        if '_uni_True' in fn:
            counter = 'True'
        elif '_uni_False' in fn:
            counter = 'False'
    # extract seed and param from path; handled externally
    return shield, counter


def main(tag, base_dir='results'):
    run_dir = find_run_dir(base_dir, tag)
    videos_root = run_dir / 'videos'
    report_dir = run_dir / 'reports'
    graphs_dir = run_dir / 'graphs'
    # some logs are stored in run_dir/logs/videos
    other_videos = run_dir / 'logs' / 'videos'

    # prepare output dirs
    for d in (report_dir, graphs_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    records = []
    # traverse log files
    for seed_dir in videos_root.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
            continue
        seed = seed_dir.name.split('_')[1]
        for param_dir in seed_dir.iterdir():
            if not param_dir.is_dir() or not param_dir.name.startswith('param_'):
                continue
            parameter = param_dir.name.split('_')[1]
            if parameter == '1.00':
                # skip the default parameter directory, as it contains no logs
                continue
            # logs in misnamed 'videos' subfolder
            log_folder = param_dir / 'videos'
            if not log_folder.exists():
                continue
            for log_file in log_folder.glob('*.log'):
                shield, counter = extract_info_from_filename(log_file.name)
                status, steps = parse_log(log_file)
                records.append({
                    'seed': int(seed),
                    'parameter': float(parameter),
                    'shield': shield,
                    'counter': counter if counter else 'False',
                    'steps': steps,
                    'status': status
                })
                
    df = pd.DataFrame(records)
    csv_path = report_dir / f'summary_{tag}.csv'
    df.to_csv(csv_path, index=False)
    print(f"CSV report saved to {csv_path}")

    # Visualization: bar graph of helipad landings per parameter and video type
    helipad_df = df[df['status'] == 'Helipad']
    # pivot counts
    pivot_helipad = helipad_df.pivot_table(
        index='parameter',
        columns=['shield', 'counter'],
        values='seed',
        aggfunc='count',
        fill_value=0
    )

    total_helipad = pivot_helipad.values.sum() if not pivot_helipad.empty else 0
    if total_helipad > 0:
        pivot_helipad.plot(kind='bar', figsize=(10, 6))
        plt.title('Helipad Landings per Parameter and Video Type')
        plt.xlabel('Parameter')
        plt.ylabel('Number of Successes')
        plt.tight_layout()
        bar_path = graphs_dir / f'bar_helipad_{tag}.png'
        plt.savefig(bar_path)
        plt.close()
        print(f"Bar graph saved to {bar_path}")
    else:
        print("No successful helipad landings found, skipping bar graph.")

    # Visualization: bar graph of outside landings per parameter and video type
    outside_df = df[df['status'] == 'Outside']
    # pivot counts
    pivot_outside = outside_df.pivot_table(
        index='parameter',
        columns=['shield', 'counter'],
        values='seed',
        aggfunc='count',
        fill_value=0
    )
    total_outside = pivot_outside.values.sum() if not pivot_outside.empty else 0
    if total_outside > 0:
        pivot_outside.plot(kind='bar', figsize=(10, 6))
        plt.title('Outside Landings per Parameter and Video Type')
        plt.xlabel('Parameter')
        plt.ylabel('Number of Successes')
        plt.tight_layout()
        bar_path = graphs_dir / f'bar_outside_{tag}.png'
        plt.savefig(bar_path)
        plt.close()
        print(f"Bar graph saved to {bar_path}")
    else:
        print("No outside landings found, skipping bar graph.")

    # Pie charts per video type
    types = df[['shield', 'counter']].drop_duplicates().to_dict('records')
    for t in types:
        shield, counter = t['shield'], t['counter']
        subset = df[(df['shield'] == shield) & (df['counter'] == counter)]
        counts = subset.groupby('parameter')['status'].apply(lambda x: (x=='Helipad').sum())
        if counts.empty:
            continue
        total = counts.sum()
        if total == 0:
            print(f"Skipping pie chart for {shield}, counter={counter}: no successes")
            continue
        plt.figure(figsize=(6, 6))
        counts.plot(kind='pie', autopct='%1.1f%%', legend=False)
        plt.title(f'Success Rate by Parameter ({shield}, Counter={counter})')
        plt.ylabel('')
        pie_path = graphs_dir / f'pie_{shield}_{counter}_{tag}.png'
        plt.savefig(pie_path)
        plt.close()
        print(f"Pie chart saved to {pie_path}")

    print("All reports and graphs generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize lunar lander videos')
    parser.add_argument('--tag', default='review', help='Tag suffix of the ppo run directory')
    parser.add_argument('--base_dir', default='results', help='Base directory for results')
    args = parser.parse_args()
    main(args.tag, args.base_dir)
