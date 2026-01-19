from os import path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import numpy as np
import os
from pathlib import Path
import argparse

# font sizes
font_size = 30
tick_size = 24
fig_size = (6,7)

# Load CSV data
def find_run_dir(base_dir, tag):
    # find ppo_*_<tag> directory
    for entry in os.listdir(base_dir):
        if entry.endswith(f'_{tag}'):
            return Path(base_dir) / entry
    raise FileNotFoundError(f"Run directory with tag '{tag}' not found in {base_dir}")


def plot_for_shield_param(rate,label,live_df,other_df):
    # Create a bar plot for each shield_type and parameter
    plt.figure(figsize=(15, 7))
    ax = sns.barplot(
        data=live_df,
        x='parameter',
        y=rate
    )
    plt.xlabel('Parameters', fontsize=font_size)
    plt.ylabel(f'{label}', fontsize=font_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.tight_layout()

def plot_for_shield(parameter,rate,label,live_df,other_df):
    plt.figure(figsize=fig_size)
    # join live_df with parameter and other_df
    live_df_param = live_df[live_df['parameter'] == parameter]
    all_df = pd.concat([live_df_param, other_df])
    ax = sns.barplot(
        data=all_df,
        x='shield_type',
        y=rate,
        color='gray',
        hue='shield_type'
    )
    plt.xlabel('Shield Type', fontsize=font_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.ylabel(f'{label}', fontsize=font_size)
    plt.tight_layout()


# Plot success_rate, outside_rate, and fail_rate for each shield_type in same bar plot
def plot_stacked_bar(parameter, live_df, other_df):
    plt.figure(figsize=fig_size)
    # join live_df with parameter and other_df
    live_df_param = live_df[live_df['parameter'] == parameter]
    other_df = other_df[other_df['parameter'] == parameter]
    # Concatenate the DataFrames for plotting
    all_df = pd.concat([live_df_param, other_df])
    ax = sns.barplot(
        data=all_df,
        x='shield_type',
        y='success_rate',
        color='mediumseagreen',
        hatch='+',
        label='Helipad'
    )
    sns.barplot(
        data=all_df,
        x='shield_type',
        y='outside_rate',
        bottom=all_df['success_rate'],
        color='lightblue',
        hatch='',
        label='Outside'
    )
    sns.barplot(
        data=all_df,
        x='shield_type',
        y='fail_rate',
        bottom=all_df['success_rate'] + all_df['outside_rate'],
        color='lightcoral',
        hatch='X',
        label='Failed'
    )
    plt.xlabel('Shield Type', fontsize=font_size)
    plt.ylabel('Landing Status Rate (%)', fontsize=font_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    # plt.legend(title='Landing Status', title_fontsize=tick_size-2, fontsize=tick_size, loc='upper right')
    plt.legend(fontsize=tick_size, loc='upper right')
    plt.tight_layout()


def main(tag, base_dir='results'):
    run_dir = find_run_dir(base_dir, tag)
    report_dir = run_dir / 'reports'
    graph_dir = run_dir / 'graphs'
    csv_file = report_dir / f'summary_{tag}.csv'

    # load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    # Replace None (np.nan) in 'shield' with 'NoShield' for later use
    noshield = 'None'
    df['shield'] = df['shield'].replace({None: noshield, 'None': noshield, float('nan'): noshield, pd.NA: noshield, np.nan: noshield})
    # join 'shield' and 'counter' into a single column for hue
    df['shield_type'] = df['shield'].astype(str) + ',' + df['counter'].astype(str)
    # Rename 'shield_type' to new names
    df['shield_type'] = df['shield_type'].replace({
        f'{noshield},False': f'{noshield}',
        "Safety,False": 'Safety',
        "Liveness,True": 'Liveness (uni)',
        "Liveness,False": 'Liveness'
    })
    # for each shield_type and parameter, find the success rate and average steps
    results = []
    for shield_type in [st for st in df['shield_type'].unique() if st != 'Liveness (uni)']:
        for parameter in df['parameter'].unique():
            subset = df[(df['shield_type'] == shield_type) & (df['parameter'] == parameter)]
            if not subset.empty:
                success_count = subset[subset['status'] == 'Helipad'].shape[0]
                outside_count = subset[subset['status'] == 'Outside'].shape[0]
                fail_count = subset[subset['status'] == 'Unsuccessful'].shape[0]
                success_rate = 100 * success_count / (success_count + fail_count + outside_count)
                outside_rate = 100 * outside_count / (success_count + fail_count + outside_count)
                fail_rate = 100 * fail_count / (success_count + fail_count + outside_count)

                avg_steps = subset['steps'].mean()

                results.append({
                    'shield_type': shield_type,
                    'parameter': parameter,
                    'success_rate': success_rate,
                    'outside_rate': outside_rate,
                    'fail_rate': fail_rate,
                    'avg_steps': avg_steps
                })
    # Fix the order of shield_type
    shield_order = ['Liveness', 'Safety', noshield]
    # Ensure the shield_type is in the correct order
    results = sorted(results, key=lambda x: shield_order.index(x['shield_type']) if x['shield_type'] in shield_order else len(shield_order))
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    live_df = results_df[results_df['shield_type'] == 'Liveness']
    other_df = results_df[results_df['shield_type'] != 'Liveness']
    shield_types = results_df['shield_type'].unique()
    # custom bar hatch patterns and colors
    hatch_patterns = ['o','','x']
    unique_hues = results_df['shield_type'].unique()
    hatch_map = {hue: hatch_patterns[i % len(hatch_patterns)] for i, hue in enumerate(unique_hues)}
    sns.set(style='whitegrid')




    # Plot success_rate for each shield_type and parameter
    plot_for_shield_param('success_rate', 'Success Rate (%)', live_df, other_df)
    plt.savefig(path.join(graph_dir, 'fig8a.png'), dpi=300)
    plt.close()

    # Plot avg_steps for each shield_type and parameter
    plot_for_shield_param('avg_steps', 'Average Steps', live_df, other_df)
    plt.savefig(path.join(graph_dir, 'fig8b.png'), dpi=300)
    plt.close()

    ## Optimal Parameter
    optimal_parameter = 0.08

    # Plot avg_steps for each shield_type
    plot_for_shield(optimal_parameter, 'avg_steps', 'Average Steps', live_df, other_df)
    plt.savefig(path.join(graph_dir, 'fig4c.png'), dpi=300)
    plt.close()

    # Plot stacked bar for success_rate, outside_rate, and fail_rate for each shield_type
    plot_stacked_bar(optimal_parameter, live_df, other_df)
    plt.savefig(path.join(graph_dir, 'fig4b.png'), dpi=300)
    plt.close()

    # print a table of results for parameter optimal_parameter
    optimal = {}
    steps = {}
    optimal['Liveness'] = live_df[live_df['parameter'] == optimal_parameter]['success_rate'].values[0]
    steps['Liveness'] = live_df[live_df['parameter'] == optimal_parameter]['avg_steps'].values[0]
    optimal['Safety'] = other_df[(other_df['shield_type'] == 'Safety')]['success_rate'].mean()
    steps['Safety'] = other_df[(other_df['shield_type'] == 'Safety')]['avg_steps'].mean()
    optimal[noshield] = other_df[(other_df['shield_type'] == noshield)]['success_rate'].mean()
    steps[noshield] = other_df[(other_df['shield_type'] == noshield)]['avg_steps'].mean()
    # Convert the optimal dictionary to a DataFrame and save
    pd.DataFrame([optimal, steps]).to_csv(path.join(report_dir, 'results.csv'), index=False, float_format='%.2f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize lunar lander videos')
    parser.add_argument('--tag', default='review', help='Tag suffix of the ppo run directory')
    parser.add_argument('--base_dir', default='results', help='Base directory for results')
    args = parser.parse_args()
    main(args.tag, args.base_dir)
