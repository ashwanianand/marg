"""
Unified plotting module for generating all experimental plots.
This module combines the functionality from plotter.py and percentPlot.py
and generates all plots (including commented ones) using JSON files from ./results/
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


class PlotGenerator:
    def __init__(self, results_dir="./results/outputs/", plots_dir="./plots/"):
        """
        Initialize the plot generator.
        
        Args:
            results_dir (str): Directory containing JSON result files
            plots_dir (str): Directory where plots will be saved
        """
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        
        # Create plots directory if it doesn't exist
        os.makedirs(plots_dir, exist_ok=True)
        
        # Initialize data storage
        self.reset_data()
    
    def reset_data(self):
        """Reset all data storage variables."""
        # Data from plotter.py logic
        self.enforcement_params = None
        self.buechi_unshielded = []
        self.buechi_shielded = []
        self.rewards_unshielded = []
        self.rewards_shielded = []
        self.buechi_unshielded_close = []
        self.buechi_shielded_close = []
        self.rewards_unshielded_close = []
        self.rewards_shielded_close = []
        self.buechi_unshielded_far = []
        self.buechi_shielded_far = []
        self.rewards_unshielded_far = []
        self.rewards_shielded_far = []
        self.value_at_initial_state_close = None
        self.value_at_initial_state_far = None
        
        # Data from percentPlot.py logic
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.buechi_shielded_sums = {t: 0 for t in self.thresholds}
        self.buechi_unshielded_sums = {t: 0 for t in self.thresholds}
        self.counts_shielded = {t: 0 for t in self.thresholds}
        self.counts_unshielded = {t: 0 for t in self.thresholds}
        self.reward_diff_shielded = []
        self.buechi_shielded_percent = []
        self.reward_diff_unshielded = []
        self.buechi_unshielded_percent = []
    
    def load_data(self):
        """Load data from JSON files in results directory."""
        if not os.path.exists(self.results_dir):
            print(f"Warning: Results directory {self.results_dir} does not exist.")
            return
        
        json_files = [f for f in os.listdir(self.results_dir) if f.endswith(".json")]
        if not json_files:
            print(f"Warning: No JSON files found in {self.results_dir}")
            return
        
        print(f"Loading data from {len(json_files)} JSON files...")
        
        for filename in json_files:
            filepath = os.path.join(self.results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                self._process_plotter_data(data, filename)
                self._process_percent_plot_data(data)
                
            except Exception as e:
                print(f"Warning: Error processing {filename}: {e}")
    
    def _process_plotter_data(self, data, filename):
        """Process data according to plotter.py logic."""
        if self.enforcement_params is None:
            self.enforcement_params = data.get("enforcement_parameters", [])
        
        # Main data aggregation
        self.buechi_unshielded.append(data.get("buechi_counter_unshielded", []))
        self.buechi_shielded.append(data.get("buechi_counter_shielded", []))
        
        value_at_initial = data.get("value_at_initial_state", 0)
        rewards_unsh = data.get("average_rewards_unshielded", [])
        rewards_sh = data.get("average_rewards_shielded", [])
        
        self.rewards_unshielded.append([-val + value_at_initial for val in rewards_unsh])
        self.rewards_shielded.append([-val + value_at_initial for val in rewards_sh])
        
        # Separate data by distance (close vs far)
        if "midr=0.1" in filename:
            self.buechi_unshielded_close.append(data.get("buechi_counter_unshielded", []))
            self.buechi_shielded_close.append(data.get("buechi_counter_shielded", []))
            self.rewards_unshielded_close.append([-val + value_at_initial for val in rewards_unsh])
            self.rewards_shielded_close.append([-val + value_at_initial for val in rewards_sh])
            
            if self.value_at_initial_state_close is None:
                self.value_at_initial_state_close = value_at_initial
        
        elif "midr=0.7" in filename:
            self.buechi_unshielded_far.append(data.get("buechi_counter_unshielded", []))
            self.buechi_shielded_far.append(data.get("buechi_counter_shielded", []))
            self.rewards_unshielded_far.append([-val + value_at_initial for val in rewards_unsh])
            self.rewards_shielded_far.append([-val + value_at_initial for val in rewards_sh])
            
            if self.value_at_initial_state_far is None:
                self.value_at_initial_state_far = value_at_initial
    
    def _process_percent_plot_data(self, data):
        """Process data according to percentPlot.py logic."""
        value_at_initial_state = data.get("value_at_initial_state", 0)
        rewards_shielded = np.array(data.get("average_rewards_shielded", []))
        rewards_unshielded = np.array(data.get("average_rewards_unshielded", []))
        buechi_shielded_vals = np.array(data.get("buechi_counter_shielded", []))
        buechi_unshielded_vals = np.array(data.get("buechi_counter_unshielded", []))
        
        # Compute reward difference for each enforcement parameter
        self.reward_diff_shielded.extend(value_at_initial_state - rewards_shielded)
        self.buechi_shielded_percent.extend(buechi_shielded_vals)
        self.reward_diff_unshielded.extend(value_at_initial_state - rewards_unshielded)
        self.buechi_unshielded_percent.extend(buechi_unshielded_vals)
        
        # Check for each threshold and update sums/counts
        for t in self.thresholds:
            shielded_indices = np.where(
                (value_at_initial_state - rewards_shielded <= t) & 
                (value_at_initial_state - rewards_shielded > t - 0.1)
            )[0]
            unshielded_indices = np.where(
                (value_at_initial_state - rewards_unshielded <= t) & 
                (value_at_initial_state - rewards_unshielded > t - 0.1)
            )[0]
            
            if len(shielded_indices) > 0:
                self.buechi_shielded_sums[t] += np.mean(buechi_shielded_vals[shielded_indices])
                self.counts_shielded[t] += 1
            
            if len(unshielded_indices) > 0:
                self.buechi_unshielded_sums[t] += np.mean(buechi_unshielded_vals[unshielded_indices])
                self.counts_unshielded[t] += 1
    
    def plot_buechi_comparison_close(self):
        """Plot 1: Büchi Counter Comparison (Close)"""
        if not self.buechi_shielded_close:
            print("Warning: No close distance data available for Büchi comparison")
            return
        
        buechi_unshielded_close = np.mean(self.buechi_unshielded_close, axis=0)
        buechi_shielded_close = np.mean(self.buechi_shielded_close, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, buechi_unshielded_close, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, buechi_shielded_close, marker="s", label="Shielded", color="red")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Average Büchi Visits")
        plt.title("Büchi Region Visit Frequency for All Grids (Close)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "buechi_comparison_all_close.png"))
        plt.close()
        print("Generated: buechi_comparison_all_close.png")
    
    def plot_rewards_comparison_close(self):
        """Plot 2: Average Rewards Comparison (Close)"""
        if not self.rewards_shielded_close:
            print("Warning: No close distance data available for rewards comparison")
            return
        
        rewards_unshielded_close = np.mean(self.rewards_unshielded_close, axis=0)
        rewards_shielded_close = np.mean(self.rewards_shielded_close, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, rewards_unshielded_close, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, rewards_shielded_close, marker="s", label="Shielded", color="red")
        if self.value_at_initial_state_close is not None:
            plt.axhline(self.value_at_initial_state_close, color="r", linestyle="--", label="Max Attainable Reward")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Difference of Max Attainable Reward and Average Reward")
        plt.title("Average Rewards Comparison for All Grids (Close)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "rewards_comparison_all_close.png"))
        plt.close()
        print("Generated: rewards_comparison_all_close.png")
    
    def plot_buechi_comparison_far(self):
        """Plot 3: Büchi Counter Comparison (Far)"""
        if not self.buechi_shielded_far:
            print("Warning: No far distance data available for Büchi comparison")
            return
        
        buechi_unshielded_far = np.mean(self.buechi_unshielded_far, axis=0)
        buechi_shielded_far = np.mean(self.buechi_shielded_far, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, buechi_unshielded_far, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, buechi_shielded_far, marker="s", label="Shielded", color="red")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Average Büchi Visits")
        plt.title("Büchi Region Visit Frequency for All Grids (Far)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "buechi_comparison_all_far.png"))
        plt.close()
        print("Generated: buechi_comparison_all_far.png")
    
    def plot_rewards_comparison_far(self):
        """Plot 4: Average Rewards Comparison (Far)"""
        if not self.rewards_shielded_far:
            print("Warning: No far distance data available for rewards comparison")
            return
        
        rewards_unshielded_far = np.mean(self.rewards_unshielded_far, axis=0)
        rewards_shielded_far = np.mean(self.rewards_shielded_far, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, rewards_unshielded_far, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, rewards_shielded_far, marker="s", label="Shielded", color="red")
        if self.value_at_initial_state_far is not None:
            plt.axhline(self.value_at_initial_state_far, color="r", linestyle="--", label="Max Attainable Reward")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Difference of Max Attainable Reward and Average Reward")
        plt.title("Average Rewards Comparison for all Grids(Far)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "rewards_comparison_all_far.png"))
        plt.close()
        print("Generated: rewards_comparison_all_far.png")
    
    def plot_combined_buechi_rewards(self):
        """Plot: Combined Büchi Counter and Average Rewards Comparison"""
        if not self.buechi_shielded_close or not self.buechi_shielded_far:
            print("Warning: Insufficient data for combined plot")
            return
        
        buechi_shielded_close = np.mean(self.buechi_shielded_close, axis=0)
        rewards_shielded_close = np.mean(self.rewards_shielded_close, axis=0)
        buechi_shielded_far = np.mean(self.buechi_shielded_far, axis=0)
        rewards_shielded_far = np.mean(self.rewards_shielded_far, axis=0)
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # Plot Büchi Counter on the left y-axis
        ax1.set_xlabel("Enforcement Parameter", fontsize=20)
        ax1.set_ylabel("Frequency of Büchi Visits", color="tab:pink", fontsize=20)
        ax1.plot(self.enforcement_params, buechi_shielded_close, marker="s", linestyle="--", 
                label="Büchi frequency (Close)", color="pink")
        ax1.plot(self.enforcement_params, buechi_shielded_far, marker="s", 
                label="Büchi frequency (Far)", color="pink")
        ax1.tick_params(axis='y', labelcolor="tab:pink", labelsize=20)
        ax1.tick_params(axis='x', labelsize=20)
        ax1.set_xscale("log")
        ax1.legend(loc="upper left", fontsize=14)
        ax1.grid(True)
        
        # Create a second y-axis for the average rewards
        ax2 = ax1.twinx()
        ax2.set_ylabel("Closeness to Optimal Reward", color="tab:green", fontsize=20)
        ax2.plot(self.enforcement_params, rewards_shielded_close, marker="o", linestyle="--", 
                label="Rewards (Close)", color="green")
        ax2.plot(self.enforcement_params, rewards_shielded_far, marker="o", 
                label="Rewards (Far)", color="green")
        ax2.tick_params(axis='y', labelcolor="tab:green", labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.set_xscale("log")
        ax2.legend(loc="upper right", fontsize=14)
        
        fig.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "buechi_rewards_comparison_all.pdf"))
        plt.close()
        print("Generated: buechi_rewards_comparison_all.pdf")
    
    def plot_buechi_comparison_all(self):
        """Plot 5: Büchi Counter Comparison (All)"""
        if not self.buechi_shielded:
            print("Warning: No data available for overall Büchi comparison")
            return
        
        buechi_unshielded = np.mean(self.buechi_unshielded, axis=0)
        buechi_shielded = np.mean(self.buechi_shielded, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, buechi_unshielded, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, buechi_shielded, marker="s", label="Shielded", color="red")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Average Büchi Visits")
        plt.title("Büchi Region Visit Frequency for All Grids")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "buechi_comparison_all.png"))
        plt.close()
        print("Generated: buechi_comparison_all.png")
    
    def plot_rewards_comparison_all(self):
        """Plot 6: Average Rewards Comparison (All)"""
        if not self.rewards_shielded:
            print("Warning: No data available for overall rewards comparison")
            return
        
        rewards_unshielded = np.mean(self.rewards_unshielded, axis=0)
        rewards_shielded = np.mean(self.rewards_shielded, axis=0)
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.enforcement_params, rewards_unshielded, marker="o", label="Unshielded")
        plt.plot(self.enforcement_params, rewards_shielded, marker="s", label="Shielded", color="red")
        plt.xscale("log")
        plt.xlabel("Enforcement Parameter")
        plt.ylabel("Difference of Max Attainable Reward and Average Reward")
        plt.title("Average Rewards Comparison for All Grids")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "rewards_comparison_all.png"))
        plt.close()
        print("Generated: rewards_comparison_all.png")
    
    def plot_threshold_vs_buechi(self):
        """Plot from percentPlot.py: Threshold vs Büchi frequency"""
        if not self.reward_diff_shielded:
            print("Warning: No data available for threshold vs Büchi plot")
            return
        
        # Compute averages
        average_buechi_shielded = [
            self.buechi_shielded_sums[t] / self.counts_shielded[t] if self.counts_shielded[t] > 0 else 0 
            for t in self.thresholds
        ]
        average_buechi_unshielded = [
            self.buechi_unshielded_sums[t] / self.counts_unshielded[t] if self.counts_unshielded[t] > 0 else 0 
            for t in self.thresholds
        ]
        
        # Configure matplotlib for LaTeX (if available)
        try:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            use_latex = True
        except:
            use_latex = False
            print("Warning: LaTeX not available, using regular text")
        
        plt.figure(figsize=(8, 6))
        
        if use_latex:
            plt.plot(self.thresholds, average_buechi_unshielded, marker="o", linestyle="--", 
                    label=r'\textsc{ApplyNaive}', color='blue')
            plt.plot(self.thresholds, average_buechi_shielded, marker="s", linestyle="-", 
                    label=r'\textsc{ApplySTARs}', color='red')
        else:
            plt.plot(self.thresholds, average_buechi_unshielded, marker="o", linestyle="--", 
                    label='ApplyNaive', color='blue')
            plt.plot(self.thresholds, average_buechi_shielded, marker="s", linestyle="-", 
                    label='ApplySTARs', color='red')
        
        plt.xlabel("Closeness to Max Possible Reward", fontsize=20)
        plt.ylabel("Frequency of Büchi Visits", fontsize=20)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.xticks(self.thresholds, fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(os.path.join(self.plots_dir, "thresholdVSBuechi.pdf"))
        plt.close()
        print("Generated: thresholdVSBuechi.pdf")
    
    def generate_all_plots(self):
        """Generate all plots from both plotter.py and percentPlot.py"""
        print("Loading data...")
        self.load_data()
        
        if not self.enforcement_params:
            print("Error: No enforcement parameters found. Check your data format.")
            return
        
        print("Generating plots...")
        
        # Plots from plotter.py (including commented ones)
        self.plot_buechi_comparison_close()
        self.plot_rewards_comparison_close()
        self.plot_buechi_comparison_far()
        self.plot_rewards_comparison_far()
        self.plot_combined_buechi_rewards()
        self.plot_buechi_comparison_all()
        self.plot_rewards_comparison_all()
        
        # Plot from percentPlot.py
        self.plot_threshold_vs_buechi()
        
        print(f"All plots generated and saved to {self.plots_dir}")


def main():
    """Main function to generate all plots."""
    generator = PlotGenerator()
    generator.generate_all_plots()


if __name__ == "__main__":
    main()
