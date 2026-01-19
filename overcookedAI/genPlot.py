import json
import os
import matplotlib.pyplot as plt
import numpy as np

# === Configuration ===
# Replace with your JSON file path
current_directory = os.path.dirname(os.path.abspath(__file__))
json_dir = os.path.join(current_directory, "./results")

# === Load JSON data from every json file in the directory ===
data = {}
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        with open(os.path.join(json_dir, filename), 'r') as f:
            file_data = json.load(f)
            data.update(file_data)

# === Extract data ===
x = [v["num_states"] / 1e6 for v in data.values()]
y = [v["strategy_time"]/60 for v in data.values()]

# === Plot with a more compact style for PDF inclusion ===
plt.figure(figsize=(3, 3))  # Smaller figure size
plt.scatter(x, y, color='blue', edgecolor='black', s=40, zorder=5)
plt.plot(x, y, color='blue', linestyle='--', linewidth=1, zorder=3)
plt.xlabel("Number of States (millions)", fontsize=11)
plt.ylabel("STARs Computation Time (mins)", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout(pad=0.5)

plt.grid(True, linestyle='--', alpha=0.6)

# === Optionally save ===
plt.savefig("overcooked_plot.png", dpi=300)
