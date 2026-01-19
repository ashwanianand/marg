# FactoryBot - Shield Evaluation Framework

This directory contains a comprehensive framework for evaluating STARs in FactoryBot instances. The framework allows for dataset generation, experimental evaluation, and visualization of results.

## Directory Structure

```
factoryBot/
├── README.md                   # This file
├── experimentRunner.sh         # Main script to run experiments
├── experiments/                # Core experimental scripts
│   ├── datasetGenerator.py     # Generate custom FactoryBot datasets
│   ├── evaluateShield.py       # Evaluate shield performance on instances
│   └── generate_plot.py        # Generate plots and visualizations
├── results/                    # Created when running experiments
│   ├── outputs/                # JSON results for each evaluated instance
│   ├── plots/                  # Generated visualizations
│   ├── done.txt                # List of completed instances
│   ├── skipped_files.txt       # List of skipped instances
│   └── logs_*.log              # Execution logs with timestamps
```

## Quick Start

### Prerequisites

Ensure you have GNU Parallel and coreutils installed. You can install it using your package manager, e.g., on Ubuntu:

```bash
sudo apt-get install parallel
sudo apt-get install coreutils
```

on macOS:

```bash
brew install coreutils parallel
```

Make sure that virtual environment is activated with the required dependencies installed as described in the main README.

### Creating Datasets
To generate FactoryBot benchmarks:

```bash
python3 experiments/datasetGenerator.py \
    --num_benchmarks 100 \
    --size 5-13 \
    --wall_density 0.2 \
    --region_ratio 0.01 \
    --min_distance_ratio 0.1 \
    --max_distance_ratio 0.2 \
    --output custom_dataset
```

You can adjust the parameters to create small or large datasets, with different wall densities and region placements. The generated files will be saved in the specified output directory.

### Running Experiments

To run experiments on the generated dataset:

```bash
# RECOMMENDED: Run with default settings (100000 steps) 
bash ./experimentRunner.sh -d ./custom_dataset

# Run with custom number of steps
bash ./experimentRunner.sh -d ./custom_dataset -s 500

# Run on a specific dataset directory
bash ./experimentRunner.sh -d ./my_custom_dataset -s 200

# Show help
bash ./experimentRunner.sh -h
```

## Output and Results

### Results Directory Structure

When you run `experimentRunner.sh`, it creates:

```
results/
├── outputs/                    # Individual result files
│   ├── instance1.json         # Detailed statistics for each instance
│   └── instance1.out          # Standard output logs for each instance
├── plots/                     # Generated visualizations
│   └── *.png                  # Various performance and comparison plots
├── done.txt                   # List of successfully processed files
├── skipped_files.txt          # List of files that were skipped
└── logs_YYYYMMDDHHMMSS.log    # Timestamped execution log
```

## Dataset Information

The included `dataset/` directory contains pre-generated FactoryBot instances with varying characteristics:

- **Grid sizes**: 5x5 to 13x13
- **Wall density**: 0.2 (20% of cells are walls)
- **Region ratio**: 0.01 (1% of cells are special regions)
- **Distance configurations**: 
  - Close regions: min_distance_ratio=0.1, max_distance_ratio=0.2
  - Far regions: min_distance_ratio=0.7, max_distance_ratio=0.9

Files are named with parameter encoding: `mpBuechi_s={size}_wd={wall_density}_rr={region_ratio}_midr={min_distance_ratio}_madr={max_distance_ratio}_i={instance_id}.json`


## Experimental Scripts

### 1. experimentRunner.sh

The main orchestrator script that runs parallel evaluation experiments across a dataset.

**Usage:** `./experimentRunner.sh [-s steps] [-d dataset] [-h]`

**Parameters:**
- `-s steps`: Number of simulation steps to run per instance (default: 100)
- `-d dataset`: Path to dataset directory containing .json files (default: ".")
- `-h`: Show help information

**What it does:**
- Creates a `results` directory with subdirectories for outputs, logs, and tracking
- Processes all .json files in the specified dataset directory in parallel
- Tracks completed files to allow resuming interrupted experiments
- Generates logs with timestamps for each processed file
- Automatically creates plots after all evaluations complete

### 2. experiments/datasetGenerator.py

Generates FactoryBot benchmark instances with configurable parameters.

**Usage:** 
```bash
python3 experiments/datasetGenerator.py \
    --num_benchmarks <int> \
    --size <min-max> \
    --wall_density <float> \
    --region_ratio <float> \
    --min_distance_ratio <float> \
    --max_distance_ratio <float> \
    [--probabilistic_ratio <float>] \
    [--ignore_attributes <bool>] \
    --output <directory>
```

**Required Parameters:**
- `--num_benchmarks`: Number of benchmark instances to generate
- `--size`: Grid size range (e.g., "10-100" for grids between 10x10 and 100x100)
- `--wall_density`: Density of walls in the grid (0.0 to 1.0)
- `--region_ratio`: Ratio of special regions in the grid
- `--min_distance_ratio`: Minimum distance ratio for region placement
- `--max_distance_ratio`: Maximum distance ratio for region placement
- `--output`: Output directory for generated benchmarks

**Optional Parameters:**
- `--probabilistic_ratio`: Probability of assigning probabilistic cells (default: 0)
- `--ignore_attributes`: Whether to ignore CSS attributes (default: True)

**What it does:**
- Generates random grid environments with Büchi acceptance conditions
- Creates mean payoff objectives with configurable parameters
- Outputs JSON files containing grid structure, robot position, and environment data
- Creates a temporary directory for overlay files

**Example:**
```bash
# Generate 500 small grids with close regions
python3 experiments/datasetGenerator.py \
    --num_benchmarks 500 \
    --size 10-20 \
    --wall_density 0.2 \
    --region_ratio 0.01 \
    --min_distance_ratio 0.1 \
    --max_distance_ratio 0.2 \
    --output small_close_dataset

# Generate 500 large grids with distant regions  
python3 experiments/datasetGenerator.py \
    --num_benchmarks 500 \
    --size 50-100 \
    --wall_density 0.2 \
    --region_ratio 0.01 \
    --min_distance_ratio 0.7 \
    --max_distance_ratio 0.9 \
    --output large_far_dataset
```

### 3. experiments/evaluateShield.py

Evaluates shield performance on individual FactoryBot instances by comparing shielded vs unshielded robot behavior.

**Usage:** `python3 experiments/evaluateShield.py <steps> <json_file_path> <output_file_path>`

**Parameters:**
- `steps`: Number of simulation steps to run
- `json_file_path`: Path to the input FactoryBot instance (.json file)
- `output_file_path`: Path where evaluation results will be saved (.json file)

**What it does:**
- Loads a grid environment from the JSON file
- Converts the grid to a game graph with Büchi acceptance conditions
- Runs robot simulations both with and without shield protection
- Measures key metrics:
  - Büchi region visitation frequency
  - Average rewards collected
  - Decision-making time per step
- Outputs detailed statistics comparing shielded vs unshielded performance

**Output Format:** JSON file containing statistical results for further analysis

### 4. experiments/generate_plot.py

Generates comprehensive visualizations from experimental results.

**Usage:** `python3 experiments/generate_plot.py [results_dir] [plots_dir]`

**Parameters:** 
- `results_dir`: Directory containing JSON result files (default: "./results/outputs/")
- `plots_dir`: Directory where plots will be saved (default: "./plots/")

**Note:** Currently the script uses hardcoded default paths, but the experimentRunner.sh attempts to pass custom directories.

**What it does:**
- Aggregates results from all JSON files in the results directory
- Generates multiple types of plots:
  - Performance comparison plots (shielded vs unshielded)
  - Büchi region visitation analysis
  - Reward distribution comparisons
  - Decision time analysis
  - Threshold analysis plots
- Saves all plots to the specified plots directory
- Provides unified plotting functionality combining multiple visualization modules


## Advanced Usage

### Resuming Interrupted Experiments

The framework automatically tracks completed experiments in `results/done.txt`. If an experiment is interrupted, simply re-run the same command and it will skip already processed files.

### Parallel Processing

The experimentRunner.sh uses GNU parallel to process multiple instances simultaneously, utilizing all available CPU cores (`-j$(nproc)`).

### Custom Analysis

Individual result files in `results/outputs/` contain detailed JSON data that can be used for custom analysis beyond the provided plotting functionality.

## Dependencies

- Python 3.x
- GNU parallel (for parallel processing)
- Standard Unix tools (bash, find, etc.)

## Troubleshooting

- **No .json files found**: Ensure your dataset directory contains valid FactoryBot instance files
- **Permission denied**: Make sure `experimentRunner.sh` is executable (`chmod +x experimentRunner.sh`)
- **Memory issues**: For large datasets, consider reducing the number of parallel jobs or processing in smaller batches