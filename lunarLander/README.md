# LunarLander Shielded Reinforcement Learning

This directory contains a complete framework for training and evaluating reinforcement learning agents on our Custom LunarLander environment with shields. 

## Directory Structure

```
lunarLander/
├── CustomEnvs/                   # Custom Gymnasium environments
├── experimentRunner.sh           # Main experiment execution script
├── experimentRunnerSmall.sh      # Experiment execution script for small batch
├── experiments/                  # Core experiment scripts
│   ├── trainer.py                # Model training script
│   ├── model_runner.py           # Model evaluation and video recording
│   ├── record_videos.py          # Parallel video generation for multiple configurations
│   ├── landerToGraph.py          # Environment-to-graph conversion for shield generation
│   └── report_generator.py       # Analysis and report generation
├── results/                      # Generated results directory
│   └── YYYYMMDD-HHMMSS_tag/      # Timestamped training runs
└── __init__.py
```

## Quick Start

Make sure that virtual environment is activated with the required dependencies installed as described in the main README.

**Note:** The experiment was run on a 32-core Debian machine with an Intel Xeon E5-V2 CPU (3.3 GHz) and up to 256 GB of RAM. It took us around 14 hours to run the experiment. 

You may modify the configuration in `record_videos.py` script to reduce the number of enforcement parameters the shields are run with and the number of seeds to be tested. But the graphs may vary from the ones that are reported in the paper.

1. **Using Pre-trained Model with Small Benchmark Set - RECOMMENDED**: Run experiments with the provided model:
   ```bash
   bash ./experimentRunnerSmall.sh --tag review
   ```

2. **Using Pre-trained Model for Full Benchmark Set**: Run full experiments with the provided model:
   ```bash
   bash ./experimentRunner.sh --tag review
   ```

3. **Training Your Own Model**: Train a new model with a custom tag:
   ```bash
   python3 experiments/trainer.py --env CustomLunarLander-v0 --tag mytag --alg ppo --trn-time 100000
   ./experimentRunner.sh --tag mytag
   ```


### Pre-provided Model

The directory `20250804-001718_review/` contains a pre-trained PPO model that you can use immediately:
- **Model**: `models/CustomLunarLander-v0.zip`
- **Training**: Completed with 100,000 timesteps using PPO algorithm
- **Usage**: Run `./experimentRunner.sh --tag review` to evaluate this model


## Results Directory Structure

When you train a model using `trainer.py` with a tag, or run experiments on existing models, the following directory structure is created:

```
results/
└── YYYYMMDD-HHMMSS_tag/     # Timestamped directory with experiment tag
    ├── models/              # Trained RL models
    │   └── CustomLunarLander-v0.zip
    ├── videos/              # Recorded evaluation videos
    │   └── seed_X/          # Videos organized by seed
    │       └── param_Y/     # Videos organized by enforcement parameter
    │           ├── CustomLunarLander-v0_seedX_unshielded.mp4
    │           ├── CustomLunarLander-v0_seedX_shielded_safety.mp4
    │           ├── CustomLunarLander-v0_seedX_shielded_uni_True.mp4
    │           └── CustomLunarLander-v0_seedX_shielded_uni_False.mp4
    ├── grids/               # Generated graph representations (created by landerToGraph.py)
    ├── graphs/              # Shield synthesis artifacts
    └── reports/             # Analysis reports and statistics
        └── summary_report.csv
```

## Workflow

### Complete Experiment Pipeline

1. **Train Model** (optional if using pre-trained):
   ```bash
   python3 experiments/trainer.py --env CustomLunarLander-v0 --tag myexperiment
   ```

2. **Run Complete Evaluation**:
   ```bash
   ./experimentRunner.sh --tag myexperiment
   ```
   or
   **Run Evaluation on small benchmark set**:
   ```bash
   ./experimentRunnerSmall.sh --tag myexperiment
   ```

This executes:
- `record_videos.py`: Generates videos for multiple seeds and parameters
- `report_generator.py`: Analyzes results and creates reports


## Experiments Directory Scripts

### 1. `trainer.py` - Model Training

Trains reinforcement learning models using various algorithms on the LunarLander environment.

**Arguments:**
- `--env` (required): Gymnasium environment name (e.g., `CustomLunarLander-v0`)
- `--tag`: Experiment tag for organizing results (creates timestamped directory)
- `--mod-dir`: Directory to save model (used when no tag specified)
- `--vid-dir`: Directory to save videos (used when no tag specified)
- `--alg`: RL algorithm (`ppo`, `dqn`, `a2c`, `sac`, `td3`) [default: `ppo`]
- `--trn-time`: Number of training timesteps [default: `100000`]
- `--seed`: Random seed for reproducibility [default: `42`]

**Example:**
```bash
python3 experiments/trainer.py --env CustomLunarLander-v0 --tag experiment1 --alg ppo --trn-time 200000 --seed 123
```

### 2. `model_runner.py` - Model Evaluation and Video Recording

Evaluates trained models and records videos with various shield configurations.

**Arguments:**
- `--tag` (required): Tag to find the model directory
- `--env` (required): Gymnasium environment name
- `--seed`: Random seed [default: `42`]
- `--alg`: RL algorithm [default: `ppo`]
- `--grid_size`: Grid size for shielded environments [default: `50`]
- `--shield`: Enable safety shielding (flag)
- `--universal`: Use universal counter for liveness properties (flag)
- `--safety`: Generate video with safety-only shield (flag)
- `--all`: Generate all video variants (unshielded, safety-only, liveness with universal/individual counters) (flag)

**Example:**
```bash
python3 experiments/model_runner.py --tag review --env CustomLunarLander-v0 --grid_size 60 --shield --universal --all
```

### 3. `record_videos.py` - Parallel Video Generation

Records videos in parallel for multiple seeds and enforcement parameters, generating comprehensive evaluation datasets.

**Arguments:**
- `--workers`: Maximum parallel workers [default: CPU count / 2]
- `--meta-seed`: Meta seed for reproducible random seed generation [default: `42`]
- `--num-seeds`: Number of random seeds to generate [default: `200`]
- `--tags`: List of tags to process [default: `["review"]`]
- `--tag`: Single tag suffix (alternative to `--tags`) [default: `"review"`]

**Generated Videos:**
For each seed and enforcement parameter (0.01-0.5), creates 4 video variants:
1. **Unshielded**: No safety intervention
2. **Safety-only Shield**: Safety constraints only
3. **Liveness + Universal Counter**: Full shield with universal counter
4. **Liveness + Individual Counter**: Full shield with individual counter

**Example:**
```bash
python3 experiments/record_videos.py --workers 8 --meta-seed 42 --num-seeds 100 --tag review
```

### 4. `landerToGraph.py` - Graph Generation

Converts the LunarLander environment into a discrete graph representation for shield synthesis.

**Arguments:**
- `--n`: Number of grid cells in each dimension [default: `10`]
- `--seed`: Random seed for reproducibility [default: `42`]
- `--output_file`: File to save the graph representation [default: `"lunar_lander_graph.json"`]

**Functionality:**
- Discretizes the continuous state space into a grid
- Identifies safe/unsafe regions based on landing zones
- Creates transition graph for formal verification
- Generates visualization of the discretized environment

**Example:**
```bash
python3 experiments/landerToGraph.py --n 60 --seed 42 --output_file lander_graph_60.json
```

### 5. `report_generator.py` - Analysis and Reporting

Analyzes recorded videos and generates comprehensive performance reports.

**Arguments:**
- `--tag`: Tag suffix of the training run directory [default: `"review"`]
- `--base_dir`: Base directory for results [default: `"results"`]

**Generated Outputs:**
- Success rate analysis by shield configuration
- Performance comparison charts
- Statistical summaries of landing outcomes
- Consolidated CSV reports

**Example:**
```bash
python3 experiments/report_generator.py --tag review --base_dir results
```


### Custom Evaluation

For fine-grained control, run scripts individually:

```bash
# Generate graph representation
python3 experiments/landerToGraph.py --n 60 --output_file custom_graph.json

# Record specific videos
python3 experiments/model_runner.py --tag myexperiment --env CustomLunarLander-v0 --all

# Generate custom reports
python3 experiments/report_generator.py --tag myexperiment
```

## Dependencies

- gymnasium
- stable-baselines3
- torch
- matplotlib
- pandas
- tqdm
- numpy
- Custom environments and MARG framework (included)

For full dependency list, see `requirements.txt` in the workspace root.
