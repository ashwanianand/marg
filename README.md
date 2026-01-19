# Follow the STARs: Dynamic ω-Regular Shielding of Learned Probabilistic Policies

This code base contains the implementation of MARG (Monitoring and Adaptive Runtime Guide) for shielding trained policies. We also include the code for the three experiments that we report in the paper: FactoryBot, OvercookedAI and LunarLander.

The repository is based on the paper:
```latex
@misc{anand2025followstarsdynamicomegaregular,
      title={Follow the STARs: Dynamic $\omega$-Regular Shielding of Learned Policies}, 
      author={Ashwani Anand and Satya Prakash Nayak and Ritam Raha and Anne-Kathrin Schmuck},
      year={2025},
      eprint={2505.14689},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.14689}, 
}
```

## Directory Structure

```
submission/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── marg/                            # MARG implementation
│   ├── grid/                        # Grid-based utilities
│   ├── shield/                      # Shielding components
│   └── web/                         # Web interface utilities
├── factoryBot/                      # FactoryBot experiment
│   ├── README.md                    # FactoryBot experiment 
│   ├── experiments/                 # Experiment scripts
│   └── experimentRunner.sh          # Script to run experiments
├── lunarLander/
│   ├── CustomEnvs/                  # Custom lunar lander environment 
│   ├── experiments/                 # Experiment scripts
│   ├── results/                     # Experiment results
│   ├── experimentRunner.sh          # Script to run full experiments
│   ├── experimentRunnerSmall.sh     # Script to run small batch of experiments
│   └── README.md                    # How to run Lunar Lander experiments
└── overcookedAI/
    ├── layouts/                     # Overcooked layouts
    ├── lib/                         # Necessary libraries
    ├── pestel/                      # pestel tool for template computation
    ├── results/                     # Experiment results
    ├── genPlot.py                   # Script to generate final plot
    ├── README.md                    # How to run overcooked experiments
    ├── run.py                       # Run shield computation on individual instances
    └── run_all.sh                   # Run whole experiment

```

## Setup

### Prerequisites

Before running any scripts, create and activate a Python virtual environment to avoid dependency conflicts:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   # On Linux/macOS:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python3 -c "import gymnasium; import stable_baselines3; print('Dependencies installed successfully')"
   ```

**Note:** Always ensure the virtual environment is activated before running any scripts. You can deactivate it later with `deactivate`.

## MARG Implementation

If you only want to use the shield, the core MARG (Monitoring and Adaptive Runtime Guide) implementation is located in the `marg/` directory. For detailed instructions on how to use the shield, see:
**[marg/README.md](marg/README.md)**

## Running Experiments

This repository contains two main experiments corresponding to the paper's evaluation:

### FactoryBot Experiment

The FactoryBot experiment demonstrates MARG's effectiveness in a grid-world factory automation scenario. For detailed instructions on running this experiment, see:

**[factoryBot/README.md](factoryBot/README.md)**

### LunarLander Experiment

The LunarLander experiment shows MARG's application to continuous control tasks in the Gymnasium's Lunar Lander environment. For detailed instructions on running this experiment, see:

**[lunarLander/README.md](lunarLander/README.md)**

### OvercookedAI Experiment

The OvercookedAI experiment applies MARG to a cooperative cooking task. For detailed instructions on running this experiment, see:

**[overcookedAI/README.md](overcookedAI/README.md)**
