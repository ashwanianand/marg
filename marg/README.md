# MARG (Monitoring and Adaptive Runtime Guide)

A framework to shield learned policies to ensure satisfaction of additional omega-regular properties. The tool uses Strategy-Template-based Adaptive Runtime Shield (STARs) to guide the policies. 


The tool is based on the paper:
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

## Features

- **Multiple Game Solvers**: Supports Parity, Mean-Payoff, and Stochastic Parity games
- **Shield Implementation**: Runtime safety enforcement through action distribution modification
- **Web-Based Visualization**: Interactive FactoryBot grid interface for real-time robot simulation
- **Liveness Guarantees**: Ensures both safety (avoiding unsafe states) and liveness (visiting goal states)

## Installation

### Requirements

- Python 3.7+
- Flask
- Flask-SocketIO
- tqdm

### Setup

```bash
# Clone or navigate to the project directory
cd marg

# Install dependencies
pip install flask flask-socketio tqdm
```

## Quick Start

### Running the Web Interface

```bash
python -m web.robot_grid
```

Then open your browser and navigate to `http://127.0.0.1:5000`

### Web Interface Controls

1. **Generate Grid**: Create a random grid layout
2. **Set Robot Position**: Click on a cell to set the robot's initial position
3. **Configure Cells**: 
   - Mark cells as unsafe/safe
   - Set cell priorities for parity games
   - Assign cell ownership (player 0/1)
4. **Compute Shield**: Generate safety templates and winning regions
5. **Toggle Shield**: Enable/disable shield enforcement
6. **Start Robot**: Begin robot simulation

## Project Structure

```
marg /
├── grid/
│  ├── __init__.py
│  └── gridGenerator.py     # Grid layout generation
├── shield/
│  ├── datastructures/      # Necessary datastructures
│  ├── solver/              # Various shield computation algorithms
│  ├── __init__.py
│  ├── shield.py            # Shield implementation
│  └── Utils.py             # Utility functions
├── web/                    
│  ├── static/
│  ├── templates/
│  └── robot_grid.py        # Flask web application
├── __init__.py
└── README.md
```

## Usage Examples

### Programmatic Usage

```python
from shield.datastructures.GameGraphs import TwoPlayerGameGraph
from shield.solver.Parity import compute_template_in_two_player_graph
from shield.shield import Shield
from shield.datastructures.Robot import Robot

# Create game graph from grid
game_graph = grid_to_game_graph(grid_layout)

# Compute safety templates
winning_region, safety_template, co_live_template, group_liveness_template = \
    compute_template_in_two_player_graph(game_graph, index=1)

# Initialize shield
shield = Shield(
    game_graph=game_graph,
    unsafe_edges=safety_template,
    colive_edges=co_live_template,
    live_groups=group_liveness_template,
    enforcement_parameter=0.3
)

# Create robot with random strategy
robot = Robot(game_graph, initial_position)
robot.set_randomized_strategy()

# Apply shield to robot's action distribution
action_distribution = robot.get_strategy_distribution(robot.robot_position)
shielded_action, shielded_state = shield.sample_next_action_and_state(
    robot.robot_position, 
    action_distribution
)
```

## Game-Theoretic Concepts

### Parity Games
- Vertices have priorities
- Player 0 (robot) wins if highest priority visited infinitely often is even
- Used for safety and liveness properties

### Mean-Payoff Games
- Vertices have weights
- Player 0 aims to maximize average payoff over infinite plays
- Useful for optimizing quantitative objectives

### Shield Enforcement
The shield modifies the robot's action distribution to ensure:
1. **Safety**: Never transition to unsafe states
2. **Co-liveness**: Visit certain states only finitely often
3. **Group Liveness**: Visit goal regions infinitely often

## Configuration

Key parameters in the web interface:
- **Enforcement Parameter** (0-1): Weight for shield corrections
- **Wall Density**: Probability of walls in random grid generation
- **Grid Size**: Dimensions of the navigation grid
- **Cell Properties**: Safety markers, priorities, ownership

## Troubleshooting

### Import Errors
Ensure you're running from the parent directory:
```bash
python -m web.robot_grid  # Not: python web/robot_grid.py
```

### Port Already in Use
Change the port in `robot_grid.py`:
```python
socketio.run(app, debug=False, port=5001)
```

### Browser Not Opening
Manually navigate to `http://127.0.0.1:5000` after starting the server.
