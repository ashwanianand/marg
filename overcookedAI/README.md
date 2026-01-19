# Overcooked AI evaluation for scalability
This directory contains all the necessary tools to run evaluate scalability of STARs. 


### Directory Structure

```
overcookedAI/
├── hoa_files/                      # Will contain the game graphs in HOA format
├── layouts/                        # Layouts for different instances
├── lib/                            # Necessary packages
│  ├── game.py
│  ├── generate_hoa.py
│  └── generate_layout.py
├── pestel/                         # Pestel for templates
├── results/                        # Stores the final results file
│  └── overall.json
├── templates/                      # Computed templates storage
├── genPlot.py                      # Plot generation script
├── overcooked_plot.png             # Final plot after evaluation
├── README.md                       
├── run.py                          # Script to run individual instances
└── run_all.sh                      # Full script
```


### Running Experiments
To run the complete experiment pipeline, execute the following command:

```bash
bash ./run_all.sh
```

This will run the experiments and generate the final plot `overcooked_plot.png`.