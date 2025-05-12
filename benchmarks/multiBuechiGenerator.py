from grid.gridGenerator import random_multi_objective_grid_generator as multiBuechi
import argparse
import json
import os
import random
from tqdm import tqdm

"""
Generates multibuechi benchmarks to test frequency of visiting buechi regions while maintaining mean payoff objectives

Example command:
python3 multiBuechi.py --num_benchmarks 100 --size 10-100 --wall_density 0.3 --num_objectives 5 --region_ratio 0.1 --output benchmarks
"""

def generate_benchmarks(parameters):
    num_benchmarks = parameters['num_benchmarks']
    size_range = parameters['size']
    wall_density = parameters['wall_density']
    num_objectives = parameters['num_objectives']
    region_ratio = parameters['region_ratio']
    ignore_attributes = parameters['ignore_attributes']
    probabilistic_ratio = parameters['probabilistic_ratio']
    output_dir = parameters['output']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in tqdm(range(num_benchmarks), desc="Generating benchmarks"):
        size = random.randint(size_range[0], size_range[1])

        file_name = f'multiBuechi_s={size}_wd={wall_density}_nbo{num_objectives}_rr={region_ratio}_i={i}{random.randint(0,50)}.json'
        grid, robot_position, overlays = multiBuechi(size, wall_density=wall_density, probabilitstic_nodes=probabilistic_ratio, num_of_buechi=num_objectives, buechi_probability=region_ratio, ignore_attributes=ignore_attributes)
        
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump({'size': size, 'grid': grid, 'robot_position': robot_position}, f)
        

            
        temp_file = os.path.join(output_dir, f'temp/{file_name}')
        with open(temp_file, 'w') as f:
            for overlay in overlays:
                for row in overlay:
                    f.write(' '.join(map(str, row)) + '\n')
                f.write('\n\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MPBuechi benchmarks.')
    parser.add_argument('--num_benchmarks', type=int, required=True, help='Number of benchmarks to generate')
    parser.add_argument('--size', type=str, required=True, help='Range of grid sizes (e.g., 10-100)')
    parser.add_argument('--wall_density', type=float, required=True, help='Density of walls in the grid')
    parser.add_argument('--num_objectives', type=int, required=True, help='Number of buechi objectives')
    parser.add_argument('--region_ratio', type=float, required=True, help='Ratio of regions in the grid')
    parser.add_argument('--probabilistic_ratio', type=float, required=False, default=0, help='Probability of assigning a probabilistic cell')
    parser.add_argument('--ignore_attributes', type=bool, required=False, default=True, help='Should ignore css attributes')
    parser.add_argument('--output', type=str, required=True, help='Output directory for the generated benchmarks')

    args = parser.parse_args()
    
    size_range = list(map(int, args.size.split('-')))

    # create a temp folder in the output directory
    temp_dir = os.path.join(args.output, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    parameters = {
        'num_benchmarks': args.num_benchmarks,
        'size': size_range,
        'wall_density': args.wall_density,
        'num_objectives': args.num_objectives,
        'region_ratio': args.region_ratio,
        'probabilistic_ratio': args.probabilistic_ratio,
        'ignore_attributes': args.ignore_attributes,
        'output': args.output
    }

    generate_benchmarks(parameters)

