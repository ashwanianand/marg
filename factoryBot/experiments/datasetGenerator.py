import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from marg.grid.gridGenerator import random_grid_with_buechi_region_meanpayoff_region as MPBuechi
from marg.grid.gridGenerator import random_grid_with_buechi_region_meanpayoff_region as MPBuechi
import argparse
import json
import os
import random
from tqdm import tqdm

"""
Generates MPBuechi benchmarks to compare with standard techniques of using just safety shields

Example command:
python3 experiments/datasetGenerator.py --num_benchmarks 500 --size 10-100 --wall_density 0.2 --region_ratio 0.01 --min_distance_ratio 0.1 --max_distance_ratio 0.2 --probabilistic_ratio 0.05 --output anotherDataset

python3 experiments/datasetGenerator.py --num_benchmarks 500 --size 10-100 --wall_density 0.2 --region_ratio 0.01 --min_distance_ratio 0.7 --max_distance_ratio 0.9 --probabilistic_ratio 0.05 --output anotherDataset

"""

def generate_benchmarks(parameters):
    num_benchmarks = parameters['num_benchmarks']
    size_range = parameters['size']
    wall_density = parameters['wall_density']
    region_ratio = parameters['region_ratio']
    min_distance_ratio = parameters['min_distance_ratio']
    max_distance_ratio = parameters['max_distance_ratio']
    ignore_attributes = parameters['ignore_attributes']
    probabilistic_ratio = parameters['probabilistic_ratio']
    output_dir = parameters['output']



    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in tqdm(range(num_benchmarks), desc="Generating benchmarks"):
        size = random.randint(size_range[0], size_range[1])

        file_name = f'mpBuechi_s={size}_wd={wall_density}_rr={region_ratio}_midr={min_distance_ratio}_madr={max_distance_ratio}_i={i}{random.randint(0,50)}.json'
        min_distance = size * min_distance_ratio
        max_distance = size * max_distance_ratio
        grid, robot_position, overlay = MPBuechi(size, wall_density, region_ratio, min_distance, max_distance, ignore_attributes, probabilistic_ratio)
        
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w') as f:
            json.dump({'size': size, 'grid': grid, 'robot_position': robot_position}, f)
            
        temp_file = os.path.join(output_dir, f'temp/{file_name}')
        with open(temp_file, 'w') as f:
            for row in overlay:
                f.write(' '.join(map(str, row)) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MPBuechi benchmarks.')
    parser.add_argument('--num_benchmarks', type=int, required=True, help='Number of benchmarks to generate')
    parser.add_argument('--size', type=str, required=True, help='Range of grid sizes (e.g., 10-100)')
    parser.add_argument('--wall_density', type=float, required=True, help='Density of walls in the grid')
    parser.add_argument('--region_ratio', type=float, required=True, help='Ratio of regions in the grid')
    parser.add_argument('--min_distance_ratio', type=float, required=True, help='Minimum distance ratio')
    parser.add_argument('--max_distance_ratio', type=float, required=True, help='Maximum distance ratio')
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
        'region_ratio': args.region_ratio,
        'min_distance_ratio': args.min_distance_ratio,
        'max_distance_ratio': args.max_distance_ratio,
        'probabilistic_ratio': args.probabilistic_ratio,
        'ignore_attributes': args.ignore_attributes,
        'output': args.output
    }

    generate_benchmarks(parameters)

