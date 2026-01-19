import itertools
import random
import math
import pprint

"""Cell states:
0: unmarked
1: unsafe
2: buechi, visit infinitely often
3: co-Buechi, visit finitely often
"""

"""Cell weights:
first coordinate: mean payoff weight
others: parity
"""


def random_grid_layout(size, wall_density=0.3):
    actual_size = 2 * size + 1
    grid = []
    for row in range(actual_size):
        cells_in_row = []
        for column in range(actual_size):
            is_wall = False
            is_wall_editable = False
            cell_color = 0
            cell_owner = 0
            cell_attributes = ['cell']
            random_cell_weight = []

            # Add walls to the border and the cell corners
            if (row % 2 == 0 and column % 2 == 0) or row == 0 or column == 0 or row == actual_size - 1 or column == actual_size - 1:
                cell_attributes.append('wall')
                is_wall = True
            elif row % 2 == 0 or column % 2 == 0:
                # Randomly add walls between the cells
                if random.random() < wall_density and 'wall' not in cell_attributes:
                    cell_attributes.append('wall')
                    is_wall = True
                
                if row % 2 != 0 or column % 2 != 0:
                    is_wall_editable = True

            # Add horizontal and vertical walls to the cells
            if (row % 2 == 0):
                cell_attributes.append('horizontal')
            if (column % 2 == 0):
                cell_attributes.append('vertical')

            # Make all cells unmarked except the cell corners
            if row % 2 != 0 and column % 2 != 0:
                cell_attributes.append('unmarked')

            cells_in_row.append({
                'attributes': ' '.join(cell_attributes),
                'is_wall': is_wall,
                'is_wall_editable': is_wall_editable,
                'cell_color': cell_color, 
                'owner': cell_owner,
                'weight': random_cell_weight
            })
        grid.append(cells_in_row)
    return grid


def iterate_over_cells(size):
    actual_size = 2 * size + 1
    return itertools.product(range(1, actual_size, 2), repeat=2)


def random_grid_generator(size, wall_density=0.3):
    grid = random_grid_layout(size, wall_density)

    for (row, column) in iterate_over_cells(size):
        random_cell_weight = []

        # random integer weight between -size and size
        random_cell_weight.append(random.randint(-size, size))
        grid[row][column]['weight'] = random_cell_weight
            
    # pick a random cell and add robot attribute
    robot_row = random.randint(0, size - 1) * 2 + 1
    robot_column = random.randint(0, size - 1) * 2 + 1
    robot_position = (robot_row, robot_column)
    grid[robot_row][robot_column]['attributes'] = grid[robot_row][robot_column]['attributes'] + ' robot'
    return grid, robot_position

def random_grid_with_buechi_region_meanpayoff_region(size, wall_density=0.3, region_ratio=0.2, min_distance=5, max_distance=10, ignore_attributes=True, probabilistic_ratio=0.2):
    grid = random_grid_layout(size, wall_density)
    region_size = int(region_ratio * size * size)
    if region_size < 1:
        region_size = 1
    grid_overlay = create_grid_overlay(size, region_size, min_distance, max_distance)
    
    players = [0, 2]
    probabilities = [1 - probabilistic_ratio, probabilistic_ratio]

    for (row, column) in iterate_over_cells(size):
        row_in_overlay = row // 2
        column_in_overlay = column // 2
        random_cell_weight = []

        if grid_overlay[row_in_overlay][column_in_overlay] == 1:
            random_cell_weight.append(1)
        else:
            random_cell_weight.append(0)

        if grid_overlay[row_in_overlay][column_in_overlay] == 2:
            random_cell_weight.append(2)
        else:
            random_cell_weight.append(1)

        grid[row][column]['weight'] = random_cell_weight
        
        # randomly assign owner
        grid[row][column]['owner'] = random.choices(players, weights=probabilities)[0]
    
    if ignore_attributes:
        for row in grid:
            for cell in row:
                # delete the key 'attributes' from the dictionary
                cell.pop('attributes', None)

    # pick a random cell and add robot attribute
    robot_row = random.randint(0, size - 1) * 2 + 1
    robot_column = random.randint(0, size - 1) * 2 + 1
    robot_position = (robot_row, robot_column)
    if not ignore_attributes:
        grid[robot_row][robot_column]['attributes'] = grid[robot_row][robot_column]['attributes'] + ' robot'
    return grid, robot_position, grid_overlay


def random_multi_objective_grid_generator(size, wall_density=0.3, probabilitstic_nodes=0.2, num_of_buechi=1, buechi_probability=0.1, num_of_cobuechi=0, cobuechi_probability=0.1, ignore_attributes=True):

    grid = random_grid_layout(size, wall_density)
    players = [0, 2]
    probabilities = [1 - probabilitstic_nodes, probabilitstic_nodes]

    overlays = []
    for _ in range(num_of_buechi):
        region_size = int(random.uniform(buechi_probability, buechi_probability + 0.05) * size * size)
        if region_size < 1:
            region_size = 1
        overlay = create_grid_overlay_one_region(size, region_size, 1)
        overlays.append(overlay)
    for _ in range(num_of_cobuechi):
        region_size = int(random.uniform(cobuechi_probability, cobuechi_probability + 0.05) * size * size)
        if region_size < 1:
            region_size = 1
        overlay = create_grid_overlay_one_region(size, region_size, 1)
        overlays.append(overlay)

    # iterate over all rows and columns such that row % 2 != 0 and column % 2 != 0
    # and assign owners, meanpayoff weights, buechi and co-buechi states

    for (row, column) in iterate_over_cells(size):
        random_cell_weight = []

        # random integer weight between -size and size
        random_cell_weight.append(random.randint(-size, size))

        # Randomly assign buechi and co-buechi states
        for i in range(num_of_buechi):
            if overlays[i][row // 2][column // 2] == 1:
                random_cell_weight.append(2)
            else:
                random_cell_weight.append(1)
        for i in range(num_of_cobuechi):
            if overlays[num_of_buechi + i][row // 2][column // 2] == 1:
                random_cell_weight.append(3)
            else:
                random_cell_weight.append(2)
        
        # randomly assign owner
        cell_owner = random.choices(players, weights=probabilities)[0]

        grid[row][column]['owner'] = cell_owner
        grid[row][column]['weight'] = random_cell_weight
    
    if ignore_attributes:
        for row in grid:
            for cell in row:
                # delete the key 'attributes' from the dictionary
                cell.pop('attributes', None)
            
    # pick a random cell and add robot attribute
    robot_row = random.randint(0, size - 1) * 2 + 1
    robot_column = random.randint(0, size - 1) * 2 + 1
    robot_position = (robot_row, robot_column)
    if not ignore_attributes:
        grid[robot_row][robot_column]['attributes'] = grid[robot_row][robot_column]['attributes'] + ' robot'
    return grid, robot_position, overlays


def random_buechi_objectives(num_of_buechi, buechi_probability):
    objectives = []
    for _ in range(num_of_buechi):
        if random.random() < buechi_probability:
            objectives.append(2)
        else:
            objectives.append(1)
    return objectives

def random_cobuechi_objectives(num_of_cobuechi, cobuechi_probability):
    objectives = []
    for _ in range(num_of_cobuechi):
        if random.random() < cobuechi_probability:
            objectives.append(3)
        else:
            objectives.append(2)
    return objectives

def choose_random_subset_of_cells(grid, subset_size):
    actual_size = len(grid) - 1 // 2
    cells = []
    for row in range(actual_size):
        for column in range(actual_size):
            cells.append((2 * row, 2* column))
    
    return random.sample(cells, subset_size)




"""Overlay generator for the benchmarks to compare with standard techniques
0 means no objective,
1 for region with +ve mean payoff weight,
2 for region with Buechi objective
"""

import random
import math

def get_neighbors(grid, x, y):
    """Returns valid neighboring coordinates (4-directional)"""
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def place_region(grid, region_size, value):
    """Places a connected region of a given size with a specific value"""
    rows, cols = len(grid), len(grid[0])
    while True:
        start_x, start_y = random.randint(0, rows-1), random.randint(0, cols-1)
        if grid[start_x][start_y] == 0:
            break

    region = [(start_x, start_y)]
    grid[start_x][start_y] = value
    while len(region) < region_size:
        x, y = random.choice(region)
        neighbors = get_neighbors(grid, x, y)
        while not neighbors:
            x, y = random.choice(region)
            neighbors = get_neighbors(grid, x, y)
        nx, ny = random.choice(neighbors)
        grid[nx][ny] = value
        region.append((nx, ny))

    return region

def get_distance(region1, region2):
    """Calculates the minimum distance between two regions"""
    return min(math.dist((x1, y1), (x2, y2)) for x1, y1 in region1 for x2, y2 in region2)

def is_valid_placement(grid, candidate_cells, region1, min_distance, max_distance):
    """Checks if a candidate region maintains the required min and max distance"""
    dist = get_distance(region1, candidate_cells)
    return min_distance <= dist <= max_distance

def place_region_with_distance(grid, region_size, value, region1, min_distance, max_distance):
    """Places a connected region ensuring the required distance from the first region"""
    rows, cols = len(grid), len(grid[0])
    attempts = 0
    while attempts < 100:
        candidate_cells = place_region([row[:] for row in grid], region_size, value)
        if is_valid_placement(grid, candidate_cells, region1, min_distance, max_distance):
            for x, y in candidate_cells:
                grid[x][y] = value
            return 1
        attempts += 1
    return 0

def create_grid_overlay(size, region_size, min_distance, max_distance):
    rows = cols = size
    """Creates the grid and places two regions of 1s and 2s"""
    grid = [[0] * cols for _ in range(rows)]
    
    # Place the first region of 1s
    region1 = place_region(grid, region_size, 1)
    
    # Place the second region of 2s ensuring min and max distance
    success_code = place_region_with_distance(grid, region_size, 2, region1, min_distance, max_distance)
    
    while success_code == 0:
        grid = [[0] * cols for _ in range(rows)]
        region1 = place_region(grid, region_size, 1)
        success_code = place_region_with_distance(grid, region_size, 2, region1, min_distance, max_distance)
    return grid

def create_grid_overlay_one_region(size, region_size, value):
    rows = cols = size
    """Creates the grid and places a region of a given value"""
    grid = [[0] * cols for _ in range(rows)]
    
    # Place the first region of 1s
    place_region(grid, region_size, value)
    
    return grid