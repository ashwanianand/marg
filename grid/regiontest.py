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
        if not neighbors:
            return place_region(grid, region_size, value)  # Restart if stuck
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
    
    while True:
        candidate_cells = place_region([row[:] for row in grid], region_size, value)
        if is_valid_placement(grid, candidate_cells, region1, min_distance, max_distance):
            for x, y in candidate_cells:
                grid[x][y] = value
            return

def create_grid_overlay(size, region_size, min_distance, max_distance):
    rows = cols = size
    """Creates the grid and places two regions of 1s and 2s"""
    grid = [[0] * cols for _ in range(rows)]
    
    # Place the first region of 1s
    region1 = place_region(grid, region_size, 1)
    
    # Place the second region of 2s ensuring min and max distance
    place_region_with_distance(grid, region_size, 2, region1, min_distance, max_distance)
    
    return grid

# Example usage
size = 100
region_size = 10
min_distance = 2
max_distance = 2

grid = create_grid_overlay(size, region_size, min_distance, max_distance)

# Print grid
for row in grid:
    print(" ".join(map(str, row)))
