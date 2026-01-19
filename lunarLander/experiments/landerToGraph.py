"""
This script converts a LunarLander environment into a graph representation.

The graph discretizes the env state space into a grid of cells, where each cell represents a state in the environment. The edges between nodes represent possible actions that can be taken from that state.

Inputs:
- n: Number of grid cells in each dimension.
- seed: Random seed for reproducibility.
- output_file: Path to save the graph representation.


"""


import json
import argparse
import numpy as np
import gymnasium as gym
import math
from matplotlib.path import Path
from typing import Optional, Any
import os
import sys

# Add the submission directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


import CustomEnvs
from marg.shield.datastructures.GameGraphs import TwoPlayerGameGraph


SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 800
SAFE_HEIGHT = 8
FUNNEL_ANGLE = 45
HELIPAD_HEIGHT = VIEWPORT_H / SCALE / 4
LEG_DOWN = 18

# CustomLunarLander-v0 action mapping
ACTION_MAP = {
    'left': 1,         # Fire left orientation engine
    'right': 3,        # Fire right orientation engine
    'top': 2,          # Fire main engine
    'do_nothing': 0    # Do nothing
}
INVERSE_ACTION_MAP = {
    1: 'left',         # Fire left orientation engine
    3: 'right',        # Fire right orientation engine
    2: 'top',          # Fire main engine
    0: 'do_nothing'    # Do nothing
}

# The nodes in the graph are represented as tuples (x, y, ground) showing the position and if the cell intersects the ground.

def isPointInSky(x, y, chunk_x, smooth_y):
    """
    Returns true if a given coordinate does not intersect with the ground
    """
    # Find the terrain segment that contains the x coordinate
    for i in range(len(chunk_x) - 1):
        if chunk_x[i] <= x <= chunk_x[i + 1]:
            # Interpolate the terrain height at x
            t = (x - chunk_x[i]) / (chunk_x[i + 1] - chunk_x[i])
            terrain_y = smooth_y[i] * (1 - t) + smooth_y[i + 1] * t
            # Check if the point is above the terrain
            return y > terrain_y
    # If x is outside the terrain bounds, assume it's in the sky
    return True

def createGrid(n):
    """
    Creates a grid of size n x n
    """
    x_min, x_max = 0, VIEWPORT_W / SCALE
    y_min, y_max = 0, VIEWPORT_H / SCALE

    # Create a grid of n x n cells in the specified range
    x_grid = np.linspace(x_min, x_max, n + 1)
    y_grid = np.linspace(y_min, y_max, n + 1)
    return (x_grid, y_grid)


def cellToCorners(i, j, grid):
    """
    Returns the coordinates of the corners of a grid cell
    """
    x_grid = grid[0]
    y_grid = grid[1]
    # Get the corners of the cell at (i, j)
    x0, x1 = x_grid[i], x_grid[i + 1]
    y0, y1 = y_grid[j], y_grid[j + 1]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

def doesCellIntersectGround(cell, grid, chunk_x, smooth_y):
    """
    Return true if an corner of the cell intersects with the ground
    """

    # Check if any point in the cell intersects the ground
    for (x, y) in cellToCorners(cell[0], cell[1], grid):
        if not isPointInSky(x, y, chunk_x, smooth_y):
            return True
    return False

def getGroundCells(n, grid, chunk_x, smooth_y):
    """
    Returns a set of all the cells of the grid 
    intersecting with the ground.

    Also, returns the set of cells that within
    SAFE_HEIGHT above the ground cells
    """
    ground_cells = set()
    ground_safety_cells = set()
    cell_height = grid[1][1] - grid[1][0]
    numCellAboveGround = int(SAFE_HEIGHT / cell_height)
    
    for i in range(n):
        for j in range(n):
            cell = (i, j)
            if doesCellIntersectGround(cell, grid, chunk_x, smooth_y):
                ground_cells.add(cell)
                # Add safety cells above the ground cell
                for k in range(0, numCellAboveGround + 1):
                    safety_cell = (i, j + k)
                    if safety_cell[1] < n:
                        ground_safety_cells.add(safety_cell)

    return ground_cells, ground_safety_cells

def allCellsBetween(x1, x2, y1, y2, grid):
    """
    Returns all the cells which are contained by the rectangle created by x_1, x_2, y_1, and y_2
    """
    x_grid = grid[0]
    y_grid = grid[1]
    cells = set()
    for i in range(len(x_grid) - 1):
        for j in range(len(y_grid) - 1):
            if (x1 <= x_grid[i] and x_grid[i + 1] <= x2) and \
               (y1 <= y_grid[j] and y_grid[j + 1] <= y2):
                # If the cell is completely within the bounds, add it
                cells.add((i, j))
    return cells


def allCellsInTrapezoid(x1, x2, y1, y2, theta_deg, grid):
    """
    Returns all the cells which are contained in the trapezoidal region defined by
    the bottom edge ((x1, y1), (x2, y1)), the top edge (expanded by theta_deg from (x1, y2), (x2, y2)),
    and the grid. The trapezoid is constructed by expanding the top edge outward by
    dx = (y2 - y1) * tan(theta_deg), forming a funnel shape.

    The trapezoid is defined as follows:
    top_left ------------------ top_right
        \                        /
         \                      /
          \                    /
           \------------------/
        bottom_left         bottom_right

    Where:
      - bottom_left = (x1, y1)
      - bottom_right = (x2, y1)
      - top_left = (x1 - dx, y2)
      - top_right = (x2 + dx, y2)
    """
    x_grid = grid[0]
    y_grid = grid[1]
    cells = set()

    # Convert angle to radians
    theta = math.radians(theta_deg)

    # Compute top trapezoid corners using tan(theta)
    dx = (y2 - y1) * math.tan(theta)
    top_left = (x1 - dx, y2)
    top_right = (x2 + dx, y2)
    bottom_left = (x1, y1)
    bottom_right = (x2, y1)

    # Trapezoid vertices in clockwise order
    trapezoid = [bottom_left, bottom_right, top_right, top_left, bottom_left]

    # Create a Path object for point-in-polygon testing
    trapezoid_path = Path(trapezoid)

    # Loop over all cells in the grid
    for i in range(len(x_grid) - 1):
        for j in range(len(y_grid) - 1):
            # Compute cell center
            xc = (x_grid[i] + x_grid[i + 1]) / 2
            yc = (y_grid[j] + y_grid[j + 1]) / 2

            # Check if cell center lies within the trapezoid
            if trapezoid_path.contains_point((xc, yc)):
                cells.add((i, j))

    return cells

def allCellsInFunnel(x1, x2, y1, y2, theta_deg, bottom_height, grid):
    """
    Returns all the cells which are contained in the funnel shape defined by
    the bottom edge ((x1, y1), (x2, y1)), the top edge (expanded by theta_deg from (x1, y2), (x2, y2)),
    and the grid. The funnel is constructed by expanding the top edge outward by
    dx = (y2 - y1) * tan(theta_deg), forming a funnel shape

    top_left ------------------ top_right
        \                        /
         \                      /
          \                    /
           \                  /
           |                  |
           |                  |
           |------------------|
        bottom_left         bottom_right
    """
    x1 = x1 + 0.5
    x2 = x2 - 0.5
    cells_above_x1_x2 = allCellsBetween(x1, x2, y1, y2, grid)
    cells_in_trapezoid = allCellsInTrapezoid(x1, x2, y1 + bottom_height, y2 + 2, theta_deg, grid)
    cells = cells_above_x1_x2.union(cells_in_trapezoid)

    return cells



class TerrainGraphOutput:
    """
    A class to hold the output elements of the terrain graph.
    This class allows for easy addition and retrieval of elements, and can convert the output to a dictionary format.
    """
    def __init__(self, outputElements: Optional[dict[str, Any]] = None):
        self.outputElements = outputElements if outputElements is not None else {}
    
    def to_dict(self):
        """
        Convert the output elements to a dictionary.
        
        Returns:
            dict: A dictionary representation of the output elements.
        """
        return self.outputElements
    
    def add_element(self, key: str, value: Any):
        """
        Add an element to the output dictionary.
        
        Args:
            key (str): The key for the element.
            value (Any): The value of the element.
        """
        self.outputElements[key] = value

    def get_element(self, key: str) -> Any:
        """
        Get an element from the output dictionary.
        
        Args:
            key (str): The key for the element.
        
        Returns:
            Any: The value of the element, or None if the key does not exist.
        """
        return self.outputElements.get(key, None)



def buildGraphWithTerrain(n, terrainInfo, logger=None):
    """ Builds a graph representation of the LunarLander environment using the provided terrain information.
    Args:
        n (int): The number of grid cells in each dimension.
        terrainInfo (dict): A dictionary containing terrain information with keys:
            - chunk_x: List of x coordinates of the terrain chunks.
            - smooth_y: List of y coordinates of the terrain chunks.
            - helipad_x1: The x coordinate of the left edge of the helipad.
            - helipad_x2: The x coordinate of the right edge of the helipad.
            - helipad_y: The y coordinate of the helipad.
            - initial_x: The initial x coordinate of the lander.
            - initial_y: The initial y coordinate of the lander.
        logger (Optional[Logger]): An optional logger for debugging information.
    Returns:
        TerrainGraphOutput: An object containing the graph representation, grid, ground cells, and other terrain information.
    """
    # terrainInfo is a dictionary with keys: chunk_x, smooth_y, helipad_x1, helipad_x2, helipad_y, initial_x, initial_y
    chunk_x = terrainInfo["chunk_x"]
    smooth_y = terrainInfo["smooth_y"]
    helipad_x1 = terrainInfo["helipad_x1"]
    helipad_x2 = terrainInfo["helipad_x2"]
    helipad_y = terrainInfo["helipad_y"]
    initial_x = terrainInfo["initial_x"]
    initial_y = terrainInfo["initial_y"]

    # Create the grid
    grid = createGrid(n)

    cell_height = grid[1][1] - grid[1][0]
    helipad_cells = allCellsBetween(helipad_x1, helipad_x2, helipad_y, helipad_y + 2*cell_height, grid)
    helipad_column = allCellsBetween(helipad_x1 + 1.5, helipad_x2 - 1.5, helipad_y, VIEWPORT_H / SCALE, grid) # A column of cells above the middle of the helipad

    # Initialize the graph
    graph = {}
    ground_cells, ground_safety_cells = getGroundCells(n, grid, chunk_x, smooth_y)
    initial_cell = map_state_to_discrete([initial_x, initial_y, 0, 0, 0, 0, 0, 0], grid)
    if logger:
        logger.info(f"Initial coordinates: {initial_x, initial_y}")
    cellsAboveHelipad = allCellsInFunnel(helipad_x1, helipad_x2, helipad_y, VIEWPORT_H, FUNNEL_ANGLE, 5.5, grid)
    if logger:
        logger.debug(f"Cells above helipad: {sorted(cellsAboveHelipad)}")


    # actual ground safety cells are the ones that are in the ground safety cells but not above the helipad
    actual_ground_safety_cells = ground_safety_cells - cellsAboveHelipad

    # Trim the top of the safety cells and make it flat, so it is easier for lander to try to land on the helipad; will not get stuck in a valley.
    cells_limit = allCellsBetween(0, VIEWPORT_W, helipad_y-4, helipad_y + 6.0, grid)
    actual_ground_safety_cells = actual_ground_safety_cells.intersection(cells_limit) 


    recovery_ground_safety_cells = ground_safety_cells - allCellsInFunnel(helipad_x1 - 1, helipad_x2 + 1, helipad_y, VIEWPORT_H, FUNNEL_ANGLE, 3.5, grid)
    recovery_cells_limit = allCellsBetween(0, VIEWPORT_W, helipad_y - 4, helipad_y + 4.0, grid)
    recovery_ground_safety_cells = recovery_ground_safety_cells.intersection(recovery_cells_limit)

    if logger:
        logger.info(f"Ground cells: {len(ground_cells)}, Ground safety cells: {len(ground_safety_cells)}, Actual Ground safety cells: {len(actual_ground_safety_cells)}, Helipad cells: {len(helipad_cells)}")

    # Iterate over each cell in the grid
    for i in range(n):
        for j in range(n):
            cell = (i, j)
            is_ground = cell in ground_cells
            node = (i, j, is_ground)
            edges = set()
            # Add edges based on possible actions
            if i > 0:
                next_node = (i - 1, j, (i - 1, j) in ground_cells)
                edges.add((next_node, 'left'))
            if i < n - 1:
                next_node = (i + 1, j, (i + 1, j) in ground_cells)
                edges.add((next_node, 'right'))
            if j < n - 1:
                next_node = (i, j + 1, (i, j + 1) in ground_cells)
                edges.add((next_node, 'top'))
            if j > 0:
                next_node = (i, j - 1, (i, j - 1) in ground_cells)
                edges.add((next_node, 'do_nothing'))
            graph[node] = edges
            

    extreme_cells = set()
    for i in range(0, n):
        extreme_cells.add((0, i))
        extreme_cells.add((n-1, i))

    if logger:
        logger.debug(f"Raw graph built with ground cells: {len(ground_cells)},actual_ground_safety_cells: {len(actual_ground_safety_cells)}, helipad_cells: {len(helipad_cells)}, initial_cell: {initial_cell}")
        logger.debug(f"Chunk X: {chunk_x}, Smooth Y: {smooth_y}")


    terrainGraphOutput = TerrainGraphOutput()
    terrainGraphOutput.add_element("graph", graph)
    terrainGraphOutput.add_element("grid", grid)
    terrainGraphOutput.add_element("ground_cells", ground_cells)
    terrainGraphOutput.add_element("chunk_x", chunk_x)
    terrainGraphOutput.add_element("smooth_y", smooth_y)
    terrainGraphOutput.add_element("actual_ground_safety_cells", actual_ground_safety_cells)
    terrainGraphOutput.add_element("recovery_ground_safety_cells", recovery_ground_safety_cells)
    terrainGraphOutput.add_element("extreme_cells", extreme_cells)
    terrainGraphOutput.add_element("helipad_cells", helipad_cells)
    terrainGraphOutput.add_element("helipad_column", helipad_column)
    terrainGraphOutput.add_element("initial_cell", initial_cell)

    return terrainGraphOutput


def buildGraph(n, seed):
    """ 
    For external use:
    Builds a graph representation of the LunarLander environment using the provided number of grid cells and seed.
    """

    env = gym.make('CustomLunarLander-v0', continuous=False)
    env.reset(seed=seed)

    # Extract terrain information
    terrain_info = None
    if hasattr(env.unwrapped, "terrain_info"):
        terrain_info = env.unwrapped.terrain_info
    else:
        raise ValueError("The environment does not have terrain data. Ensure you are using CustomLunarLander-v0.")
    
    return buildGraphWithTerrain(n, terrain_info)


def graphToDataStructure(terrainGraphOutput: TerrainGraphOutput, logger=None):
    """ 
    Converts the terrain graph output to a TwoPlayerGameGraph data structure.
    """
    graph = terrainGraphOutput.get_element("graph")
    ground_cells = terrainGraphOutput.get_element("ground_cells")
    recovery_ground_safety_cells = terrainGraphOutput.get_element("recovery_ground_safety_cells")
    helipad_cells = terrainGraphOutput.get_element("helipad_cells")
    initial_cell = terrainGraphOutput.get_element("initial_cell")
    helipad_column = terrainGraphOutput.get_element("helipad_column")
    extreme_cells = terrainGraphOutput.get_element("extreme_cells")

    vertices = set()
    priorities = {}
    priorities_safe_buechi = {}
    ownership = {}
    outgoing_edges = {(cell[0], cell[1]): [] for cell in graph.keys()}  # Initialize outgoing edges for each cell
    incoming_edges = {(cell[0], cell[1]): [] for cell in graph.keys()}  # Initialize incoming edges for each cell
    edge_labels = {}
    vertex_labels = {}

    if logger:
        logger.info(f"Helipad cells: {helipad_cells}")

    for cell, edges in graph.items():
        node = (cell[0], cell[1])  # Ignore the ground part for vertices
        if logger:
            logger.debug(f"Processing cell {node}")
        vertices.add(node)  # Add only (i, j) part of the node ignoring the fact if it is ground or not

        priorities[node] = [1] # Default priority for all cells is 1

        if node in helipad_column:
            priorities[node] = [2] # Makes the cell a Buechi cell, which should be visited infinitely often
        if node in extreme_cells:
            priorities[node] = [-1] # Makes the cell unsafe
        
        
        ### ONLY SAFETY SHIELD ###
        if node in ground_cells or node in recovery_ground_safety_cells or node in extreme_cells:
            priorities_safe_buechi[node] = [-1] # Makes the cell unsafe
        else:
            priorities_safe_buechi[node] = [1] # Default priority for all other cells is 1


        ownership[node] = 0 #Player 0 owns all cells

        edge_labels[node] = {}
        vertex_labels[node] = {}

        for next_node, action in edges:
            outgoing_edges[node].append((action, (next_node[:2])))  # Store only (i, j) part of the next node
            incoming_edges[next_node[:2]].append((action, node))  # Store only (i, j) part of the next node
    
    initial_state = initial_cell[:2]  # Use only (i, j) part of the initial cell
    
    # Return two graphs: one for STARs and one for safety shield 
    return TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels=edge_labels, vertex_labels=vertex_labels), TwoPlayerGameGraph(vertices, priorities_safe_buechi, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels=edge_labels, vertex_labels=vertex_labels)



def map_state_to_discrete(state, grid):
    """ 
    Maps a continuous state from the LunarLander environment to a discrete grid cell.
    """
    # state: [x, y, vx, vy, angle, vangle, left_leg, right_leg]
    x, y = state[0], state[1]

    # x is in the range -1 to 1, we first need to scale it to 0 and VIEWPORT_W/SCALE 
    x = (x + 1) * (VIEWPORT_W / (2.0 * SCALE))

    # Scale y to Helipad height and VIEWPORT_H/SCALE, better solution than that for x
    y = y * (VIEWPORT_H / (2.0 * SCALE)) + (HELIPAD_HEIGHT + LEG_DOWN / SCALE)

    # Find i, j efficiently
    i = np.searchsorted(grid[0], x, side='right') - 1
    j = np.searchsorted(grid[1], y, side='right') - 1
    i = np.clip(i, 0, len(grid[0]) - 2)
    j = np.clip(j, 0, len(grid[1]) - 2)

    return (int(i), int(j))

def landedOutsideHelipad(obs, helipad_x1, helipad_x2, logger=None):
    """ Checks if the lander has landed outside the helipad."""
    # obs: [x, y, vx, vy, angle, vangle, left_leg, right_leg]
    x = obs[0]
    landed = obs[6] == 1 and obs[7] == 1  # Check if any leg is touching the ground

    # x and y are in the range -2.5 to 2.5, we first need to scale them to 0 and VIEWPORT_W/SCALE and VIEWPORT_H/SCALE
    x = (x + 1) * (VIEWPORT_W / (2.0 * SCALE))

    # check if the x coordinate is within the helipad bounds
    if landed:
        if helipad_x1 - 0.5 <= x <= helipad_x2 + 0.5:
            if logger:
                logger.info("Lander landed on the helipad.")
            return False
        else:
            if logger:
                logger.info("Lander landed outside the helipad.")
            return True
    else:
        if logger:
            logger.info(f"Lander is not landed, current position: ({x}, {obs[1]})")
        return False

def landedOnHelipad(obs, helipad_x1, helipad_x2):
    """ Checks if the lander has landed on the helipad."""
    # obs: [x, y, vx, vy, angle, vangle, left_leg, right_leg]
    x = obs[0]
    landed = obs[6] == 1 and obs[7] == 1  # Check if any leg is touching the ground

    # x and y are in the range -2.5 to 2.5, we first need to scale them to 0 and VIEWPORT_W/SCALE and VIEWPORT_H/SCALE
    x = (x + 1) * (VIEWPORT_W / (2.0 * SCALE))

    # check if the x coordinate is within the helipad bounds
    if landed:
        if helipad_x1 <= x <= helipad_x2:
            return True
        else:
            return False
    else:
        return False


class VisualElement:
    """
    A class to represent a visual element in the graph visualization.
    Each element consists of a set of cells, a color, and an alpha value for transparency.
    """
    def __init__(self, cells, color, alpha=0.5):
        self.cells = cells
        self.color = color
        self.alpha = alpha

class VisualElements:
    """
    A class to hold multiple visual elements for graph visualization.
    """
    def __init__(self):
        self.elements = []

    def add(self, cells, color, alpha=0.5):
        self.elements.append(VisualElement(cells, color, alpha))

    def get_elements(self):
        return self.elements

def visualizeGraph(graph, grid, visualElements, file_location="lunar_lander_graph.png"):
    """
    Visualizes the grid using matplotlib and saves it to a file.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, ax = plt.subplots()
    x_grid = grid[0]
    y_grid = grid[1]

    for element in visualElements.get_elements():
        for (i, j) in element.cells:
            cell = cellToCorners(i, j, grid)
            polygon = plt.Polygon(cell, color=element.color, alpha=element.alpha)
            ax.add_patch(polygon)

    ax.set_xlim(0, VIEWPORT_W / SCALE)
    ax.set_ylim(0, VIEWPORT_H / SCALE)
    ax.set_aspect('equal')
    
    # Hide axes, ticks, labels
    ax.axis('off')

    # Save tightly cropped image
    plt.savefig(file_location, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Convert LunarLander environment to graph representation.")
    parser.add_argument("--n", type=int, default=10, help="Number of grid cells in each dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_file", type=str, default="lunar_lander_graph.json", help="File to save the graph representation.")

    args = parser.parse_args()

    # graph, grid, ground_cells, chunk_x, smooth_y, ground_safety_cells, recovery_ground_safety_cells, helipad_cells, initial_cell = buildGraph(args.n, args.seed)
    terrainGraphOutput = buildGraph(args.n, args.seed)  
    graph = terrainGraphOutput.get_element("graph")
    grid = terrainGraphOutput.get_element("grid")
    ground_cells = terrainGraphOutput.get_element("ground_cells")
    chunk_x = terrainGraphOutput.get_element("chunk_x")
    smooth_y = terrainGraphOutput.get_element("smooth_y")
    ground_safety_cells = terrainGraphOutput.get_element("ground_safety_cells")
    recovery_ground_safety_cells = terrainGraphOutput.get_element("recovery_ground_safety_cells")
    helipad_cells = terrainGraphOutput.get_element("helipad_cells")
    initial_cell = terrainGraphOutput.get_element("initial_cell")

    # Convert the graph to a TwoPlayerGameGraph data structure
    game_graph = graphToDataStructure(graph, ground_safety_cells, recovery_ground_safety_cells, ground_cells, helipad_cells, initial_cell)

    # Save the graph to a file
    with open(args.output_file, 'w') as f:
        json.dump({
            "graph": {str(k): [str(e) for e in v] for k, v in graph.items()},
            "grid": (grid[0].tolist(), grid[1].tolist()),
            "ground_cells": list(ground_cells),
            "chunk_x": chunk_x,
            "smooth_y": smooth_y
        }, f)

    print(f"Graph saved to {args.output_file}")

    # Optionally visualize the graph
    visualElements = VisualElements()
    visualElements.add(ground_cells, 'green', alpha=0.5)
    visualElements.add(ground_safety_cells, 'green', alpha=0.1)
    visualElements.add(recovery_ground_safety_cells, 'green', alpha=0.2)
    visualElements.add(helipad_cells, 'blue', alpha=0.5)

    visualizeGraph(graph, grid, visualElements)

if __name__ == "__main__":
    main()