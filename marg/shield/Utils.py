from .datastructures.GameGraphs import TwoPlayerGameGraph


def normalize_distribution(distribution):
        total = sum(distribution.values())
        for action in distribution:
            distribution[action] /= total
        return distribution


def grid_to_game_graph(grid, robot_position):
    """
    def __init__(self, vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state):
        self.initial_state = initial_state
        self.vertices = vertices
        self.priorities = priorities
        self.ownership = ownership
        self.outgoing_edges = outgoing_edges
        self.incoming_edges = incoming_edges
        """
    
    grid_size = len(grid)
    vertices = set()
    priorities = {}
    ownership = {}
    outgoing_edges = {}
    incoming_edges = {}
    edge_labels = {}
    vertex_labels = {}
    for row in range(1, grid_size, 2):
        for column in range(1, grid_size, 2):
            cell_properties = grid[row][column]
            vertices.add((row, column)) # Add all the states to the vertices
            priorities[(row, column)] = get_priorities_from_weights(cell_properties['weight'])
            ownership[(row, column)] = cell_properties['owner']
            outgoing_edges[(row, column)] = []
            incoming_edges[(row, column)] = []
            edge_labels[(row, column)] = {}
            vertex_labels[(row, column)] = cell_properties['weight'][0]

    for row in range(1, grid_size, 2):
        for column in range(1, grid_size, 2):

            # Add the edges

            # if there is no wall above the cell and the cell above is not marked unsafe
            if grid[row - 1][column]['is_wall'] == False:
                outgoing_edges[(row, column)].append(("up", (row - 2, column)))
                incoming_edges[(row - 2, column)].append(("up", (row, column)))
            
            # if there is no wall below the cell and the cell below is not marked unsafe
            if grid[row + 1][column]['is_wall'] == False:
                outgoing_edges[(row, column)].append(("down", (row + 2, column)))
                incoming_edges[(row + 2, column)].append(("down", (row, column)))
            
            # if there is no wall to the left of the cell and the cell to the left is not marked unsafe
            if grid[row][column - 1]['is_wall'] == False:
                outgoing_edges[(row, column)].append(("left", (row, column - 2)))
                incoming_edges[(row, column - 2)].append(("left", (row, column)))

            # if there is no wall to the right of the cell and the cell to the right is not marked unsafe
            if grid[row][column + 1]['is_wall'] == False:
                outgoing_edges[(row, column)].append(("right", (row, column + 2)))
                incoming_edges[(row, column + 2)].append(("right", (row, column)))
    initial_state = robot_position
    return TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels, vertex_labels)

def show_grid(grid):
    for row in grid:
        for cell in row:
            if cell['is_wall']:
                print("O", end = " ")
            elif cell['owner'] == 2:
                print("X", end = " ")
            else:
                print(" ", end = " ")
        print()

def get_priorities_from_weights(weights_list):
    # apply max(1, cell_properties['cell_color']) if cell_properties['cell_color'] != 1 else -1 to every element in weights_list[1:]
    return [max(1, weight) for weight in weights_list[1:]]

def update_game_graph_for_wall(game_graph, grid, wall):
    (wall_row, wall_column) = wall
    if "horizontal" in grid[wall_row][wall_column]['attributes']:
        (state_row, state_column) = (wall_row - 1, wall_column)
        (neighbor_row, neighbor_column) = (wall_row + 1, wall_column)
    else:
        (state_row, state_column) = (wall_row, wall_column - 1)
        (neighbor_row, neighbor_column) = (wall_row, wall_column + 1)

    state = (state_row, state_column)
    neighbor = (neighbor_row, neighbor_column)
    (wall_position_hor, wall_position_ver) = (wall_row - state_row, wall_column - state_column)

    wall_direction = {
        (-1, 0): "up",
        (1, 0): "down",
        (0, -1): "left",
        (0, 1): "right"
    }

    # update the edges between the state and the neighbor
    if grid[wall_row][wall_column]['is_wall'] == True:
        game_graph.outgoing_edges[state] = [(action, next_state) for action, next_state in game_graph.outgoing_edges[state] if next_state != neighbor]
        game_graph.incoming_edges[neighbor] = [(action, prev_state) for action, prev_state in game_graph.incoming_edges[neighbor] if prev_state != state]

        game_graph.outgoing_edges[neighbor] = [(action, next_state) for action, next_state in game_graph.outgoing_edges[neighbor] if next_state != state]
        game_graph.incoming_edges[state] = [(action, prev_state) for action, prev_state in game_graph.incoming_edges[state] if prev_state != neighbor]

    else:
        game_graph.outgoing_edges[state].append((wall_direction[(wall_position_hor, wall_position_ver)], neighbor))
        game_graph.incoming_edges[neighbor].append((wall_direction[(wall_position_hor, wall_position_ver)], state))

        game_graph.outgoing_edges[neighbor].append((wall_direction[(-wall_position_hor, -wall_position_ver)], state))
        game_graph.incoming_edges[state].append((wall_direction[(-wall_position_hor, -wall_position_ver)], neighbor))



def get_unsafe_region(game_graph, marked_unsafe_states):
    unsafe_states = marked_unsafe_states.copy()
    old_unsafe_states = set()
    while unsafe_states != old_unsafe_states:
        old_unsafe_states = unsafe_states.copy()
        for state in old_unsafe_states:
            for _, prev_state in game_graph.incoming_edges[state]:
                if prev_state not in old_unsafe_states:
                    if (game_graph.ownership[prev_state] == 0 and set(game_graph.get_next_states(prev_state)).issubset(old_unsafe_states)) or (game_graph.ownership[prev_state] == 1):
                        unsafe_states.add(prev_state)
    return unsafe_states

def solve_reachability_game(game_graph, target_region, player):
    old_target_region = set()
    current_target_region = set(target_region.copy())
    while current_target_region != old_target_region:
        old_target_region = current_target_region.copy()
        for state in old_target_region:
            for _, prev_state in game_graph.incoming_edges[state]:
                if prev_state not in old_target_region:
                    if player == game_graph.ownership[prev_state]:
                        current_target_region.add(prev_state)
                    else:
                        if set(game_graph.get_next_states(prev_state)).issubset(old_target_region):
                            current_target_region.add(prev_state)
    return current_target_region    

# def restrict_multi_objective_game_graph(game_graph, index):
#     vertices = game_graph.vertices
#     priorities = {state: self.priorities[state] for state in region}
#     ownership = game_graph.ownership
#     outgoing_edges = game_graph.outgoing_edges
#     incoming_edges = game_graph.incoming_edges
#     initial_state = self.initial_state if self.initial_state in region else None
#     edge_labels = {state: self.edge_labels[state] for state in region} if self.edge_labels is not None else None
#     vertex_labels = {state: self.vertex_labels[state] for state in region} if self.vertex_labels is not None else None
#     return TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels, vertex_labels)


def make_graph_adversarial(game_graph):
    vertices = game_graph.vertices
    priorities = game_graph.priorities
    ownership = game_graph.ownership
    outgoing_edges = game_graph.outgoing_edges
    incoming_edges = game_graph.incoming_edges
    initial_state = game_graph.initial_state
    edge_labels = game_graph.edge_labels
    vertex_labels = game_graph.vertex_labels
    new_game_graph = TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, edge_labels, vertex_labels)
    for state in game_graph.vertices:
        if game_graph.ownership[state] == 2:
            new_game_graph.ownership[state] = 1
    return new_game_graph