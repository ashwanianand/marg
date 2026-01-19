from ..datastructures.GameGraphs import TwoPlayerGameGraph
from math import ceil, floor
import random
from tqdm import tqdm

"""
The value computation function is based on the algorithm described in the paper "The complexity of mean payoff games on graphs" by Zwick and Paterson.
"""


def compute_value_function(game_graph, edge_based=False):
    value_function = {state: game_graph.vertex_labels[state] for state in game_graph.vertices}
    strategy = {state: None for state in game_graph.get_player_vertices(0)}
    old_value_function = {state: 0 for state in game_graph.vertices}
    iterations = 0
    max_iterations = 4 * (len(game_graph.vertices)**3) * game_graph.get_maximum_vertex_label()
    while value_function != old_value_function and iterations < max_iterations:
        iterations += 1
        old_value_function = value_function.copy()
        for state in game_graph.vertices:
            if edge_based:
                update_value_function_edge_based(game_graph, strategy, old_value_function, value_function, state)
            else:
                update_value_function_vertex_based(game_graph, strategy, old_value_function, value_function, state)
    normalize_value_function(value_function, game_graph, iterations)
    return get_actual_value_function(game_graph, value_function), strategy

def normalize_value_function(value_function, game_graph, iterations):
    for state in game_graph.vertices:
        value_function[state] = value_function[state] / iterations  
    return value_function


def update_value_function_vertex_based(game_graph, strategy, old_value_function, value_function, state):
    neighbor_values = {(action, next_state): old_value_function[next_state] + game_graph.vertex_labels[next_state] for action, next_state in game_graph.outgoing_edges[state]}
    if game_graph.ownership[state] == 0:
        value_function[state] = max(neighbor_values.values()) if neighbor_values else 0
        if old_value_function[state] != value_function[state]:
            strategy[state] = random.choice([next_state for next_state, value in neighbor_values.items() if value == value_function[state]])
    else:
        value_function[state] = min(neighbor_values.values()) if neighbor_values else 0

def update_value_function_edge_based(game_graph, strategy, old_value_function, value_function, state):
    if game_graph.ownership[state] == 0:
        value_function[state] = max([old_value_function[next_state] + game_graph.edge_labels[state][next_state] for _, next_state in game_graph.outgoing_edges[state]])
    else:
        value_function[state] = min([old_value_function[next_state] + game_graph.edge_labels[state][next_state] for _, next_state in game_graph.outgoing_edges[state]])

def get_actual_value_function(game_graph, approx_value_function):
    n = len(game_graph.vertices)
    ball_radius = 1/(2 * n * (n-1))
    value_function = {state: 0 for state in game_graph.vertices}
    for state in game_graph.vertices:
        # if there is an integer between the value of the state minus the ball radius and the value of the state plus the ball radius
        # then the value of the state is the integer
        # otherwise, find the rational number p/q in the ball such that q <= n
        value = approx_value_function[state]
        lower_limit = value - ball_radius
        upper_limit = value + ball_radius
        # if lower_limit < int(value) < upper_limit:
        #     value_function[state] = int(value)
        # else:
        value = find_rational_between(lower_limit, upper_limit, n)
        if value is not None:
            value_function[state] = value
        else:
            raise ValueError("No rational number found in the ball")
    return value_function
            
def find_rational_between(A, B, N):
    for q in range(1, N + 1):
        p_min = ceil(A * q)
        p_max = floor(B * q)
        if p_min <= p_max:
            return p_min/q  # Return the first valid fraction
    return None  # No solution found

def get_winning_region_with_threshold(game_graph, value_function, threshold):
    return {state for state in game_graph.vertices if value_function[state] >= threshold}


