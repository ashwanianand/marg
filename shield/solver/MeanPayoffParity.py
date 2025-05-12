from ..datastructures.GameGraphs import TwoPlayerGameGraph
from .Parity import compute_template_in_two_player_graph
from .MeanPayoff import compute_value_function as compute_value_function_mean_payoff
from ..Utils import solve_reachability_game as solve_reachability_game


"""
The implementation is based on the algorithm presented in the paper "Mean-Payoff Parity Games" by Chatterjee, Henzinger, and Jurdzinski.

The differences are noted below:
- W_1 -> W_0 and W_2 -> W_1
- The parity condition is reversed, i.e. Player 0 wins if the play has max (as opposed to min, in the paper) of infinitely often priority is even.
"""

    

def compute_value_function(game_graph):
    value_function = {state: 0 for state in game_graph.vertices}
    winning_region, _, _, _ = compute_template_in_two_player_graph(game_graph)

    for state in game_graph.vertices - winning_region: 
        value_function[state] = -float('inf')
    
    W_0 = set()
    subgame = game_graph.subgame(winning_region)
    vertices = subgame.vertices
    while vertices != set():
        subgame_winning_region, _, _, _ = compute_template_in_two_player_graph(subgame)
        # print(subgame_winning_region, subgame.vertices)
        if subgame_winning_region != subgame.vertices:
            g = max(value_function[state] for state in W_0 if exists_edge_from_set_to_state(subgame.get_player_vertices(0).intersection(subgame.vertices - subgame_winning_region), state, subgame.incoming_edges))
            T_1 = {state for state in get_least_value_class(subgame).intersection(subgame.get_player_vertices(0)) if exists_edge_from_state_to_set(state, {w for w in W_0 if value_function[w] == g}, game_graph.outgoing_edges) }
            universal_reach_set = get_universal_reach_set(game_graph, T_1)
            W_0 = W_0.union(universal_reach_set)
            for state in universal_reach_set:
                value_function[state] = g
        else:
            if vertices.intersection(game_graph.get_vertices_with_priority(game_graph.get_maximum_priority())) != set():
                L, l = get_least_value_class(subgame)
                for state in L: # line 6 in Alg 1
                    value_function[state] = l
                skip = set_values(game_graph, value_function, W_0, L, l)
                if not skip:
                    W_0 = W_0.union(L)
                    for state in L:
                        value_function[state] = l
            else:
                G, g = get_greatest_value_class(subgame)
                for state in G: # line 5 in Alg 2
                    value_function[state] = g
                skip = set_values(game_graph, value_function, W_0, G, g)
                if not skip:
                    W_0 = W_0.union(G)
                    for state in G:
                        value_function[state] = g
        subgame = game_graph.subgame(vertices)
        vertices = vertices - W_0
    return value_function


def set_values(game_graph, value_function, W_0, J, j):
    if W_0 == set():
        return False
    g = max(value_function[state] for state in W_0 if exists_edge_from_set_to_state(J.intersection(game_graph.get_player_vertices(0)), state, game_graph.incoming_edges))
    if g > j:
        T_1 = {state for state in J.intersection(game_graph.get_player_vertices(0)) if exists_edge_from_state_to_set(state, {w for w in W_0 if value_function[w] == g}, game_graph.outgoing_edges)} 
        universal_reach_set = get_universal_reach_set(game_graph, T_1)
        W_0 = W_0.union(universal_reach_set)
        for state in universal_reach_set:
            value_function[state] = g
        return True
    l = min(value_function[state] for state in W_0 if exists_edge_from_set_to_state(J.intersection(game_graph.get_player_vertices(1)), state, game_graph.incoming_edges) and value_function[state] != -float('inf'))

    if l < j:
        T_2 = {state for state in J.intersection(game_graph.get_player_vertices(1)) if exists_edge_from_state_to_set(state, {w for w in W_0 if value_function[w] == l}, game_graph.outgoing_edges)} 
        universal_reach_set = get_universal_reach_set(game_graph, T_2)
        W_0 = W_0.union(universal_reach_set)
        for state in universal_reach_set:
            value_function[state] = l
        return True



def get_universal_reach_set(game_graph, vertices):
    old_vertices = set()
    while vertices != old_vertices:
        old_vertices = vertices.copy()
        for state in old_vertices:
            for _, prev_state in game_graph.incoming_edges[state]:
                if set(game_graph.get_next_states(prev_state)).issubset(old_vertices):
                    vertices.add(prev_state)
    return vertices

def get_least_value_class(game_graph):
    winning_region, _, _, _ = compute_template_in_two_player_graph(game_graph)
    if winning_region != game_graph.vertices:
        raise AssertionError("Winning region does not match the game graph vertices when computing the least value class")
    if game_graph.get_maximum_priority() % 2 != 0:
        raise AssertionError("The maximum priority is not even when computing the least value class")
    F = solve_reachability_game(game_graph, game_graph.get_vertices_with_priority(game_graph.get_maximum_priority()), 0)
    H = game_graph.vertices - F
    G_tilda = None
    if H != set():
        calH = game_graph.subgame(H)
        subgraph_value_function = compute_value_function(calH)
        G_tilda = construct_mean_payoff_game_graph_from_section_3(game_graph, H, subgraph_value_function) 
    else:
        G_tilda = game_graph
    # print("g_tilda is ", G_tilda)
    value_function_G_tilda = compute_value_function_mean_payoff(G_tilda)
    # print("value function is ", value_function_G_tilda)
    l = min(value_function_G_tilda[state] for state in G_tilda.vertices if value_function_G_tilda[state] != -float('inf'))
    G = {state for state in G_tilda.vertices if value_function_G_tilda[state] == l}
    LV = G.intersection(game_graph.vertices)
    return LV, l




def get_greatest_value_class(game_graph):
    winning_region, _, _, _ = compute_template_in_two_player_graph(game_graph)
    if winning_region != game_graph.vertices:
        raise AssertionError("Winning region does not match the game graph vertices when computing the greatest value class")
    if game_graph.get_maximum_priority() % 2 == 0:
        raise AssertionError("The maximum priority is not odd when computing the greatest value class")
    F = solve_reachability_game(game_graph, game_graph.get_vertices_with_priority(game_graph.get_maximum_priority()), 1)
    H = game_graph.vertices - F
    calH = game_graph.subgame(H)
    subgraph_value_function = compute_value_function(calH)
    g = max(subgraph_value_function[state] for state in calH.vertices)
    G = {state for state in calH.vertices if subgraph_value_function[state] == g}
    return G, g

def exists_edge_from_state_to_set(state, vertices, outgoing_edges):
    for _, next_state in outgoing_edges[state]:
        if next_state in vertices:
            return True
    return False

def exists_edge_from_set_to_state(vertices, state, incoming_edges):
    for _, prev_state in incoming_edges[state]:
        if prev_state in vertices:
            return True
    return False

def construct_mean_payoff_game_graph_from_section_3(mean_payoff_parity_game_graph, additional_vertices, value_function):
    vertices = mean_payoff_parity_game_graph.vertices.copy()
    priorities = mean_payoff_parity_game_graph.priorities.copy()
    ownership = mean_payoff_parity_game_graph.ownership.copy()
    outgoing_edges = mean_payoff_parity_game_graph.outgoing_edges.copy()
    incoming_edges = mean_payoff_parity_game_graph.incoming_edges.copy()
    initial_state = mean_payoff_parity_game_graph.initial_state
    vertex_labels = mean_payoff_parity_game_graph.vertex_labels.copy() if mean_payoff_parity_game_graph.vertex_labels is not None else {}
    edge_labels = mean_payoff_parity_game_graph.edge_labels.copy() if mean_payoff_parity_game_graph.edge_labels is not None else None

    new_vertices = {(state[0], state[1], 1) for state in additional_vertices}
    vertices.update(new_vertices)
    
    for state in new_vertices:
        original_state = (state[0], state[1])
        priorities[state] = 0
        ownership[state] = 0
        outgoing_edges[state] = []
        incoming_edges[state] = []
        vertex_labels[state] = value_function[state]
        # add edges
        outgoing_edges[outgoing_edges].append(("to_copy", state))
        incoming_edges[state].append(("to_copy", original_state))
        outgoing_edges[state].append(("stay", state))
        incoming_edges[state].append(("stay", state))

    return TwoPlayerGameGraph(vertices, priorities, ownership, outgoing_edges, incoming_edges, initial_state, vertex_labels, edge_labels)

