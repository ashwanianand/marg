from solver.Parity import compute_template_in_two_player_graph
from ..datastructures.GameGraphs import TwoPlayerGameGraph


def get_augmented_graph(game_graph, index):
    """
    Based on the reduction from the paper "Simple Stochastic Parity Games" by Chatterjee, Jurdzinski, and Henzinger.
    """
    
    # create a new graph with the same vertices as the original graph
    augmented_graph = TwoPlayerGameGraph(game_graph.vertices, game_graph.priorities, game_graph.ownership, game_graph.outgoing_edges, game_graph.incoming_edges, game_graph.initial_state)
    
    # add a new vertex for each vertex with owner 2
    for state in game_graph.get_player_vertices(2):
        for priority in range(0, game_graph.ownership[state] + 2, 2):
            new_state = (state[0], state[1], priority)
            augmented_graph.add_vertex(new_state, [priority], 0)
            augmented_graph.add_edge(state, new_state, "next")

            for next_priority in neighbours_for_gadget(game_graph.priorities[state][index], priority):
                next_state = (state[0], state[1], priority, next_priority)
                if next_priority % 2 == 0:
                    augmented_graph.add_vertex(new_state, [next_state], 1)
                else:
                    augmented_graph.add_vertex(new_state, [next_state], 0)
                augmented_graph.add_edge(new_state, next_state, "next")
                for action, next_next_state in game_graph.outgoing_edges[state]:
                    augmented_graph.add_edge(next_state, next_next_state, action)
        augmented_graph.assign_owner(state, 1)
    return augmented_graph


def neighbours_for_gadget(original_priority, inter_priority):
    next_priorities = []
    if inter_priority == 0:
        next_priorities.append(0)
    else:
        next_priorities.append(inter_priority - 1)
        if inter_priority <= original_priority:
            next_priorities.append(inter_priority)
    return next_priorities

def solve_stochastic_game(game_graph, index):
    augmented_graph = get_augmented_graph(game_graph, index)
    winning_region, safety_template, co_live_template, group_liveness_template  = compute_template_in_two_player_graph(augmented_graph)
    safety_template = safety_template.get_sub_template(game_graph.vertices)
    co_live_template = co_live_template.get_sub_template(game_graph.vertices)
    group_liveness_template = group_liveness_template.get_sub_template(game_graph.vertices)

    return winning_region, safety_template, co_live_template, group_liveness_template