from ..datastructures.GameGraphs import TwoPlayerGameGraph
from ..Utils import solve_reachability_game as solve_reachability_game, get_unsafe_region as get_unsafe_region

def compute_template_in_one_player_graph(game_graph, index):
    # Warning: this function name is misleading. The game graph is actually a two-player game graph, played between player 0 and player 1. 
    safety_template = game_graph.UnsafeEdges({state: set() for state in game_graph.vertices})
    co_live_template = game_graph.ColiveEdges({state: set() for state in game_graph.vertices})
    group_liveness_template = game_graph.LiveGroups([])


    unsafe_states = set([vertex for vertex in game_graph.vertices if game_graph.priorities[vertex][index] == -1])

    for state in game_graph.vertices:
        for action, next_state in game_graph.outgoing_edges[state]:
            if next_state in unsafe_states:
                safety_template.add_unsafe_edge(state, action, next_state)
    
    game_graph_without_unsafe_states = game_graph.subgame(game_graph.vertices - unsafe_states)

    winning_region = compute_template_in_one_player_graph_recursively(game_graph_without_unsafe_states, index, co_live_template, group_liveness_template)


    make_live_groups_complete(game_graph.vertices, group_liveness_template)

    return winning_region, safety_template, co_live_template, group_liveness_template




def compute_template_in_one_player_graph_recursively(game_graph, index, co_live_template, group_liveness_template):
    
    
    if game_graph.vertices == set():
        return game_graph.vertices
    
    max_priority = game_graph.get_maximum_priority(index)
    vertices_with_max_priority = set([vertex for vertex in game_graph.vertices if game_graph.priorities[vertex][index] == max_priority ])


    if max_priority % 2 == 1: # max_priority is odd
        regionA = solve_reachability_game(game_graph, vertices_with_max_priority, 1) 
        regionCompA = game_graph.vertices - regionA
        if regionA == game_graph.vertices:
            return regionCompA
        
        subgame = game_graph.subgame(regionCompA)
        # print("regionA", regionA)

        subgame_winning_region = compute_template_in_one_player_graph_recursively(subgame, index, co_live_template, group_liveness_template)
        # print(subgame_winning_region)

        if subgame_winning_region == set():
            return subgame_winning_region
        else:
            regionB = solve_reachability_game(game_graph, subgame_winning_region, 0)
            co_live_template.add_edges_between_regions_one_player(subgame_winning_region, game_graph.vertices - subgame_winning_region, game_graph.outgoing_edges)

            group_liveness_template.add_live_groups_to_reach_one_player(subgame_winning_region, game_graph.incoming_edges)

            subgameB = game_graph.subgame(game_graph.vertices - regionB)
            return regionB.union(compute_template_in_one_player_graph_recursively(subgameB, index, co_live_template, group_liveness_template))
        

    else: # max_priority is even
        regionA = solve_reachability_game(game_graph, vertices_with_max_priority, 0)
        regionCompA = game_graph.vertices - regionA
        if regionA == game_graph.vertices:
            group_liveness_template.add_live_groups_to_reach_one_player(vertices_with_max_priority, game_graph.incoming_edges)
            return regionA
        
        subgame = game_graph.subgame(regionCompA)

        subgame_winning_region = compute_template_in_one_player_graph_recursively(subgame, index, co_live_template, group_liveness_template)

        if subgame_winning_region == subgame.vertices:
            group_liveness_template.add_live_groups_to_reach_one_player(vertices_with_max_priority, game_graph.incoming_edges)
            return subgame_winning_region
        else:
            regionB = solve_reachability_game(game_graph, subgame.vertices - subgame_winning_region, 1)
            subgame_without_regionB = game_graph.subgame(game_graph.vertices - regionB)
            return compute_template_in_one_player_graph_recursively(subgame_without_regionB, index, co_live_template, group_liveness_template)



def compute_template_in_two_player_graph(game_graph, index):
    safety_template = game_graph.UnsafeEdges({state: set() for state in game_graph.vertices})
    co_live_template = game_graph.ColiveEdges({state: set() for state in game_graph.vertices})
    group_liveness_template = game_graph.LiveGroups([])


    marked_unsafe_states = set([vertex for vertex in game_graph.vertices if game_graph.priorities[vertex][index] == -1])

    unsafe_states = get_unsafe_region(game_graph, marked_unsafe_states)

    for state in game_graph.vertices:
        for action, next_state in game_graph.outgoing_edges[state]:
            if state not in unsafe_states and next_state in unsafe_states:
                safety_template.add_unsafe_edge(state, action, next_state)
    
    game_graph_without_unsafe_states = game_graph.subgame(game_graph.vertices - unsafe_states)

    winning_region = compute_template_in_two_player_graph_recursively(game_graph_without_unsafe_states, index, co_live_template, group_liveness_template)

    for state in winning_region:
        for action, next_state in game_graph.outgoing_edges[state]:
            if next_state not in winning_region:
                safety_template.add_unsafe_edge(state, action, next_state)

    make_live_groups_complete(game_graph.vertices, group_liveness_template)

    return winning_region, safety_template, co_live_template, group_liveness_template




def compute_template_in_two_player_graph_recursively(game_graph, index, co_live_template, group_liveness_template):
    
    
    if game_graph.vertices == set():
        return game_graph.vertices
    
    max_priority = game_graph.get_maximum_priority()
    vertices_with_max_priority = set([vertex for vertex in game_graph.vertices if game_graph.priorities[vertex][index] == max_priority ])


    if max_priority % 2 == 1: # max_priority is odd
        regionA = solve_reachability_game(game_graph, vertices_with_max_priority, 1) 
        regionCompA = game_graph.vertices - regionA
        if regionA == game_graph.vertices:
            return regionCompA
        
        subgame = game_graph.subgame(regionCompA)

        subgame_winning_region = compute_template_in_two_player_graph_recursively(subgame, index, co_live_template, group_liveness_template)

        if subgame_winning_region == set():
            return subgame_winning_region
        else:
            regionB = group_liveness_template.add_live_groups_to_reach(game_graph, subgame_winning_region)
            co_live_template.add_edges_between_regions_one_player(subgame_winning_region, game_graph.vertices - subgame_winning_region, game_graph.outgoing_edges)

            subgameB = game_graph.subgame(game_graph.vertices - regionB)
            return regionB.union(compute_template_in_two_player_graph_recursively(subgameB, index, co_live_template, group_liveness_template))
        

    else: # max_priority is even
        regionA = solve_reachability_game(game_graph, vertices_with_max_priority, 0)
        regionCompA = game_graph.vertices - regionA
        if regionA == game_graph.vertices:
            group_liveness_template.add_live_groups_to_reach(game_graph, vertices_with_max_priority)
            return regionA
        
        subgame = game_graph.subgame(regionCompA)

        subgame_winning_region = compute_template_in_two_player_graph_recursively(subgame, index, co_live_template, group_liveness_template)

        if subgame_winning_region == subgame.vertices:
            group_liveness_template.add_live_groups_to_reach(game_graph, vertices_with_max_priority)
            return subgame_winning_region
        else:
            regionB = solve_reachability_game(game_graph, subgame.vertices - subgame_winning_region, 1)
            subgame_without_regionB = game_graph.subgame(game_graph.vertices - regionB)
            return compute_template_in_two_player_graph_recursively(subgame_without_regionB, index, co_live_template, group_liveness_template)



def make_live_group_complete(vertices, live_group):
    for state in vertices:
        if state not in live_group:
            live_group[state] = set()

def make_live_groups_complete(vertices, live_groups):
    for live_group in live_groups.live_groups_list:
        make_live_group_complete(vertices, live_group.live_group_edges)