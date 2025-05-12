# from .datastructures.GameGraphs import TwoPlayerGameGraph
from .Utils import normalize_distribution
import random

class Shield:
    def __init__(self, game_graph, unsafe_edges, colive_edges, live_groups, enforcement_parameter):
        self.game_graph = game_graph
        self.unsafe_edges = unsafe_edges
        self.colive_edges = colive_edges
        self.colive_edges_with_counter = []
        for colive_temp in colive_edges:
            self.colive_edges_with_counter.append({state:{action:(next_state, 0) for action, next_state in colive_temp.get_colive_next_states_with_actions(state)} for state in colive_temp.get_colive_states()})
        # {state:{action:(next_state, 0) for action, next_state in colive_edges.get_colive_next_states_with_actions(state)} for state in colive_edges.get_colive_states()}

        self.live_groups = live_groups
        self.live_groups_counter = []
        for live_group_temp in live_groups:
            self.live_groups_counter.append([0]*len(live_group_temp))
        
        self.enforcement_parameter = enforcement_parameter
    
    def add_live_groups(self, live_groups, index):
        for live_group in live_groups.live_groups_list:
            self.live_groups[index].add_live_group(live_group)
        self.live_groups_counter[index] = [0]*len(self.live_groups[index])

    def set_enforcement_parameter(self, enforcement_parameter):
        self.enforcement_parameter = enforcement_parameter


    def modify_action_distribution(self, state, input_distribution):
        distribution = input_distribution.copy()
        # distribution is a dictionary mapping actions to probabilities
        
        for index in range(len(self.colive_edges)):
            for action in self.colive_edges_with_counter[index][state].keys():
                distribution[action] = distribution[action] - self.enforcement_parameter * self.colive_edges_with_counter[index][state][action][1]

        for index in range(len(self.live_groups)):
            for i in range(len(self.live_groups[index])):
                for action in self.live_groups[index].get_live_group_by_index(i).get_live_actions(state):
                    distribution[action] = distribution[action] + self.enforcement_parameter * self.live_groups_counter[index][i]

        for index in range(len(self.unsafe_edges)):
            for action in self.unsafe_edges[index].get_unsafe_actions(state):
                distribution[action] = 0

        return normalize_distribution(distribution)
    
    def update_counters(self, state, last_action):
        for index in range(len(self.colive_edges)):
            if last_action in self.colive_edges_with_counter[index][state]:
                (next_state, counter) = self.colive_edges_with_counter[index][state][last_action]
                self.colive_edges_with_counter[index][state][last_action] = (next_state, counter + 1)
        
        for index in range(len(self.live_groups)):
            for i in range(len(self.live_groups[index])):
                if last_action in self.live_groups[index].get_live_group_by_index(i).get_live_actions(state):
                    self.live_groups_counter[index][i] = 0
                else:
                    # Increment the counter of the live groups where the state is a live state
                    if state in self.live_groups[index].get_live_group_by_index(i).get_live_states():
                        self.live_groups_counter[index][i] = self.live_groups_counter[index][i] + 1


    def get_next_action(self, state, distribution):
        modified_distribution = self.modify_action_distribution(state, distribution)
        return max(modified_distribution, key=modified_distribution.get)
    
    def get_next_state(self, state, distribution):
        next_action = self.get_next_action(state, distribution)
        self.update_counters(state, next_action)
        return self.game_graph.get_state(state, next_action)
    
    def sample_next_action_and_state(self, state, distribution):
        modified_distribution = self.modify_action_distribution(state, distribution)
        actions = list(modified_distribution.keys())
        probabilities = list(modified_distribution.values())
        next_action = random.choices(actions, probabilities, k=1)[0]
        self.update_counters(state, next_action)
        next_state = self.game_graph.get_state(state, next_action)
        return next_action, next_state


    def reset_counters(self):
        for index in range(len(self.colive_edges)):
            self.colive_edges_with_counter[index] = {state:{action:(next_state, 0) for action, next_state in self.colive_edges[index].get_colive_next_states_with_actions(state)} for state in self.colive_edges[index].get_colive_states()}

        for index in range(len(self.live_groups)):
            self.live_groups_counter[index] = [0]*len(self.live_groups[index])

    def delete_parities(self, parities):
        parities.sort(reversed=True)
        self.game_graph.delete_parities(parities)

        for index in parities:
            del self.unsafe_edges[index]
            del self.colive_edges[index]
            del self.colive_edges_with_counter[index]
            del self.live_groups[index]
            del self.live_groups_counter[index]

    def add_templates(self, unsafe_edges, colive_edges, live_groups):
        self.unsafe_edges.append(unsafe_edges)
        self.colive_edges.append(colive_edges)
        self.colive_edges_with_counter.append({state:{action:(next_state, 0) for action, next_state in colive_edges.get_colive_next_states_with_actions(state)} for state in colive_edges.get_colive_states()})
        self.live_groups.append(live_groups)
        self.live_groups_counter.append([0]*len(live_groups))

    def modify_template_at_index(self, index, unsafe_edges, colive_edges, live_groups):
        self.unsafe_edges[index] = unsafe_edges
        self.colive_edges[index] = colive_edges
        self.colive_edges_with_counter[index] = {state:{action:(next_state, 0) for action, next_state in colive_edges.get_colive_next_states_with_actions(state)} for state in colive_edges.get_colive_states()}
        self.live_groups[index] = live_groups
        self.live_groups_counter[index] = [0]*len(live_groups)

    def add_all_templates(self, unsafe_edges_list, colive_edges_list, live_groups_list):
        for unsafe_edges in unsafe_edges_list:
            self.unsafe_edges.append(unsafe_edges)
        for colive_edges in colive_edges_list:
            self.colive_edges.append(colive_edges)
            self.colive_edges_with_counter.append({state:{action:(next_state, 0) for action, next_state in colive_edges.get_colive_next_states_with_actions(state)} for state in colive_edges.get_colive_states()})
        for live_groups in live_groups_list:
            self.live_groups.append(live_groups)
            self.live_groups_counter.append([0]*len(live_groups))
    
    def clear_template_for_index(self, index):
        self.unsafe_edges[index].clear()
        self.colive_edges[index].clear()
        self.colive_edges_with_counter[index] = {state:{action:(next_state, 0) for action, next_state in self.colive_edges[index].get_colive_next_states_with_actions(state)} for state in self.colive_edges[index].get_colive_states()}
        self.live_groups[index].clear()
        self.live_groups_counter[index] = [0]*len(self.live_groups[index])

    def clear_template_for_indices(self, indices):
        for index in indices:
            self.clear_template_for_index(index)