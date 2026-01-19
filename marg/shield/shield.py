
from .Utils import normalize_distribution
import random
# random.seed(42)  # Replace 42 with your desired seed


class Shield:
    """
    Implements liveness shield using strategy templates.
    
    The shield modifies action distributions to ensure safety (avoiding unsafe states)
    and liveness (visiting goal states infinitely often) properties.
    
    Attributes:
        game_graph: The underlying game graph representing the environment
        unsafe_edges: Edges leading to unsafe states
        colive_edges: Edges relevant to co-liveness properties
        live_groups: Groups of states for liveness guarantees
        enforcement_parameter: Weight for shield corrections (0-1)
        
    Example:
        >>> shield = Shield(game_graph, unsafe_edges, colive_edges, 
        ...                 live_groups, enforcement_parameter=0.3)
        >>> modified_dist = shield.modify_action_distribution(state, distribution)
    """
    def __init__(self, game_graph, unsafe_edges, colive_edges, live_groups, enforcement_parameter, universal_counter=False, max_action_repetitions=100, logger=None):
        self.universal_counter = universal_counter
        self.game_graph = game_graph
        self.unsafe_edges = unsafe_edges
        self.colive_edges = colive_edges
        self.colive_edges_with_counter = {state:{action:(next_state, 0) for action, next_state in colive_edges.get_colive_next_states_with_actions(state)} for state in colive_edges.get_colive_states()}

        self.live_groups = live_groups
        self.live_groups_counter = [0] if universal_counter else [0]*len(live_groups) 
        self.enforcement_parameter = enforcement_parameter
        self.max_action_repetitions = max_action_repetitions
        self.last_action = None
        self.num_repetitions = 0

        if logger is not None:
            self.logger = logger
        else:
            self.logger = None

        if self.logger is not None:
            self.logger.info(f"There are {len(self.live_groups)} live groups in the shield.")

    def set_enforcement_parameter(self, enforcement_parameter):
        self.enforcement_parameter = enforcement_parameter

    def modify_action_distribution(self, state, input_distribution):
        """
        The function modifies a given action distribution based on unsafe actions and colive/live edges.    
        
        Args:
            state: Current state as (row, column) tuple
            input_distribution: Original action probabilities {action: probability}
            
        Returns:
            Modified and normalized distribution {action: probability}
            
        Raises:
            ValueError: If no actions are available for the state
            
        Example:
            >>> dist = {'left': 0.5, 'right': 0.5}
            >>> modified = shield.modify_action_distribution((1, 1), dist)
            >>> sum(modified.values())  # Always sums to 1.0
            1.0
        """
        distribution = input_distribution.copy()
        # distribution is a dictionary mapping actions to probabilities
        # for action in distribution.keys():
        #     if action not in self.game_graph.get_actions(state):
        #         distribution[action] = 0

        if self.logger is not None:
            self.logger.debug(f"Modifying action distribution for state {state}. Input distribution: {input_distribution}.")

        # if the action has been repeated too many times, remove it from the distribution
        if self.num_repetitions >= self.max_action_repetitions:
            distribution[self.last_action] = 0
            self.num_repetitions = 0
            if self.logger is not None:
                self.logger.warning(f"Action {self.last_action} has been repeated too many times. Setting its probability to 0.")

        for action in self.unsafe_edges.get_unsafe_actions(state):
            distribution[action] = 0
            if self.logger is not None:
                self.logger.debug(f"Action {action} is unsafe in state {state}. Setting its probability to 0.")
        
        for action in self.colive_edges_with_counter[state].keys():
            distribution[action] = distribution[action] - self.enforcement_parameter * self.colive_edges_with_counter[state][action][1]

        
        for i in range(len(self.live_groups)):
            for action in self.live_groups.get_live_group_by_index(i).get_live_actions(state):
                if self.universal_counter:
                    distribution[action] = distribution[action] + self.enforcement_parameter * self.live_groups_counter[0]
                else:
                    distribution[action] = distribution[action] + self.enforcement_parameter * self.live_groups_counter[i]

        try:
            normalized_distribution = normalize_distribution(distribution)
            if self.logger is not None:
                self.logger.debug(f"Normalized distribution for state {state}: {normalized_distribution}.")
            return normalized_distribution
        except ZeroDivisionError:
            error_msg = f"All actions have zero probability for state {state}"
            if self.logger is not None:
                self.logger.error(error_msg)
            
            # Return uniform distribution over available actions
            actions = list(distribution.keys())
            if not actions:
                raise ValueError(f"No actions available for state {state}")
            
            uniform_prob = 1.0 / len(actions)
            return dict.fromkeys(actions, uniform_prob)
            
            
    
    def update_counters(self, state, last_action):
        """
        This function updates the counters for colive edges and live groups based on the last action taken in a given state.
        
        - state: The current state in the game graph.
        - last_action: The last action taken in the current state.
        """
        if last_action in self.colive_edges_with_counter[state]:
            (next_state, counter) = self.colive_edges_with_counter[state][last_action]
            self.colive_edges_with_counter[state][last_action] = (next_state, counter + 1)
        

        for i in range(len(self.live_groups)):
            if last_action in self.live_groups.get_live_group_by_index(i).get_live_actions(state):
                if self.universal_counter:
                    self.live_groups_counter[0] = 1
                else:
                    self.live_groups_counter[i] = 1
            else:
                if self.universal_counter:
                    self.live_groups_counter[0] += 1
                else:
                    # Increment the counter of the live groups where the state is a live state
                    if state in self.live_groups.get_live_group_by_index(i).get_live_states():
                        self.live_groups_counter[i] += 1


    def get_next_action(self, state, distribution):
        modified_distribution = self.modify_action_distribution(state, distribution)
        return max(modified_distribution.keys(), key=lambda k: modified_distribution[k])
    
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

        if self.last_action is None or next_action != self.last_action:
            self.num_repetitions = 0
            self.last_action = next_action
        else:
            self.num_repetitions += 1

        return next_action, next_state


    def reset_counters(self):
        self.colive_edges_with_counter = {state:{action:(next_state, 0) for action, next_state in self.colive_edges.get_colive_next_states_with_actions(state)} for state in self.colive_edges.get_colive_states()}
        self.live_groups_counter = [0]*len(self.live_groups)