from .GameGraphs import TwoPlayerGameGraph
import random
from pprint import pformat
from ..Utils import normalize_distribution

class Robot:
    def __init__(self, game_graph, robot_position):
        self.game_graph = game_graph
        self.robot_position = robot_position
        self.strategy_distribution = {state: {action: 0} for state in game_graph.vertices for action in game_graph.get_actions(state)}
    
    def __str__(self):
        return f"Robot at {self.robot_position} with strategy distribution {pformat(self.strategy_distribution)}"
        
    def update_strategy_distribution(self, state, action, probability):
        if probability == 0:
            probability = 0.0000001
        self.strategy_distribution[state][action] = probability
    
    def set_randomized_strategy_for_state(self, state):
        actions = self.game_graph.get_actions(state)
        # get random probability distribution over actions
        random_values = [random.random() for _ in range(len(actions))]
        total = sum(random_values)
        for i in range(len(actions)):
            self.update_strategy_distribution(state, actions[i], random_values[i]/total)

    def set_distribution_for_strategy(self, state, strategy, delta):
        actions = self.game_graph.get_actions(state)
        chosen_action = strategy[state]
        if chosen_action != None:
            self.update_strategy_distribution(state, chosen_action[0], 1)
            for action in actions:
                if action != chosen_action[0]:
                    self.update_strategy_distribution(state, action, delta)
        else:
            for action in actions:
                self.update_strategy_distribution(state, action, delta)
        self.strategy_distribution[state] = normalize_distribution(self.strategy_distribution[state])
        

    def set_randomized_strategy(self):
        for state in self.game_graph.vertices:
            self.set_randomized_strategy_for_state(state)

    def set_robot_position(self, robot_position):
        self.robot_position = robot_position

    def get_strategy_distribution(self, state):
        return self.strategy_distribution[state]
    
    def get_next_action_from_state_by_max(self, state):
        distribution = self.strategy_distribution[state]
        return max(distribution, key=distribution.get)
    
    def get_next_state_from_state_by_max(self, state):
        return self.game_graph.get_state(state, self.get_next_action_from_state_by_max(state))
    
    def get_next_action_by_max(self):
        return self.get_next_action_from_state_by_max(self.robot_position)
    
    def get_next_state_by_max(self):
        return self.get_next_state_from_state_by_max(self.robot_position)
    
    def sample_next_action_and_state(self):
        distribution = self.strategy_distribution[self.robot_position]
        actions = list(distribution.keys())
        probabilities = list(distribution.values())
        next_action = random.choices(actions, probabilities, k=1)[0]
        next_state = self.game_graph.get_state(self.robot_position, next_action)
        return next_action, next_state
    
    def set_strategy_distribution_from_strategy(self, strategy, delta, adversarial_policy):
        for state in strategy.keys():
            outgoing_edges = self.game_graph.get_next_states(state)
            if outgoing_edges == []:
                continue
            self.set_distribution_for_strategy(state, strategy, delta)
        for state in adversarial_policy:
            self.strategy_distribution[state] = adversarial_policy[state]

        

