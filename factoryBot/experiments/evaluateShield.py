import json
import sys
import os
# Add the submission directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from marg.shield.shield import Shield
from marg.shield.datastructures.Robot import Robot
from marg.shield.Utils import grid_to_game_graph, make_graph_adversarial, normalize_distribution, show_grid
from marg.shield.solver.MeanPayoff import compute_value_function as compute_value_function_mean_payoff
from marg.shield.solver.Parity import compute_template_in_one_player_graph
from time import time

def load_grid_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        grid = data.get('grid', [])
        robot_position = data.get('robot_position', (0, 0))
        return grid, (robot_position[0], robot_position[1])


def run_robot_for_steps(robot, steps):
    buechi_region = [state for state in robot.game_graph.vertices if robot.game_graph.priorities[state][0] == 2]
    buechi_counter = 0
    sum_of_rewards = robot.game_graph.vertex_labels[robot.robot_position]
    decision_time = 0
    for _ in range(steps):
        start_decision_time = time()
        next_action, next_state = robot.sample_next_action_and_state()
        end_decision_time = time()
        decision_time += end_decision_time - start_decision_time
        # print(f"Next action: {next_action}, Next state: {next_state}")
        robot.set_robot_position(next_state)
        if next_state in buechi_region:
            buechi_counter += 1
        sum_of_rewards += robot.game_graph.vertex_labels[next_state]
    
    return (buechi_counter/steps), (sum_of_rewards/steps), (decision_time/steps)

def run_robot_with_shield_for_steps(robot, shield, steps):

    buechi_region = [state for state in robot.game_graph.vertices if robot.game_graph.priorities[state][0] == 2]
    buechi_counter = 0
    sum_of_rewards = robot.game_graph.vertex_labels[robot.robot_position]
    decision_time = 0
    for _ in range(steps):
        
        start_decision_time = time()
        action_distribution = robot.get_strategy_distribution(robot.robot_position)
        # next_state = shield.get_next_state(robot.robot_position, action_distribution)
        next_action, next_state = shield.sample_next_action_and_state(robot.robot_position, action_distribution)
        end_decision_time = time()
        decision_time += end_decision_time - start_decision_time
        # print(f"Next action: {next_action}, Next state: {next_state}")
        robot.set_robot_position(next_state)
        if next_state in buechi_region:
            buechi_counter += 1
        sum_of_rewards += robot.game_graph.vertex_labels[next_state]
    
    return (buechi_counter/steps), (sum_of_rewards/steps), (decision_time/steps)


def evaluate_instance(json_file_path, steps):
    enforcement_parameters = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]
    stats = {"reachable": False,
             "enforcement_parameters": enforcement_parameters,
             "buechi_counter_unshielded": [],
             "average_rewards_unshielded": [],
             "shield_computation_time": [],
             "total_computation_time": [],
             "buechi_counter_shielded": [],
             "average_rewards_shielded": [],
             "average_decision_time_shielded": [],
             "average_decision_time_unshielded": []}

    # Start the timer for shield computation
    start_time = time()

    # Load the grid and robot position from the json file
    grid, robot_position = load_grid_from_json(json_file_path)

    # Convert the grid to a game graph
    game_graph = grid_to_game_graph(grid, robot_position)

    # Make the game graph adversarial
    adversarial_game_graph = make_graph_adversarial(game_graph)

    # Compute the template for the robot
    _, safety_template, co_live_template, group_liveness_template = compute_template_in_one_player_graph(game_graph, 0)

    # Compute the shield
    shield = Shield(game_graph, safety_template, co_live_template, group_liveness_template, 0.02)

    # Stop the timer for shield computation
    end_time = time()
    stats['shield_computation_time'].append(end_time - start_time)

    # Compute the value function and strategy for the robot
    value_function, strategy = compute_value_function_mean_payoff(adversarial_game_graph)

    # Initialize the adversarial policy
    adversarial_policy = {state: {action: 0} for state in game_graph.get_player_vertices(2) for action in game_graph.get_actions(state)}

    for state in adversarial_game_graph.get_player_vertices(1):
        adversarial_policy[state] = {action: 0 for action in game_graph.get_actions(state)}
        for action, _ in game_graph.outgoing_edges[state]:
            adversarial_policy[state][action] = 1
        adversarial_policy[state] = normalize_distribution(adversarial_policy[state])

    total_computation_time = time() - start_time
    stats['total_computation_time'].append(total_computation_time)


    # Check if the initial state is reachable
    value_at_initial_state = value_function[robot_position]
    buechi_region = [state for state in game_graph.vertices if game_graph.priorities[state][0] == 2]

    stats['value_at_initial_state'] = value_at_initial_state

    for state in buechi_region:
        if value_function[state] == value_at_initial_state:
            stats['reachable'] = True
            break                                      

    # For every enforcement parameter, run the robot and the robot with shield
    robot_unshielded = Robot(game_graph, robot_position)
    robot_shielded = Robot(game_graph, robot_position)
    for enforcement_parameter in enforcement_parameters:
        robot_unshielded.set_strategy_distribution_from_strategy(strategy, enforcement_parameter, adversarial_policy)
        robot_shielded.set_strategy_distribution_from_strategy(strategy, enforcement_parameter, adversarial_policy)

        shield.set_enforcement_parameter(enforcement_parameter)
        shield.reset_counters()

        unshielded_frequency, unshielded_rewards, average_decision_time_unshielded = run_robot_for_steps(robot_unshielded, steps)
        shielded_frequency, shielded_rewards, average_decision_time_shielded = run_robot_with_shield_for_steps(robot_shielded, shield, steps)

        stats['buechi_counter_unshielded'].append(unshielded_frequency)
        stats['average_rewards_unshielded'].append(unshielded_rewards)
        stats['buechi_counter_shielded'].append(shielded_frequency)
        stats['average_rewards_shielded'].append(shielded_rewards)
        stats['average_decision_time_shielded'].append(average_decision_time_shielded)
        stats['average_decision_time_unshielded'].append(average_decision_time_unshielded)

    return stats

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 evaluateShield.py <steps> <json_file_path> <output_file_path>")
        sys.exit(1)

    steps = int(sys.argv[1])
    json_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    stats = evaluate_instance(json_file_path, steps)

    with open(output_file_path, 'w') as file:
        json.dump(stats, file)

