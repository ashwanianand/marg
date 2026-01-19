from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from grid.gridGenerator import random_grid_generator
from shield.solver.Parity import compute_template_in_two_player_graph
from shield.datastructures.GameGraphs import TwoPlayerGameGraph
from shield.datastructures.Robot import Robot
from marg.shield import Shield
from shield.Utils import grid_to_game_graph, update_game_graph_for_wall
import time
import threading
import json
import os
import webbrowser

app = Flask(__name__)
socketio = SocketIO(app)
grid_layout = []
enforcing_parameter = 0.3
robot_position = None

apply_shield = False

game_graph = None
robot = None
shield = None
winning_region = set()
safety_template = None
co_live_template = None
group_liveness_template = None

kill_robot = threading.Event()  # Event to control pausing

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/reset-program', methods=['POST'])
def reset_program():
    print("Resetting program")
    global grid_layout
    global robot_position
    global game_graph
    global robot
    global shield
    global kill_robot
    global apply_shield
    global enforcing_parameter
    global winning_region
    global safety_template
    global co_live_template
    global group_liveness_template

    apply_shield = False
    enforcing_parameter = 0.3
    winning_region = set()
    safety_template = None
    co_live_template = None
    group_liveness_template = None
    grid_layout = []
    robot_position = None
    game_graph = None
    robot = None
    shield = None
    kill_robot.set()
    return jsonify(status="success")

@app.route('/generate-random-grid', methods=['POST'])
def generate_random_grid():
    global grid_layout 
    global robot_position
    global kill_robot
    kill_robot.set()
    try:
        size = int(request.form['grid_size'])
    except ValueError:
        return jsonify(error="Invalid grid size"), 400

    wall_density = float(request.form['wall_density'])
    grid_layout, robot_position = random_grid_generator(size, wall_density)

    return jsonify(grid_layout=grid_layout, grid_size=size, robot_position=robot_position)

@app.route('/toggle-wall', methods=['POST'])
def toggle_wall():
    global grid_layout
    global game_graph
    global kill_robot
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error="Invalid row or column"), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error="Invalid row or column"), 400
    cell = grid_layout[row][column]
    attributes = cell['attributes'].split(' ')
    if 'wall' in attributes:
        attributes.remove('wall')
    else:
        attributes.append('wall')

    grid_layout[row][column]['attributes'] = ' '.join(attributes)
    update_game_graph_for_wall(game_graph, grid_layout, (row, column))

    kill_robot.set()
    print(f"toggle wall: {row}, {column}")

    return jsonify(grid_layout=grid_layout) 


@app.route('/toggle-cell-state', methods=['POST'])
def toggle_cell_state():
    global grid_layout
    global kill_robot
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error="Invalid row or column"), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error="Invalid row or column"), 400
    cell = grid_layout[row][column]
    new_state = (cell['cell_color'] + 1) % 4

    grid_layout[row][column]['cell_color'] = new_state

    kill_robot.set()
    print(f"toggle cell state ({row}, {column}) to {new_state}")

    return jsonify(grid_layout=grid_layout)


@app.route('/toggle-initial-state', methods=['POST'])
def toggle_initial_state():
    global grid_layout
    global game_graph
    global kill_robot
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error="Invalid row or column"), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error="Invalid row or column"), 400
    for row_cells in grid_layout:
        for cell in row_cells:
            attributes = cell['attributes'].split(' ')
            if 'robot' in attributes:
                attributes.remove('robot')
            cell['attributes'] = ' '.join(attributes)
    

    cell = grid_layout[row][column]
    attributes = cell['attributes'].split(' ')
    attributes.append('robot')
    grid_layout[row][column]['attributes'] = ' '.join(attributes)
    game_graph.set_initial_state((row, column))
    print(f"the initial state is now {game_graph.initial_state}")
    

    kill_robot.set()
    

    return jsonify(grid_layout=grid_layout)

@app.route('/modify-weight', methods=['POST'])
def modify_weight():
    global grid_layout
    global game_graph
    global kill_robot
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
        weight = int(request.form['weight'])
    except ValueError:
        return jsonify(error="Invalid row or column"), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error="Invalid row or column"), 400

    grid_layout[row][column]['weight'] = weight

    kill_robot.set()
    print(f"modify weight ({row}, {column}) to {weight}")

    return jsonify(grid_layout=grid_layout)

@app.route('/toggle-owner', methods=['POST'])
def toggle_owner():
    global game_graph
    global grid_layout
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error="Invalid row or column"), 400
    
    grid_layout[row][column]['owner'] = 1 - grid_layout[row][column]['owner']
    game_graph.assign_owner((row, column), 1 - game_graph.ownership[(row, column)])
    kill_robot.set()
    print(f"toggle owner ({row}, {column}) to {game_graph.ownership[(row, column)]}")
    return jsonify(status="success")



@socketio.on('save_grid')
def save_grid():
    global grid_layout
    global robot_position
    save_path = os.path.join(os.path.dirname(__file__), 'saved_grid.json')
    with open(save_path, 'w') as f:
        json.dump({'grid_layout': grid_layout, 'robot_position': robot_position}, f)


# @socketio.on('toggle_shield')
@app.route('/toggle-shield', methods=['POST'])
def toggle_shield():
    global apply_shield
    selected_option = request.form['use_shield']
    apply_shield = selected_option == 'use_shield'
    print(f"apply shield: {apply_shield}")
    return jsonify(status="success")

@socketio.on('update_enforcing_parameter')
def update_parameter(data):
    global enforcing_parameter
    global shield
    enforcing_parameter = data['value']
    if shield is not None:
        shield.set_enforcement_parameter(enforcing_parameter)
    print(f"Updated enforcing parameter: {enforcing_parameter}")

@socketio.on('compute_game')
def compute_game():
    global grid_layout
    global robot_position
    global kill_robot

    global game_graph

    kill_robot.clear()
    try:
        game_graph = grid_to_game_graph(grid_layout, robot_position)
        print("game graph created")
    except Exception as e:
        print("creating game error", e)
        return jsonify(error=str(e)), 400
    
    return jsonify(status="success")


@socketio.on('solve_game')
def solve_game():
    global grid_layout
    global robot_position
    global kill_robot

    global winning_region
    global safety_template
    global co_live_template
    global group_liveness_template
    global game_graph
    global enforcing_parameter
    global shield

    kill_robot.clear()
    try:
        print("Solving game")
        winning_region, safety_template, co_live_template, group_liveness_template = compute_template_in_two_player_graph(game_graph)
        print("template computed", winning_region)
    except Exception as e:
        print("solve game error", e)
        socketio.emit('solving_error', {'error': str(e)})
        # return jsonify(error=str(e)), 400
    

    shield = Shield(game_graph, safety_template, co_live_template, group_liveness_template, enforcing_parameter)
    shield.set_enforcement_parameter(enforcing_parameter)
    print("Shield created")
    socketio.emit('solving_success', {'winning_region': list(winning_region)})
    print("Solving success")
    # return jsonify(status="success")

@socketio.on('prepare_robot')
def prepare_robot():
    global robot
    global game_graph
    global robot_position
    print("Preparing robot")
    robot = Robot(game_graph, robot_position)
    robot.set_randomized_strategy()
    print(robot)
    return jsonify(status="success")

@socketio.on('start_robot')
def start_robot():
    global kill_robot
    print(f"Starting robot with {kill_robot.is_set()}")
    threading.Thread(target=move_robot, daemon=True).start()

@socketio.on('print_info')
def print_info(data):
    print(data)

def move_robot():
    global grid_layout
    global game_graph
    global robot
    global shield
    global apply_shield
    global kill_robot
    global enforcing_parameter
    global robot_position
    while not kill_robot.is_set():
        print("Moving robot")
        time.sleep(0.5)  # Simulate a delay for movement
        next_state = None
        if apply_shield:
            action_distribution = robot.get_strategy_distribution(robot.robot_position)
            # next_state = shield.get_next_state(robot.robot_position, action_distribution)
            next_action, next_state = shield.sample_next_action_and_state(robot.robot_position, action_distribution)
        else:
            # next_state = robot.get_next_state_by_max()
            next_action, next_state = robot.sample_next_action_and_state()

        print(f"Next action: {next_action}, Next state: {next_state}")
        robot.set_robot_position(next_state)
        robot_position = next_state  # Update the global robot_position

        socketio.emit('update_position', {'position': robot_position})


if __name__ == '__main__':
    # url = "http://127.0.0.1:5000"
    # webbrowser.open("http://127.0.0.1:5000", new=1)
    socketio.run(app, debug=False)
