
#========================================================================================
#
# Marg: A Web Interface for Shield
#
#========================================================================================


"""
The program is a web application for Marg.

The flask application is used to create a web interface for Marg. The user can create a grid, 
modify the grid, and solve the grid using the shield algorithm. The user can add/delete/modify 
walls, objectives, and robot position. The user can also apply the shield algorithm to the 
grid and see the robot move in the grid.
"""


#========================================================================================
#
# TODO
# 1. Add the ability to add/delete/modify objectives [DONE]
# 2. Better return messages
#
#========================================================================================



#========================================================================================
#
# Imports
#
#========================================================================================

# Standard library imports
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, g
from flask_socketio import SocketIO, disconnect, emit, join_room, leave_room
from flask_session import Session
import itertools
import json
import logging
import os
import time
import threading
from uuid import uuid4

# Local application imports
from grid.gridGenerator import iterate_over_cells, random_grid_with_buechi_region_meanpayoff_region as random_mpb_grid_generator, random_objective, modify_grid_objectives
from shield.solver.Parity import compute_template_in_two_player_graph
from shield.solver.MeanPayoff import compute_value_function as compute_value_function_mean_payoff
from shield.datastructures.Robot import Robot
from shield.shield import Shield
from shield.Utils import grid_to_game_graph, update_game_graph_for_wall, make_graph_adversarial, normalize_distribution, move_robot_on_grid_layout



#========================================================================================
#
# Constants and Global Variables
#
#========================================================================================

# Constants
ROW_COL_ERROR = "Invalid row or column"
INPUT_ERROR = "Input error"
INIT_ENFORCEMENT_PARAMETER = 0.5
USER_TIMEOUT = 300 # User will be disconnected if there is not request in last 10 mins
CLEANUP_INTERVAL = 30

# Grid generation constants
NEW_REGION_RATIO = 0.2
REGION_RATIO = 0.2
MIN_DISTANCE = 3
MAX_DISTANCE = 6
IGNORE_ATTRIBUTES = False
RATIO_OF_PROBABILISTIC_CELLS = 0.2

# Logging constants
INFO = "[INFO] "

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_PERMANENT=False,
    SESSION_FILE_DIR='/tmp/flask_session',
    SESSION_FILE_THRESHOLD=100,
    SESSION_COOKIE_NAME='marg_session',
    SESSION_COOKIE_SECURE=True,   # Only send cookies over HTTPS
    SESSION_COOKIE_HTTPONLY=True, # Prevent JavaScript access
    SESSION_COOKIE_SAMESITE='Strict' # Protect against CSRF attacks
)
socketio = SocketIO(app)
Session(app)

user_data = {}



#========================================================================================
#
# Logging and Telemetry
#
#========================================================================================


#========================================================================================
#
# Connection - Old
#
#========================================================================================

# Prepare the client connection
# @app.before_request
# def before_request():
#     """Enforce HTTPS."""
    
#     if request.headers.get("X-Forwarded-Proto") == "http":
#         return redirect(request.url.replace("http://", "https://"), code=301)
    

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/user-exists')
# def user_exists():
#     exists = ("user_id" in session) and (session.get('user_id') in user_data)
#     if exists:
#         user_grid_layout = user_data[session.get('user_id')]['grid_layout']
#         return jsonify(exists="true", grid_layout=user_grid_layout)
#     else:
#         return jsonify(exists="false", grid_layout="")
    
# @app.after_request
# def after_request(response):
#     print(f"Cookies being sent: {response.headers.get('Set-Cookie')}")
#     return response

# @app.route('/debug-session')
# def debug_session():
#     print(f"Session Data: {session.items()}")
#     return jsonify(session_data=dict(session))

# @socketio.on('connect')
# def handle_connect():
#     # Assign a unique user ID if it doesn't exist
#     if ("user_id" not in session) or (session.get('user_id') not in user_data):
#         if "user_id" not in session:
#             session['user_id'] = str(uuid4())
#             print(f"Created user: {session.get('user_id')}")
#         else:
#             print(f"Using old session for user: {session.get('user_id')}")
#         initialize_user(session.get('user_id'), request.sid)
#         print(f"User {session.get('user_id')} connected with socket ID {request.sid}")
#     else:
#         print(f"User exists: {session.get('user_id')}")
#         user_data[session.get('user_id')]['socket_id'] = request.sid
#         print(f"Existing user {session.get('user_id')} connected with socket ID {request.sid}")
    

    

# @socketio.on('disconnect')
# def handle_disconnect():
#     user_id = session.get('user_id')
#     if user_id and user_id in user_data:
#         socketio.emit('user_disconnected', room=user_data[user_id]['socket_id'])
#         print(f"User {user_id} disconnected")



#========================================================================================
#
# Connection - New
#
#========================================================================================

# Prepare the client connection
@app.before_request
def before_request():
    """Enforce HTTPS."""
    
    if request.headers.get("X-Forwarded-Proto") == "http":
        return redirect(request.url.replace("http://", "https://"), code=301)
    

@app.route('/')
def home():
    print("Handling request")
    if ("user_id" not in session) or (session.get('user_id') not in user_data):
        if "user_id" not in session:
            session['user_id'] = str(uuid4())
            print(f"Created user: {session.get('user_id')}")
        else:
            print(f"Using old session for user: {session.get('user_id')}")
        print(f"User {session.get('user_id')} connected")
    else:
        print(f"Existing user {session.get('user_id')} connected")

    return render_template('index.html')

@app.route('/user-exists')
def user_exists():
    exists = ("user_id" in session) and (session.get('user_id') in user_data)
    if exists:
        user_grid_layout = user_data[session.get('user_id')]['grid_layout']
        return jsonify(exists="true", grid_layout=user_grid_layout)
    else:
        return jsonify(exists="false", grid_layout="")
    
@socketio.on('connect')
def handle_connect():
    if ("user_id" not in session) or (session.get('user_id') not in user_data):
        if "user_id" not in session:
            print("user id not in session")
            disconnect()
            return
        else:
            print(f"Using old session for user: {session.get('user_id')}")
            initialize_user(session.get('user_id'), request.sid)
    else:
        user_data[session.get('user_id')]['socket_id'] = request.sid
    print(f"User {session.get('user_id')} connected with socket ID {request.sid}")

    

@socketio.on('disconnect')
def handle_disconnect():
    user_id = session.get('user_id')
    if user_id and user_id in user_data:
        socketio.emit('user_disconnected', room=user_data[user_id]['socket_id'])

        # Save the user data to a file 
        save_path = os.path.join(os.path.dirname(__file__), f'user_logs/{user_id}.json')
        with open(save_path, 'w') as f:
            json.dump(user_data[user_id], f)
        print(f"User data saved to {save_path}")
        print(f"User {user_id} disconnected")




#========================================================================================
#
# Utils
#
#========================================================================================

def get_user_from_socket_id(sid):
    for user in user_data:
        if user_data[user]['socket_id'] == sid:
            return user
    return None


def send_error_message_to_user(user_id):
    sid = user_data[user_id]['socket_id']

    # Save the user data to a file 
    save_path = os.path.join(os.path.dirname(__file__), f'user_logs/{user_id}.json')
    with open(save_path, 'w') as f:
        # write that the user had an error in the file 
        f.write(json.dumps({"error": "User had an error"}))
        f.write("\n")
        f.write(json.dumps(user_data[user_id]))
        f.write("\n")
        json.dump(user_data[user_id], f)
    print(f"User data saved to {save_path}")

    socketio.emit('display-error', room=sid)


def disconnect_user(user_id):
    socket_id = user_data[user_id]['socket_id']
    if socket_id:
        socketio.emit('user_disconnected', room=socket_id)

    # Save the user data to a file 
    save_path = os.path.join(os.path.dirname(__file__), f'user_logs/{user_id}.json')
    with open(save_path, 'w') as f:
        json.dump(user_data[user_id], f)
    print(f"User data saved to {save_path}")
    # Disconnect the user from the socket
    user_data.pop(user_id, None)


def initialize_user(user_id, sid):
    user_data[user_id] = {
            "user_id": session.get('user_id'),
            "grid_layout": [],
            "robot_position": None,
            "game_graph": None,
            "robot": None,
            "shield": None,
            "winning_region": set(),
            "safety_template": [],
            "co_live_template": [],
            "group_liveness_template": [],
            "random_player_strategy": None,
            "mean_payoff_strategy": None,
            "mean_payoff_modified": True,
            "parity_modified_indices": [],
            "enforcing_parameter": INIT_ENFORCEMENT_PARAMETER,
            "apply_shield": False,
            "socket_id": sid,
            "last_request": time.time(),
            "kill_robot": threading.Event()
    }


def kill_bot(user_id):
    # user_data[user_id]['kill_robot'].set()
    user_data[user_id]['kill_robot'] = True

def resume_bot(user_id):
    # user_data[user_id]['kill_robot'].clear()
    user_data[user_id]['kill_robot'] = False
    move_robot(user_id)

def process_request(user_id, pause_bot=True):
    user_data[user_id]['last_request'] = time.time()

    if pause_bot:
        kill_bot(user_id)



#========================================================================================
#
# Setup
#
#========================================================================================

#========================================================================================
#
# Routes
#
#========================================================================================


@app.route('/reset-program', methods=['POST'])
def reset_program():
    print("Resetting program")
    if "user_id" in session:
        user_id = session.get('user_id')
        print(f"Resetting user {user_id}")
        initialize_user(user_id, user_data[user_id]['socket_id'])
    return jsonify(status="success")


@app.route('/generate-random-grid', methods=['POST'])
def generate_random_grid():

    user_id = session.get('user_id')
    process_request(user_id, True)

    try:
        size = int(request.form['grid_size'])
    except ValueError:
        return jsonify(error="Invalid grid size"), 400

    wall_density = float(request.form['wall_density'])

    print("Generating random grid")
    grid_layout, robot_position, _ = random_mpb_grid_generator(size, 
                                                               wall_density, 
                                                               region_ratio=REGION_RATIO, 
                                                               min_distance=MIN_DISTANCE, 
                                                               max_distance=MAX_DISTANCE, 
                                                               ignore_attributes=IGNORE_ATTRIBUTES,
                                                               probabilistic_ratio=RATIO_OF_PROBABILISTIC_CELLS)

    user_data[user_id]['grid_layout'] = grid_layout
    user_data[user_id]['robot_position'] = robot_position

    print("Generated random grid")
    return jsonify(grid_layout=grid_layout, grid_size=size, robot_position=robot_position) # TODO: send success message




@app.route('/toggle-wall', methods=['POST'])
def toggle_wall():
    user_id = session.get('user_id')
    process_request(user_id, True)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']
    
    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error=ROW_COL_ERROR), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error=ROW_COL_ERROR), 400

    cell = grid_layout[row][column]
    attributes = cell['attributes'].split(' ')
    if 'wall' in attributes:
        attributes.remove('wall')
    else:
        attributes.append('wall')
    grid_layout[row][column]['attributes'] = ' '.join(attributes)

    update_game_graph_for_wall(game_graph, grid_layout, (row, column))

    print(f"toggle wall: {row}, {column}")

    return jsonify(grid_layout=grid_layout)  # TODO: send success message


# TODO: Modify the code here
# @app.route('/modify-parity', methods=['POST'])
# def modify_parity():
#     #  Ask the server to stop the robot from following the modifiable parities
#     # Contains indices of parity objectives 
#     user_id = session.get('user_id')
#     grid_layout = user_data[user_id]['grid_layout']
#     game_graph = user_data[user_id]['game_graph']

#     try:
#         row = int(request.form['row'])
#         column = int(request.form['column'])
#         index = int(request.form['index'])
#         result = int(request.form['result'])
#     except ValueError:
#         return jsonify(error=INPUT_ERROR), 400
    
#     grid_layout[row][column]['weight'][index] = ((grid_layout[row][column]['weight'][index] + 1) % 3) + 1
#     game_graph.modify_parity((row, column), index, grid_layout[row][column]['weight'][index])
#     if result != grid_layout[row][column]['weight'][index]:
#         print(f"Error in modifying parity ({row}, {column}) to {grid_layout[row][column]['weight'][index]}")
#         return jsonify(error="Error in modifying parity"), 400
#     return jsonify("success")



@app.route('/toggle-owner', methods=['POST'])
def toggle_owner():
    user_id = session.get('user_id')
    process_request(user_id, True)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']

    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error=ROW_COL_ERROR), 400
    
    user_data[user_id]['mean_payoff_modified'] = True
    grid_layout[row][column]['owner'] = 1 - grid_layout[row][column]['owner']
    game_graph.assign_owner((row, column), 1 - game_graph.ownership[(row, column)])
    print(f"toggle owner ({row}, {column}) to {game_graph.ownership[(row, column)]}")
    return jsonify(status="success")


@app.route('/modify-weights', methods=['POST'])
def modify_weights():
    user_id = session.get('user_id')
    process_request(user_id, True)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']

    try:
        weights = json.loads(request.form['weights'])
    except ValueError:
        return jsonify(error=ROW_COL_ERROR), 400

    for (row, column) in iterate_over_cells(len(grid_layout)//2):
        if grid_layout[row][column]['weight'][0] != weights[row][column]:
            user_data[user_id]['mean_payoff_modified'] = True
            grid_layout[row][column]['weight'][0] = weights[row][column]
            game_graph.modify_mp((row, column), weights[row][column])

    print("Modified weights")
    return jsonify(status="success", grid_layout=grid_layout) # TODO: send success message


@app.route('/toggle-initial-state', methods=['POST'])
def toggle_initial_state():
    
    user_id = session.get('user_id')
    process_request(user_id, True)
    
    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']
    

    try:
        row = int(request.form['row'])
        column = int(request.form['column'])
    except ValueError:
        return jsonify(error=ROW_COL_ERROR), 400

    try:
        grid_layout[row][column]
    except IndexError:
        return jsonify(error=ROW_COL_ERROR), 400
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
    
    return jsonify(grid_layout=grid_layout) # TODO: send success message



@app.route('/add-parity', methods=['POST'])
def add_parity():
    user_id = session.get('user_id')
    process_request(user_id, False)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']

    new_parity = {}

    try:
        random = request.form['random']
    except ValueError:
        return jsonify(error=INPUT_ERROR), 400

    if random == 'true':
        print("Adding random parity")
        region_size = int(NEW_REGION_RATIO * (len(grid_layout) // 2) * (len(grid_layout) // 2))
        new_parity = random_objective(len(grid_layout), region_size)
    else:
        print("Adding empty parity")
        new_parity = {cell: 1 for cell in game_graph.vertices}
    
    success = game_graph.add_parity(new_parity)

    if success:
        add_objective_to_grid(new_parity, grid_layout)
        return jsonify(status="success", grid_layout=grid_layout)
    else:
        return jsonify(error="Invalid objective"), 400

def add_objective_to_grid(objective, grid_layout):
    for (row, column) in iterate_over_cells(len(grid_layout)//2):
        cell = grid_layout[row][column]
        cell['weight'].append(objective[(row, column)])
        grid_layout[row][column] = cell    


@app.route('/delete-parity', methods=['POST'])
def delete_parity():
    user_id = session.get('user_id')
    process_request(user_id, False)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']

    try:
        indices = json.loads(request.form['selected_indices'])
    except ValueError:
        return jsonify(error=INPUT_ERROR), 400
    
    # indices in the reverse sorted order
    indices = [int(index) for index in indices]
    indices = sorted(indices, reverse=True)

    game_graph.delete_parity(indices)
    if user_data[user_id]['shield'] is not None:
        user_data[user_id]['shield'].game_graph = game_graph
        user_data[user_id]['shield'].delete_parities(indices)
    for (row, column) in iterate_over_cells(len(grid_layout)//2):
        cell = grid_layout[row][column]
        cell['weight'] = [cell['weight'][i] for i in range(len(cell['weight'])) if i not in indices]
        grid_layout[row][column] = cell
    return jsonify(success="success", grid_layout=grid_layout)



@app.route('/modify-parity', methods=['POST'])
def modify_parity():
    user_id = session.get('user_id')
    process_request(user_id, False)

    shield = user_data[user_id]['shield']

    try:
        indices = json.loads(request.form['indices'])
    except ValueError:
        return jsonify(error=INPUT_ERROR), 400
    
    indices = [int(index) for index in indices]
    indices = sorted(indices, reverse=True)
    user_data[user_id]['parity_modified_indices'] = indices
    clear_template_for_user(user_id, indices)
    if shield is not None:
        shield.clear_template_for_indices(indices) 
    return jsonify(success="success")

def clear_template_for_user(user_id, indices):
    for index in indices:
        user_data[user_id]['safety_template'][index].clear()
        user_data[user_id]['co_live_template'][index].clear()
        user_data[user_id]['group_liveness_template'][index].clear()

@app.route('/save-parity', methods=['POST'])
def save_parity():
    user_id = session.get('user_id')
    process_request(user_id, False)

    grid_layout = user_data[user_id]['grid_layout']
    game_graph = user_data[user_id]['game_graph']

    try:
        modified_parities = json.loads(request.form['modified_parities'])
    except ValueError:
        return jsonify(error=INPUT_ERROR), 400

    modified_parities = convert_modified_parities_to_dict(modified_parities)

    try:
        assert set(modified_parities.keys()) == set(user_data[user_id]['parity_modified_indices'])
    except AssertionError:
        return jsonify(error="Parity indices do not match"), 400

    game_graph.modify_priority(modified_parities)
    if user_data[user_id]['shield'] is not None:
        user_data[user_id]['shield'].game_graph = game_graph
    modify_grid_objectives(grid_layout, modified_parities)
    solve_modified_game(user_id)
    return jsonify(success="success", grid_layout=grid_layout)

def convert_list_of_list_to_dict(list_of_list):
    return {(i, j): list_of_list[i][j] for i, j in itertools.product(range(0, len(list)), repeat=2)}

def convert_modified_parities_to_dict(modified_parities):
    return {index: convert_list_of_list_to_dict(modified_parities[index]) for index in modified_parities}
    



@socketio.on('kill_bot')
def kill_bot_from_socket():
    user_id = get_user_from_socket_id(request.sid)
    process_request(user_id, True)

@socketio.on('save_grid')
def save_grid():
    user_id = session.get('user_id')
    process_request(user_id, False)
    grid_layout = user_data[user_id]['grid_layout']
    robot_position = user_data[user_id]['robot_position']
    save_path = os.path.join(os.path.dirname(__file__), f'{user_id}.json')
    with open(save_path, 'w') as f:
        json.dump({'grid_layout': grid_layout, 'robot_position': robot_position}, f)
    return send_file(save_path, as_attachment=True)


# @socketio.on('toggle_shield')
@app.route('/toggle-shield', methods=['POST'])
def toggle_shield():
    user_id = session.get('user_id')
    process_request(user_id, False)
    selected_option = request.form['use_shield']
    user_data[user_id]['apply_shield'] = selected_option == 'use_shield'
    print(f"apply shield: {user_data[user_id]['apply_shield']}")
    return jsonify(status="success")

@socketio.on('update_enforcing_parameter')
def update_parameter(data):
    
    user_id = session.get('user_id')
    process_request(user_id, False)
    shield = user_data[user_id]['shield']
    user_data[user_id]['enforcing_parameter'] = data['value']
    if shield is not None:
        shield.set_enforcement_parameter(user_data[user_id]['enforcing_parameter'])
    print(f"Updated enforcing parameter: {user_data[user_id]['enforcing_parameter']}")

@socketio.on('compute_game')
def compute_game():
    
    user_id = session.get('user_id')
    process_request(user_id, True)

    grid_layout = user_data[user_id]['grid_layout']
    robot_position = user_data[user_id]['robot_position']

    try:
        user_data[user_id]['game_graph'] = grid_to_game_graph(grid_layout, robot_position)
        socketio.emit('compute_game_success', room=user_data[user_id]['socket_id'])
        print("game graph created")
    except Exception as e:
        print("creating game error", e)
        send_error_message_to_user(user_id)
        return jsonify(error=str(e)), 400
    
    return jsonify(status="success")


@socketio.on('solve_game') # TODO: needs more for online computation
def solve_game():
    user_id = session.get('user_id')
    process_request(user_id, False)


    # grid_layout = user_data[user_id]['grid_layout']
    # robot_position = user_data[user_id]['robot_position']
    game_graph = user_data[user_id]['game_graph']
    enforcing_parameter = user_data[user_id]['enforcing_parameter']
    robot_position = user_data[user_id]['robot_position']

    safety_template = user_data[user_id]['safety_template']
    co_live_template = user_data[user_id]['co_live_template']
    group_liveness_template = user_data[user_id]['group_liveness_template']
    
    try:
        print("Solving game")

        # Make the game graph adversarial
        adversarial_game_graph = make_graph_adversarial(game_graph)
        print("game graph adversarial")

        for i in range(0, len(game_graph.priorities[robot_position])):
            print(f"Computing template for index {i}")
            _, safety_template_i, co_live_template_i, group_liveness_template_i = compute_template_in_two_player_graph(adversarial_game_graph, i)
            safety_template.append(safety_template_i)
            co_live_template.append(co_live_template_i)
            group_liveness_template.append(group_liveness_template_i)
        print("template computed")

        # Compute the value function and strategy for the robot
        if user_data[user_id]['mean_payoff_modified']:
            value_function, user_data[user_id]['mean_payoff_strategy'] = compute_value_function_mean_payoff(adversarial_game_graph)

            # Initialize the adversarial policy
            adversarial_policy = {state: {action: 0} for state in game_graph.get_player_vertices(2) for action in game_graph.get_actions(state)}

            for state in adversarial_game_graph.get_player_vertices(1):
                adversarial_policy[state] = {action: 0 for action in game_graph.get_actions(state)}
                for action, _ in game_graph.outgoing_edges[state]:
                    adversarial_policy[state][action] = 1
                adversarial_policy[state] = normalize_distribution(adversarial_policy[state])
            
            user_data[user_id]['random_player_strategy'] = adversarial_policy

    except Exception as e:
        print("solve game error", e)
        send_error_message_to_user(user_id)
        socketio.emit('solving_error', {'error': str(e)}, room=user_data[user_id]['socket_id'])
    

    user_data[user_id]['shield'] = Shield(game_graph, safety_template, co_live_template, group_liveness_template, enforcing_parameter)
    print("Shield created")
    socketio.emit('solving_success')
    print("Solving success")
    return jsonify(status="success")

def solve_modified_game(user_id):
    for index in user_data[user_id]['parity_modified_indices']:
        safety_template = user_data[user_id]['safety_template'][index]
        co_live_template = user_data[user_id]['co_live_template'][index]
        group_liveness_template = user_data[user_id]['group_liveness_template'][index]
        _, safety_template, co_live_template, group_liveness_template = compute_template_in_two_player_graph(user_data[user_id]['game_graph'], index, safety_template, co_live_template, group_liveness_template)
        user_data[user_id]['safety_template'][index] = safety_template
        user_data[user_id]['co_live_template'][index] = co_live_template
        user_data[user_id]['group_liveness_template'][index] = group_liveness_template

        if user_data[user_id]['shield'] is not None:
            user_data[user_id]['shield'].modify_template_at_index(index, safety_template, co_live_template, group_liveness_template)
        
        
    print("Modified game solved")
    

@socketio.on('prepare_robot')
def prepare_robot():

    user_id = session.get('user_id')
    process_request(user_id, True)

    game_graph = user_data[user_id]['game_graph']
    robot_position = user_data[user_id]['robot_position']
    enforcing_parameter = user_data[user_id]['enforcing_parameter']
    mean_payoff_strategy = user_data[user_id]['mean_payoff_strategy']
    adversarial_policy = user_data[user_id]['random_player_strategy']
    

    print("Preparing robot")
    user_data[user_id]['robot'] = Robot(game_graph, robot_position)
    user_data[user_id]['robot'].set_strategy_distribution_from_strategy(mean_payoff_strategy, enforcing_parameter, adversarial_policy)
    print("Robot prepared")
    socketio.emit('prepare_robot_success', room=user_data[user_id]['socket_id'])
    return jsonify(status="success")

@socketio.on('start_robot')
def start_robot():
    user_id = session.get('user_id')
    process_request(user_id, False)
    print(f"Starting robot with shield: {user_data[user_id]['apply_shield']}")
    # if not user_data[user_id]['kill_robot'].is_set():
    #     resume_bot(user_id)

    move_robot(user_id)

    # threading.Thread(target=move_robot, args=(user_id,), daemon=True).start()

@socketio.on('print_info')
def print_info(data):
    print(data)

def move_robot(user_id):

    print("Moving robot")

    grid_layout = user_data[user_id]['grid_layout']
    robot = user_data[user_id]['robot']
    shield = user_data[user_id]['shield']
    apply_shield = user_data[user_id]['apply_shield']
    old_position = user_data[user_id]['robot_position']
    user_data[user_id]['kill_robot'] = False

    # while not user_data[user_id]['kill_robot'].is_set():
    while not user_data[user_id]['kill_robot']:
        # print("Moving robot")
        time.sleep(0.3)  # Simulate a delay for movement
        next_state = None
        if apply_shield:
            action_distribution = robot.get_strategy_distribution(robot.robot_position)
            # next_state = shield.get_next_state(robot.robot_position, action_distribution)
            next_action, next_state = shield.sample_next_action_and_state(robot.robot_position, action_distribution)
        else:
            # next_state = robot.get_next_state_by_max()
            next_action, next_state = robot.sample_next_action_and_state()

        # print(f"Next action: {next_action}, Next state: {next_state}")
        user_data[user_id]['robot'].set_robot_position(next_state)
        user_data[user_id]['robot_position'] = next_state  # Update the global robot_position

        move_robot_on_grid_layout(grid_layout, old_position, next_state)
        old_position = next_state

        socketio.emit('update_position', {'position': user_data[user_id]['robot_position']}, room=user_data[user_id]['socket_id'])
        # print(f"Robot moved to {user_data[user_id]['robot_position']}")


# def fix_grid_layout(grid):
    
#     for (row, column) in iterate_over_cells(len(grid)//2):
#         cell = grid[row][column]
#         attributes = cell['attributes'].split(' ')
#         weights = cell['weight']
#         if weights[1] == 2:
#             attributes.remove('unmarked')
#             attributes.append('buechi')
#         attributes = ' '.join(attributes)
#         cell['attributes'] = attributes
#         grid[row][column] = cell
#     return grid


#========================================================================================
#
# Cleanup
#
#========================================================================================

def cleanup_inactive_users():
    """
    Background task to remove inactive users and their data.
    """

    while True:
        time.sleep(CLEANUP_INTERVAL)
        current_time = time.time()
        inactive_users = [user_id for user_id in user_data.keys() 
                          if current_time - user_data[user_id]['last_request'] > USER_TIMEOUT]
        for user_id in inactive_users:
            print(f"[info] Removing user due to inactivity: {user_id}")
            disconnect_user(user_id)

threading.Thread(target=cleanup_inactive_users, daemon=True).start()

#========================================================================================
#
# Main
#
#========================================================================================

if __name__ == '__main__':
    socketio.run(app, debug=True)
