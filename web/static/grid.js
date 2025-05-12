// This file contains the client-side code for MARG.



/**
 * Declare the constants and global variables
 */
const socket = io.connect();
const initialWallDensity = 0.3;
const initialEnforcingParameter = 0.05;
const heartbeatInterval = 30000;

let gridLayout = [];
let cells = [];
let editMode = "owner";
let modifiedParities = {};
let meanPayoffWeights = [];
let heartbeat;


//=============================================================================
// Initial setup
//=============================================================================

/**
 * When the document is loaded, show the initial buttons and form to generate the grid
 */
document.addEventListener('DOMContentLoaded', async function() {

    // Start the heartbeat
    // startHeartbeat();

    // Reset the interface
    resetInterface();
    // askServerToReset(); // DELETE THIS IF USER DATA SHOULD BE KEPT

    // Prepare the buttons with event listeners
    prepareButtons();

    // Check if the user data exists on the server
    const userExists = await fetch('/user-exists', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
    });

    const existsCheckResponse  = await userExists.json()

    // If the user data exists, then retrieve the grid. 
    if (existsCheckResponse.exists === "true") {
        gridLayout = existsCheckResponse.grid_layout
        console.log("[Info] User exists. Retrieving old grid.");
        if (gridLayout.length === 0) {
            console.log("[Info] No old grid found");
        } else {
            processGridLayout();
        }
    } else {
        console.log("[Info] User did not exist. No grid could be retrieved.");
    }
    

    // Prepare the grid form that sends the server the grid size and the wall density
    document.getElementById('grid-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        const gridSize = document.getElementById('grid-size').value;
        const wallDensity = document.getElementById('wall-density').value;
        
        
        // Send the grid size and wall density to the server
        // and await the response
        const response = await fetch('/generate-random-grid', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `grid_size=${gridSize}&wall_density=${wallDensity}`
        });

        const data = await response.json();

        // Update the grid layout and display the grid
        gridLayout = data.grid_layout;
        processGridLayout();
        
    });

    // Listen for the server's response to move the robot on the grid
    socket.on('update_position', (data) => {
        cells.forEach(cell => cell.classList.remove('robot'));
        const [row, col] = data.position;  // Assuming position is an array [row, col]
        console.log("[Update] New robot position: ", row, col);
        const cellDiv = cells.find(cell => cell.dataset.row == row && cell.dataset.col == col);
        if (cellDiv) {
            cellDiv.classList.add('robot');
        }
    });
});

/**
 * This function sends heartbeat signal every heartbeatInterval to the server
 */
function startHeartbeat() {
    heartbeat = setInterval(() => {
        fetch('/heartbeat', {
            method: 'POST'
        });
    }, heartbeatInterval);
}

/**
 * The function processes and displays the current gridLayout
 */
function processGridLayout() {
    setInitialWeights(); // Keep a copy of the meanpayoff weights to be modified
    displayGrid();
    setWeightInputs(false); // Disable the weight update initially

    // Preemptively compute the game in case the grid is not modified
    socket.emit('compute_game');
    
    // When the game is computed internally, show the solving section
    socket.on('compute_game_success', () => {
        console.log("[Info] Game computed on the server");
        
        // Show the solving section
        document.getElementById('grid-solver-section').style.display = 'block';

        // Show the toolbar
        prepareAndShowToolbar();
        // makeDraggable(document.getElementById("toolbar"));
    });
}

/**
 * Functions to reset the interface, and reset the client data on the server
 */
async function resetInterface() {

    console.log("[info] Resetting interface")

    document.getElementById('grid-solver-section').style.display = 'none';
    document.querySelector('.toolbar').style.display = 'none';
    document.getElementById('movement-options').style.display = 'none';
    document.getElementById('enforcing-parameter-group').style.display = 'none';
    document.getElementById('grid').innerHTML = '';
    
    // Get all the radio buttons in the grid modification section
    const gridModificationRadioButtons = document.querySelectorAll('#grid-modification-options input[type="radio"]');

    // Set the "Modify objective" radio button as checked
    gridModificationRadioButtons.forEach(button => {
        if (button.id === "modify-ownership") {
            button.checked = true;
        } else {
            button.checked = false;
        }
    });


    // Get all the radio buttons in the movement options section
    const movementRadioButtons = document.querySelectorAll('#movement-options input[type="radio"]');

    // Set the "movement options" radio button as checked
    movementRadioButtons.forEach(button => {
        if (button.id === "no-shield") {
            button.checked = true;
        } else {
            button.checked = false;
        }
    });

    // reset the solve game button
    solvedGame();

    // reset the wall density
    document.getElementById('wall-density').value = initialWallDensity;
    document.getElementById('wall-density-value').textContent = initialWallDensity;

    //reset the enforcing parameter
    document.getElementById('enforcing-parameter').value = initialEnforcingParameter;
    document.getElementById('enforcing-parameter-value').textContent = initialEnforcingParameter;
}

/**
 * The function to keep a copy of the mean payoff weights.
 * This is used to send the server the modified weights.
 */
function setInitialWeights() {
    // Create a 0 filled matrix of the same size as the grid layout
    meanPayoffWeights = Array(gridLayout.length).fill(0).map(() => Array(gridLayout[0].length).fill(0));
    
    // Copy the weights from the grid layout
    gridLayout.forEach((row, rowIndex) => {
        row.forEach((cell, colIndex) => {
            meanPayoffWeights[rowIndex][colIndex] = cell.weight[0];
        });
    });
}



/**
 * The function to prepare all the buttons on the page and their event listeners
 */
function prepareButtons() {

    // When the wall density is changed, update the wall density value displayed on the screen
    document.getElementById('wall-density').addEventListener('input', async function(event) {
        document.getElementById('wall-density-value').textContent = event.target.value;
    });

    // When the reset button is clicked, reset the interface and ask the server to forget all user data
    document.getElementById('reset-session').addEventListener('click', () => {
        resetInterface()
        askServerToReset();
    });

    // This group of radio buttons is for the grid modification options, i.e. modify ownership, modify reward, modify initial state
    document.getElementById('grid-modification-options').querySelectorAll('input[name="options"]').forEach(radio => {radio.addEventListener('click', async function(e) {
        const gridModificationOptions = document.getElementById('grid-modification-options');
        const selectedOption = gridModificationOptions.querySelector('input[name="options"]:checked').value;
        console.log("[Info] Changing grid modification mode to: ", selectedOption);
        socket.emit('kill_bot')
        if (selectedOption === "modify_ownership") {
            editMode = "owner";
            setWeightInputs(false);
        }
        else if (selectedOption === "modify_reward") {
            editMode = "weight";
            setWeightInputs(true);
        }
        else if (selectedOption === "modify_initial_state") {
            editMode = "initial";
            cells.forEach(cell => cell.classList.remove('robot'));
            setWeightInputs(false);
        }
        console.log("[Update] Changed grid modification mode to: ", editMode);
        });
    });

    // When the save grid button is clicked, send the save grid request to the server
    document.getElementById('save-grid').addEventListener('click', function() {
        socket.emit('save_grid');
    });

    // When the solve game button is clicked, send the solve game request to the server via the solveGame function
    document.getElementById('solve-game').addEventListener('click', () => solveGame());


    // This group of radio buttons is for the robot movement options, i.e. with or without shield
    document.getElementById('movement-options').querySelectorAll('input[name="options"]').forEach(radio => {
        radio.addEventListener('click', () => toggleShield());
    });

    // When the enforcing parameter is changed,  call the function updateEnforcementParameter, 
    // that updates the enforcing parameter value displayed on the screen and sends the new value to the server
    document.getElementById('enforcing-parameter').addEventListener('input', (e) => updateEnforcementParameter(e));

}


/**
 * Function to ask the server to reset user data
 */
function askServerToReset() {
    // Send reset request to the server to reset the client data
    fetch('/reset-program', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: ''
    });
}

/**
 * Function to display the grid on the screen.
 * The function creates the grid from the gridLayout
 */
function displayGrid() {
    const gridContainer = document.getElementById('grid');
    gridContainer.innerHTML = '';
    cells = [];

    for (let rowIndex = 0; rowIndex < gridLayout.length; rowIndex++) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'row';
        for (let colIndex = 0; colIndex < gridLayout[rowIndex].length; colIndex++) {

            // Create a cell div
            const cellDiv = document.createElement('div');
            cellDiv.className = gridLayout[rowIndex][colIndex].attributes || 'empty';
            cellDiv.dataset.row = rowIndex;
            cellDiv.dataset.col = colIndex;
            
            // If the cell is a wall, add an event listener to toggle the wall
            if (gridLayout[rowIndex][colIndex].is_wall) {
                cellDiv.addEventListener('click', toggleWall);
            }

            // If the cell is an actual cell in the grid (not wall or corner)
            if (rowIndex % 2 === 1 && colIndex % 2 === 1) {

                // Add event listener to the cell to add the robot when needed
                cellDiv.addEventListener('click', (e) => placeRobot(e));

                // Create a cell info div to contain the ownership and objectives information
                const cellInfoDiv = document.createElement('div');
                cellInfoDiv.id = 'cell-info';
                
                // Create a player div to show the owner of the cell in the info div
                const playerDiv = document.createElement('div');
                playerDiv.id = 'owner';
                if (gridLayout[rowIndex][colIndex].owner === 0) {
                    playerDiv.className = 'playerZero';
                }
                else {
                    playerDiv.className = 'playerOne'
                }
                playerDiv.addEventListener('click', toggleOwner);
                cellInfoDiv.appendChild(playerDiv);

                // Create an objectives div to show the objectives of the cell in the info div
                const objectivesDiv = document.createElement('div');
                objectivesDiv.id = 'objectives';

                // For every parity objective in the grid, add a div to the objectives div
                // showing the current color of the cell in the objective
                gridLayout[rowIndex][colIndex].weight.forEach((priority, index) => {
                    if (index !== 0) {
                        const priorityDiv = document.createElement('div');
                        priorityDiv.dataset.parityIndex = index;
                        if (priority === 2) {
                            priorityDiv.className = 'buechi';
                        } else if (priority === 3) {
                            priorityDiv.className = 'cobuechi';
                        } else {
                            priorityDiv.className = 'unmarked';
                        }

                        objectivesDiv.appendChild(priorityDiv);
                    }
                
                });
                cellInfoDiv.appendChild(objectivesDiv);
                cellDiv.appendChild(cellInfoDiv);

                // Create a weight input for the cell, showing the mean payoff weight of the cell
                const weightInp = document.createElement('input');
                weightInp.type = 'text';
                weightInp.className = 'weight-display';
                weightInp.dataset.row = rowIndex;
                weightInp.dataset.col = colIndex;
                weightInp.dataset.bsToggle = 'tooltip';
                weightInp.dataset.trigger = "manual";
                weightInp.dataset.placement = 'top';
                weightInp.title = "Please enter an integer.";
                weightInp.value = gridLayout[rowIndex][colIndex].weight[0];

                // Add event listeners to the weight input to modify the weight of the cell
                // Add the following for 'change', 'keyup', 'paste', 'click' iff needed
                weightInp.addEventListener("input", (e) => {
                    e.stopPropagation();
                    modifyWeight(e);
                });
                cellDiv.appendChild(weightInp);
            }
            rowDiv.appendChild(cellDiv);
            cells.push(cellDiv);
        }
        gridContainer.appendChild(rowDiv);
    }

    // Set the weight inputs to be disabled initially, so it can not be modified
    setWeightInputs(false);

}


//=============================================================================
// Modify and Solve Game Section
//=============================================================================

/**
 * Function to solve the game, activated when the solve game button is clicked.
 * 
 * The function first sends the modified mean payoff weights to the server.
 * Then, it sends a request to the server to solve the game.
 * When the game is solved, the function sends a request to the server to prepare the robot.
 * When the robot is prepared, the function shows the enforcing parameter group and movement options.
 * Finally, the function sends a request to the server to start the robot.
 */
async function solveGame() {
    processingGame();

    // Send the modified mean payoff weights to the server
    const response = await fetch('/modify-weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `weights=${JSON.stringify(meanPayoffWeights)}`
    });
    const data = await response.json();
    console.log("[Info] Modified weights sent to server", data);

    // If the server successfully receives the modified weights, compute and solve the game
    if (data.status === "success") {
        socket.emit('compute_game');
        solvingGame();
        socket.emit('solve_game');

        await new Promise((resolve, reject) => {
            // If the game is solved internally, prepare the robot and show the enforcing parameter group and movement options, and start moving the robot
            socket.on('solving_success', (message) => {
                console.log('[Info] Game solved internally');

                socket.emit('prepare_robot');

                document.getElementById('enforcing-parameter-group').style.display = 'block';
                document.getElementById('movement-options').style.display = 'flex';
                
                solvedGame();
                socket.on('prepare_robot_success', () => {
                    console.log('[Info] Robot prepared successfully');
                    socket.emit('start_robot');
                });
                resolve();
            });


            socket.on('solving_error', (error) => {
                console.error('[Error] Solving error:', error);
                solvingError();
                reject(error);
            });
        });
    } 
}

/**
 * The function to toggle the wall of a cell. 
 */
async function toggleWall(e) {
    const cell = e.target;
    const row = cell.dataset.row;
    const col = cell.dataset.col;
    cell.classList.toggle('wall'); // Toggle wall class
    gridLayout[row][col].is_wall = !gridLayout[row][col].is_wall;

    // Send the cell position and color to the server
    await fetch('/toggle-wall', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `row=${row}&column=${col}`
    });
}


/**
 * The function to toggle the owner of a cell.
 * First, the function changes the owner of the cell in the displayed grid.
 * Then, the function sends the new owner of the cell to the server.
 * 
 * Activated when the owner div of a cell is clicked. Checks if the edit mode is "owner".
 * @param {Event} e 
 */
async function toggleOwner(e) {
    e.stopPropagation(); // Prevents the event from propagating to parent elements

    const cell = e.target;
    const row = cell.parentElement.dataset.row;
    const col = cell.parentElement.dataset.col;

    if (editMode === "owner") {
        cell.classList.toggle('playerZero');
        cell.classList.toggle('playerOne');

        gridLayout[row][col].owner = gridLayout[row][col].owner === 0 ? 1 : 0;

        // Send the cell position to the server
        await fetch('/toggle-owner', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `row=${row}&column=${col}`
        });
    }
}


/**
 * The function to modify the weight of a cell.
 * If the edit mode is "weight", the function updates the weight of the cell in the grid layout, 
 * and in the mean payoff weights copy.
 * 
 * Activated when the weight input of a cell is changed. Checks if the edit mode is "weight".
 * @param {event} e 
 */
async function modifyWeight(e) {
    const cell = e.target;
    const row = cell.parentElement.dataset.row;
    const col = cell.parentElement.dataset.col;
    const value = cell.value;
    if (editMode === "weight") {
        if (/^-?\d+$/.test(value) || value === "") {
            cell.classList.remove('is-invalid');
            $(cell).tooltip('hide');

            console.log("MP value was:", meanPayoffWeights[row][col]);
            gridLayout[row][col].weight = parseFloat(value);
            meanPayoffWeights[row][col] = parseFloat(value);
            console.log("MP value became:", meanPayoffWeights[row][col]);
        } else {
            cell.classList.add('is-invalid')
            $(cell).tooltip('show')
        }
    }
}


/**
 * The function to place the robot on the grid.
 * First, the function removes the robot from all cells.
 * Then, the function places the robot on the clicked cell.
 * Finally, the function sends the new robot position to the server.
 * 
 * Activated when a cell is clicked. Checks if the edit mode is "initial".
 * @param {Event} e
 */
async function placeRobot(e) {
    const cell = e.target;
    const row = cell.dataset.row;
    const col = cell.dataset.col;
    if (editMode === "initial") {
        cells.forEach(cell => cell.classList.remove('robot'));
        cell.classList.toggle('robot');
        // Send the cell position to the server
        await fetch('/toggle-initial-state', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `row=${row}&column=${col}`
        });
    }
}


/**
 * The function makes the weight inputs enabled or disabled.
 * Activated when the edit mode is changed to "weight".
 * @param {Boolean} isEnabled 
 */
async function setWeightInputs(isEnabled) {
    cells.forEach(cell => {
        const weightInp = cell.querySelector('.weight-display');
        if (weightInp) {
            weightInp.disabled = !isEnabled;
        }});
}


//=============================================================================
// Robot Movement Section
//=============================================================================

/**
 * The function to update the enforcing parameter value displayed on the screen and send the new value to the server
 * Activated when the enforcing parameter is changed.
 * 
 * @param {event} e 
 */
function updateEnforcementParameter(e) {
    document.getElementById('enforcing-parameter-value').textContent = e.target.value;
    const value = parseFloat(e.target.value);
    socket.emit('update_enforcing_parameter', { value });
}


/**
 * The function to toggle the shield mode.
 * The function sends the new shield mode to the server.
 * Activated when the shield mode radio buttons are clicked.
 */
async function toggleShield() {
    console.log("[Info] Changing shield mode");
    const movementOptions = document.getElementById('movement-options')
    const selectedOption = movementOptions.querySelector('input[name="options"]:checked').value;
    console.log("[Update] Changed shield mode to: ", selectedOption);
    await fetch('/toggle-shield', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `use_shield=${selectedOption}`
    });
}



//=============================================================================
// Toolbar 
//=============================================================================

/**
 * The function to prepare and show the toolbar to add, delete, and modify parities.
 * Displayed only after the grid is generated on the display.
 */
function prepareAndShowToolbar() {
    const toolbar = document.getElementById("toolbar");

    // Show the parity objectives in the toolbar
    showParityObjectivesInToolbar();

    // Remove existing event listeners to avoid adding them multiple times
    const addParityButton = document.getElementById('toolbar-add-parity');
    const newAddParityButton = addParityButton.cloneNode(true);
    addParityButton.parentNode.replaceChild(newAddParityButton, addParityButton);

    const deleteParityButton = document.getElementById('toolbar-delete-parity');
    const newDeleteParityButton = deleteParityButton.cloneNode(true);
    deleteParityButton.parentNode.replaceChild(newDeleteParityButton, deleteParityButton);

    // Add event listeners to the toolbar buttons
    document.getElementById('toolbar-add-parity').addEventListener('click', () => addParity());
    document.getElementById('toolbar-delete-parity').addEventListener('click', () => deleteParity());
    document.getElementById('toolbar-modification-options').querySelectorAll('input[name="options"]').forEach(radio => {
        radio.addEventListener('change', modifyOrSave);
    });
    
    // Show the toolbar
    toolbar.style.display = (toolbar.style.display === "none" || toolbar.style.display === "") ? "flex" : "flex";
}


/**
 * The function to show the parity objectives in the toolbar.
 * The function creates a div for each parity objective in the gridLayout[1][1]['weight'] 
 * (i.e. all the weights except the first one).
 */
function showParityObjectivesInToolbar() {
    const toolbarObjectivesDiv = document.getElementById('toolbar-objectives');// Clear existing toolbar content
    
    toolbarObjectivesDiv.innerHTML = '';

    // Create divs for each weight in gridLayout[1][1]['weight']
    gridLayout[1][1]['weight'].forEach((weight, index) => {
        if (index !== 0) {
            const weightDiv = document.createElement('span');
            weightDiv.addEventListener('click', () => toggleParitySelection(weightDiv));
            weightDiv.className = 'toolbar-parity';
            weightDiv.dataset.toolbarParityIndex = index;
            weightDiv.textContent = index;
            toolbarObjectivesDiv.appendChild(weightDiv);
        }
    });
    
}

/**
 * The function to toggle the selection of a parity in the toolbar.
 * Activated when a parity div in the toolbar is clicked.
 * @param {Element} weightDiv 
 */
function toggleParitySelection(weightDiv) {
    weightDiv.classList.toggle('selected');
}

/**
 * The function to add a parity.
 * The function sends a request to the server to add a parity.
 * The function checks if the random checkbox is checked, and sends the random parameter to the server.
 * If the random checkbox is not checked, it will generate an empty parity objective.
 * Else, the server generates a random parity objective.
 * The function updates the objectives in each cell, and shows the objectives in the toolbar.
 */
async function addParity() {
    const toolbarCheckbox = document.getElementById('toolbar-random-checkbox');
    const toolbarCheckboxState = toolbarCheckbox.checked;
    let response = null;
    console.log("[Info] Adding parity with random: ", toolbarCheckboxState);
    if (toolbarCheckboxState) {
        // Send the request to the server to add a random parity
        response = await fetch('/add-parity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'random=true'
        });
    } else {
        // Send the request to the server to add a parity
        response = await fetch('/add-parity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'random=false'
        });
    }

    const data = await response.json();
    
    gridLayout = data.grid_layout;
    updateParityObjectivesInEachCell();
    showParityObjectivesInToolbar();
}


/**
 * The function to delete a parity.
 * The function gets the selected parities divs in the toolbar, 
 * and sends the indices of the selected parities to the server to delete.
 */
async function deleteParity() {
    const selectedParities = document.querySelectorAll('.toolbar-parity.selected');
    const indices = [];
    selectedParities.forEach(parity => indices.push(parity.dataset.toolbarParityIndex));
    console.log("[Info] Deleting parities: ", indices);
    // Send the request to the server to delete a parity
    const response = await fetch('/delete-parity', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: 'selected_indices=' + JSON.stringify(indices)
    });

    const data = await response.json();
    gridLayout = data.grid_layout;
    updateParityObjectivesInEachCell();
    showParityObjectivesInToolbar();
}

/**
 * The function to update the parity objectives in each cell.
 * The function clears the objectives div in each cell, and adds a div for each parity objective in the cell.
 * The function is called after a parity is added or deleted.
 */
function updateParityObjectivesInEachCell() {

    for (let rowIndex = 0; rowIndex < gridLayout.length; rowIndex++) {
        for (let colIndex = 0; colIndex < gridLayout[rowIndex].length; colIndex++) {

            // If the cell is an actual cell in the grid (not wall or corner)
            if (rowIndex % 2 === 1 && colIndex % 2 === 1) {

                // Get the cell div
                const cellDiv = document.querySelector(`[data-row="${rowIndex}"][data-col="${colIndex}"]`);

                // Get the cell info div to contain the ownership and objectives information
                const cellInfoDiv = cellDiv.querySelector('#cell-info');
                
                // Get the objectives div to show the objectives of the cell in the info div
                const objectivesDiv = cellInfoDiv.querySelector('#objectives');

                // Clear the objectives div
                objectivesDiv.innerHTML = '';

                // For every parity objective in the grid, add a div to the objectives div
                // showing the current color of the cell in the objective
                gridLayout[rowIndex][colIndex].weight.forEach((priority, index) => {
                    if (index !== 0) {
                        const priorityDiv = document.createElement('div');
                        priorityDiv.dataset.parityIndex = index;
                        if (priority === 2) {
                            priorityDiv.className = 'buechi';
                        } else if (priority === 3) {
                            priorityDiv.className = 'cobuechi';
                        } else {
                            priorityDiv.className = 'unmarked';
                        }

                        objectivesDiv.appendChild(priorityDiv);
                    }
                
                });
            }
        }
    }
}


/**
 * The function adds the functionality to the modify/save button in the toolbar.
 * The function gets the selected option in the toolbar, and the selected parities in the toolbar.
 * 
 * If the selected option is "modify_parity", the function makes the selected parities modifiable.
 * Modifiable parities can be clicked to change the parity objective.
 * The function sends the indices of the selected parities to the server,
 *  to stop the robot from following the parities that are currently being modified.
 * 
 * Else, the function saves the modified parity objectives.
 * The function sends the modified parity objectives to the server
 * to save the parity objectives in the grid layout.
 * The function removes the modifiable class from the modifiable parities, hence the objectives can not be modified further.
 */
function modifyOrSave() {
    const toolbarButton = document.getElementById('toolbar-modification-options');
    const selectedOption = toolbarButton.querySelector('input[name="options"]:checked').value;


    const selectedParities = document.querySelectorAll('.toolbar-parity.selected');
    const indices = [];
    selectedParities.forEach(parity => indices.push(parity.dataset.toolbarParityIndex));
    
    if (selectedOption === "modify_parity") {
        indices.forEach(index => {
            modifiedParities[index] = Array(gridLayout.length).fill(0).map(() => Array(gridLayout[0].length).fill(0));
            const parityDivs = document.querySelectorAll(`[data-parity-index="${index}"]`);
            parityDivs.forEach(parityDiv => makeModifiable(parityDiv));
        });
        // Ask the server to stop the robot from following the modifiable parities
        fetch('/modify-parity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'indices=' + JSON.stringify(indices)
        });
    }
    else {
        const modifiableParities = document.querySelectorAll('.modifiable');
        modifiableParities.forEach(parity => {
            parity.classList.remove('modifiable');
            parity.removeEventListener('click', toggleParity);
        });

        // Send the request to the server to save the parity conditions
        fetch('/save-parity', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'modified_parities=' + JSON.stringify(modifiedParities)
        });
    }
}

/**
 * The function makes the given parity div modifiable.
 * The function adds the modifiable class to the parity div, and adds an event listener to toggle the parity.
 * @param {Element} priorityDiv 
 */
function makeModifiable(priorityDiv) {
    priorityDiv.classList.add('modifiable');
    priorityDiv.addEventListener('click', (e) => toggleParity(e));
}


/**
 * The function to toggle the parity objective of a cell.
 * The function changes the color of the parity div in the cell.
 * The function cycles through the colors: unmarked, buechi, cobuechi.
 * The function updates the parity objective in the grid layout and the modified parities.
 * 
 * Activated when a parity div with modifiable property in a cell is clicked.
 * @param {Event} e 
 */
function toggleParity(e) {
        const objDiv = e.target.parentElement;
        const cellInfo = objDiv.parentElement;
        const cell = cellInfo.parentElement;
        const row = cell.dataset.row;
        const col = cell.dataset.col;
        const index = e.target.dataset.parityIndex;
        if (e.target.classList.contains('unmarked')) {
            e.target.classList.remove('unmarked');
            e.target.classList.add('buechi');
        } else if (e.target.classList.contains('buechi')) {
            e.target.classList.remove('buechi');
            e.target.classList.add('cobuechi');
        } else if (e.target.classList.contains('cobuechi')) {
            e.target.classList.remove('cobuechi');
            e.target.classList.add('unmarked');
        }
        gridLayout[row][col].weight[index] = ((gridLayout[row][col].weight[index] + 1) % 3) + 1;
        modifiedParities[index][row][col] = gridLayout[row][col].weight[index];
}

// Make the toolbar draggable
// function makeDraggable(element) {
//     let offsetX = 0, offsetY = 0, mouseX = 0, mouseY = 0;
//     element.onmousedown = (e) => {
//         e.preventDefault();
//         mouseX = e.clientX;
//         mouseY = e.clientY;
//         document.onmousemove = dragElement;
//         document.onmouseup = stopDrag;
//     };

//     function dragElement(e) {
//         offsetX = mouseX - e.clientX;
//         offsetY = mouseY - e.clientY;
//         mouseX = e.clientX;
//         mouseY = e.clientY;
//         element.style.left = (element.offsetLeft - offsetX) + "px";
//         element.style.top = (element.offsetTop - offsetY) + "px";
//     }

//     function stopDrag() {
//         document.onmousemove = null;
//         document.onmouseup = null;
//     }
// }




//=============================================================================
// Spinner functions
//=============================================================================


/**
 * The function starts the spinner when the game is being computed from the layout by the server. 
 */
function processingGame() {
    const button = document.getElementById('solve-game');
    button.disabled = true;
    button.innerHTML = '<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span> Processing game...';
}

/**
 * When the server moves on to solving the game, the function starts the spinner in the solve-game button.
 */
function solvingGame() {
    const button = document.getElementById('solve-game');
    button.disabled = true;
    button.innerHTML = '<span class="spinner-grow spinner-grow-sm" role="status" aria-hidden="true"></span> Solving game...';
}

/**
 * When the game is solved, the function stops the spinner in the solve-game button.
 */
function solvedGame() {
    const button = document.getElementById('solve-game');
    button.disabled = false;
    button.innerHTML = 'Solve Game';
}

/**
 * When there is an error in solving the game, the function stops the spinner in the solve-game button.
 * The solve-game button is disabled, and the text is changed to "Solving error. Retry."
 */
function solvingError() {
    const button = document.getElementById('solve-game');
    button.disabled = true;
    button.innerHTML = 'Solving error. Retry.';
}



//=============================================================================
// Modals
//=============================================================================


/**
 * Create a popup when the server disconnects the client
 */
socket.on('user_disconnected', () => {
    const disconnectModal = document.createElement('div');
    disconnectModal.className = 'modal fade';
    disconnectModal.id = 'staticBackdrop';
    disconnectModal.dataset.backdrop = 'static';
    disconnectModal.dataset.keyboard = 'false';
    disconnectModal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"> 
                    <i class="bi bi-exclamation-octagon"></i> Disconnected</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <p>You have been disconnected due to inactivity. Please reload the page before starting again.</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" onclick="location.reload();">Reload Page</button>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(disconnectModal);
    $(disconnectModal).modal('show');
});


/**
 * Create a modal informing user of an error
 */

socket.on('display-error', () => {
    const errorModal = document.createElement('div');
    errorModal.className = 'modal fade';
    errorModal.id = 'staticBackdrop';
    errorModal.dataset.backdrop = 'static';
    errorModal.dataset.keyboard = 'false';
    errorModal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered modal-dialog-scrollable">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title"> 
                    <i class="bi bi-exclamation-octagon"></i> Error</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <p>There was some error. Please reload the page. </p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" onclick="location.reload();">Reload Page</button>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(errorModal);
    $(errorModal).modal('show');

})