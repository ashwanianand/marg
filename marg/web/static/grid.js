const socket = io();
let gridLayout = [];
let cells = [];
let editMode = "parity";


document.addEventListener('DOMContentLoaded', function() {

    reset_interface();


    document.getElementById('wall-density').addEventListener('input', async function(event) {
        document.getElementById('wall-density-value').textContent = event.target.value;
    });


    document.querySelector('#movement-options').addEventListener('click', async function(e) {
        console.log("changing shield mode: ");
        const selectedOption = document.querySelector('input[name="options"]:checked').value;
        console.log("Selected option: ", selectedOption);
        await fetch('/toggle-shield', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `use_shield=${selectedOption}`
        });
    });


    document.querySelector('#grid-modification-options').addEventListener('click', async function(e) {
        console.log("Changing grid modification mode: ");
        const selectedOption = document.querySelector('input[name="options"]:checked').value;
        console.log("Selected option: ", selectedOption);
        if (selectedOption === "modify_objective") {
            editMode = "parity";
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
        console.log("Edit mode: ", editMode);
    });

    document.getElementById('grid-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        const gridSize = document.getElementById('grid-size').value;
        const wallDensity = document.getElementById('wall-density').value;
        const response = await fetch('/generate-random-grid', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `grid_size=${gridSize}&wall_density=${wallDensity}`
        });

        const data = await response.json();

        socket.emit('compute_game');

        const gridContainer = document.getElementById('grid');
        gridLayout = data.grid_layout;

        gridContainer.innerHTML = '';
        cells = [];

        for (let rowIndex = 0; rowIndex < gridLayout.length; rowIndex++) {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'row';
            for (let colIndex = 0; colIndex < gridLayout[rowIndex].length; colIndex++) {
                const cellDiv = document.createElement('div');
                cellDiv.className = gridLayout[rowIndex][colIndex].attributes || 'empty';
                cellDiv.dataset.row = rowIndex;
                cellDiv.dataset.col = colIndex;
                
                if (gridLayout[rowIndex][colIndex].is_wall) {
                    cellDiv.addEventListener('click', toggleWall);
                }

                if (rowIndex % 2 === 1 && colIndex % 2 === 1) {
                    cellDiv.addEventListener('click', toggleCell);

                    const playerDiv = document.createElement('div');
                    if (gridLayout[rowIndex][colIndex].owner === 0) {
                        playerDiv.className = 'playerZero';
                    }
                    else {
                        playerDiv.className = 'playerOne'
                    }
                    playerDiv.addEventListener('click', toggleOwner);
                    cellDiv.appendChild(playerDiv);

                    const weightInp = document.createElement('input');
                    weightInp.type = 'text';
                    weightInp.className = 'weight-display';
                    weightInp.dataset.row = rowIndex;
                    weightInp.dataset.col = colIndex;
                    weightInp.value = gridLayout[rowIndex][colIndex].weight;
                    weightInp.addEventListener("change", async function(e) {
                        const cell = e.target;
                        const row = cell.parentElement.dataset.row;
                        const col = cell.parentElement.dataset.col;
                        if (editMode === "weight") {
                            e.stopPropagation();
                            gridLayout[row][col].weight = parseFloat(cell.value);
                            // Send the cell position and weight to the server
                            fetch('/modify-weight', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/x-www-form-urlencoded',
                                },
                                body: `row=${row}&column=${col}&weight=${cell.value}`
                            });
                        }
                    });
                    weightInp.addEventListener("keyup", async function(e) {
                        const cell = e.target;
                        const row = cell.parentElement.dataset.row;
                        const col = cell.parentElement.dataset.col;
                        if (editMode === "weight") {
                            e.stopPropagation();
                            gridLayout[row][col].weight = parseFloat(cell.value);
                            // Send the cell position and weight to the server
                            fetch('/modify-weight', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/x-www-form-urlencoded',
                                },
                                body: `row=${row}&column=${col}&weight=${cell.value}`
                            });
                        }
                    });
                    weightInp.addEventListener("paste", async function(e) {
                        const cell = e.target;
                        const row = cell.parentElement.dataset.row;
                        const col = cell.parentElement.dataset.col;
                        if (editMode === "weight") {
                            e.stopPropagation();
                            gridLayout[row][col].weight = parseFloat(cell.value);
                            // Send the cell position and weight to the server
                            fetch('/modify-weight', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/x-www-form-urlencoded',
                                },
                                body: `row=${row}&column=${col}&weight=${cell.value}`
                            });
                        }
                    });
                    weightInp.addEventListener("click", async function(e) {
                        const cell = e.target;
                        const row = cell.parentElement.dataset.row;
                        const col = cell.parentElement.dataset.col;
                        if (editMode === "weight") {
                            e.stopPropagation();
                            gridLayout[row][col].weight = parseFloat(cell.value);
                            // Send the cell position and weight to the server
                            fetch('/modify-weight', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/x-www-form-urlencoded',
                                },
                                body: `row=${row}&column=${col}&weight=${cell.value}`
                            });
                        }
                    });
                    cellDiv.appendChild(weightInp);
                    // weightDiv.className = 'weight-display';
                    // weightDiv.textContent = gridLayout[rowIndex][colIndex].weight;
                    // cellDiv.appendChild(weightDiv);
                }

                rowDiv.appendChild(cellDiv);
                cells.push(cellDiv);
            }
            gridContainer.appendChild(rowDiv);
            
        }

        // Disable the weight update initially
        setWeightInputs(false);

        // Show the enforcing-parameter-group
        document.getElementById('save-grid').style.display = 'inline';

        document.getElementById('grid-solver-section').style.display = 'block';



        document.getElementById('enforcing-parameter').addEventListener('input', async function(e) {
            document.getElementById('enforcing-parameter-value').textContent = e.target.value;
            const value = parseFloat(e.target.value);
            socket.emit('update_enforcing_parameter', { value });
        });


        document.getElementById('save-grid').addEventListener('click', function() {
            socket.emit('save_grid');
        });

        document.getElementById('solve-game').addEventListener('click', async function() {
            
            socket.emit('compute_game');


            socket.emit('solve_game');
            await new Promise((resolve, reject) => {
                socket.on('solving_success', (message) => {
                    console.log('Solved success message:', message);
                    console.log("Game solved successfully");

                    socket.emit('prepare_robot');

                    document.getElementById('display-templates').style.display = 'inline';
                    document.getElementById('enforcing-parameter-group').style.display = 'block';
                    document.getElementById('movement-options').style.display = 'flex';

                    socket.emit('start_robot');
                    resolve();
                });

                socket.on('solving_error', (error) => {
                    console.error('Solving error:', error);
                    reject(error);
                });
            });
        });


        // document.getElementById('display-templates').addEventListener('click', function() {
        //     document.getElementById('templates').style.display = 'block';
        // });

        socket.on('update_position', (data) => {
            cells.forEach(cell => cell.classList.remove('robot'));
            console.log("Robot position: ", data.position);
            const [row, col] = data.position;  // Assuming position is an array [row, col]
            console.log("Robot position: ", row, col);
            const cellDiv = cells.find(cell => cell.dataset.row == row && cell.dataset.col == col);
            if (cellDiv) {
                cellDiv.classList.add('robot');
            }
        });





    });
});

// async function toggleShield(e) {
//     const useShield = document.getElementById('use-shield').checked;

//     // Send the shield status to the server
//     await fetch('/toggle-shield', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/x-www-form-urlencoded',
//         },
//         body: `use_shield=${useShield}`
//     });
// }

async function toggleOwner(e) {
    e.stopPropagation(); // Prevents the event from propagating to parent elements

    const cell = e.target;
    const row = cell.parentElement.dataset.row;
    const col = cell.parentElement.dataset.col;
    // if (cell.classList.contains('playerZero')) {
    //     cell.classList.remove('playerZero');
    //     cell.classList.add('playerOne');
    // } else if (cell.classList.contains('playerOne')) {
    //     cell.classList.remove('playerOne');
    //     cell.classList.add('playerZero');
    // }
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

async function toggleCell(e) {
    const cell = e.target;
    const row = cell.dataset.row;
    const col = cell.dataset.col;
    if (editMode === "parity") {
    if (cell.classList.contains('unmarked')) {
        cell.classList.remove('unmarked');
        cell.classList.add('unsafe');
    } else if (cell.classList.contains('unsafe')) {
        cell.classList.remove('unsafe');
        cell.classList.add('buechi');
    } else if (cell.classList.contains('buechi')) {
        cell.classList.remove('buechi');
        cell.classList.add('cobuechi');
    } else if (cell.classList.contains('cobuechi')) {
        cell.classList.remove('cobuechi');
        cell.classList.add('unmarked');
    }


    // Send the cell position to the server
    await fetch('/toggle-cell-state', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `row=${row}&column=${col}`
    });} else if (editMode === "initial") {
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

async function setWeightInputs(isEnabled) {
    cells.forEach(cell => {
        const weightInp = cell.querySelector('.weight-display');
        if (weightInp) {
            weightInp.disabled = !isEnabled;
        }});
}

async function reset_interface() {
    
        // Get all the radio buttons in the group
        const radioButtons = document.querySelectorAll('#grid-modification-options input[type="radio"]');

        // Set the "Modify objective" radio button as checked
        radioButtons.forEach(button => {
            if (button.id === "modify-objective") {
                button.checked = true;
            } else {
                button.checked = false;
            }
        });

        // Set the "movement options" radio button as checked
        radioButtons.forEach(button => {
            if (button.id === "movement-options") {
                button.checked = true;
            } else {
                button.checked = false;
            }
        });

        // reset the wall density
        document.getElementById('wall-density').value = 0.3;
        document.getElementById('wall-density-value').textContent = 0.3;

        //reset the enforcing parameter
        document.getElementById('enforcing-parameter').value = 0.3;
        document.getElementById('enforcing-parameter-value').textContent = 0.3;

        // Send reset request to the server
        await fetch('/reset-program', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: ''
        });



}