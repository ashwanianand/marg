import tkinter as tk
from tkinter import messagebox, filedialog
import json
import sys
import random

class GridGenerator:
    def __init__(self, root, n, paths_file):
        self.root = root
        self.n = n
        self.paths_file = paths_file
        self.grid_size = 2 * n + 1
        self.cell_size = 90
        self.wall_size = self.cell_size // 10
        self.start_point = None
        self.end_point = None
        self.path = []
        self.paths = []
        self.row_coordinates = []
        self.col_coordinates = []
        self.coordinates()
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=(self.n * self.cell_size) + ((self.n + 1) * self.wall_size), height=(self.n * self.cell_size) + ((self.n + 1) * self.wall_size))
        self.canvas.pack()
        self.draw_grid()
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Key>", self.on_key_press)

        self.save_button = tk.Button(self.root, text="Save", command=self.save_grid)
        self.save_button.pack(side=tk.LEFT)

        self.load_button = tk.Button(self.root, text="Load", command=self.load_grid)
        self.load_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_grid)
        self.reset_button.pack(side=tk.LEFT)

        self.reset_cells_button = tk.Button(self.root, text="Reset Cells", command=self.reset_cells)
        self.reset_cells_button.pack(side=tk.LEFT)

        self.random_grid_button = tk.Button(self.root, text="Random Grid", command=self.random_grid)
        self.random_grid_button.pack(side=tk.LEFT)

        # self.generate_paths_button = tk.Button(self.root, text="Generate Paths", command=self.generate_paths)
        # self.generate_paths_button.pack(side=tk.LEFT)

    def random_grid(self):
        self.reset_grid()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i % 2 == 0 and j % 2 == 0:
                    self.canvas.itemconfig(self.grid[i][j], fill="black", outline="black")
                elif i % 2 == 0 or j % 2 == 0:
                    self.canvas.itemconfig(self.grid[i][j], fill="#f3f3f3", outline="#d8d8d8")
                    if random.random() < 0.3:  # 30% chance of being True
                        self.canvas.itemconfig(self.grid[i][j], fill="black", outline="black")
                else:
                    self.canvas.itemconfig(self.grid[i][j], fill="white")
                if i == 0 or j == 0 or i == self.grid_size - 1 or j == self.grid_size - 1:
                    self.canvas.itemconfig(self.grid[i][j], fill="black", outline="black")
        self.start_point = (random.randrange(1, self.grid_size, 2), random.randrange(1, self.grid_size, 2))
        self.end_point = (random.randrange(1, self.grid_size, 2), random.randrange(1, self.grid_size, 2))
        while self.end_point == self.start_point:
            self.end_point = (random.randrange(1, self.grid_size, 2), random.randrange(1, self.grid_size, 2))
        self.canvas.itemconfig(self.grid[self.start_point[0]][self.start_point[1]], fill="green", outline="green")
        self.canvas.itemconfig(self.grid[self.end_point[0]][self.end_point[1]], fill="red", outline="red")
        self.path = []
        self.paths = []
        self.canvas.delete("line")

    # def draw_grid(self):
    #     for i in range(self.grid_size):
    #         for j in range(self.grid_size):
    #             x1 = j * self.cell_size
    #             y1 = i * self.cell_size
    #             x2 = x1 + self.cell_size
    #             y2 = y1 + self.cell_size
    #             if i % 2 == 0 and j % 2 == 0:
    #                 self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black")
    #             elif i % 2 == 0 or j % 2 == 0:
    #                 self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="#f3f3f3")
    #             else:
    #                 self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

    def draw_grid(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.grid[i][j] is not None:
                    continue
                if i % 2 == 0 and j % 2 == 0:
                    x1 = j//2 * (self.cell_size + self.wall_size)
                    y1 = i//2 * (self.cell_size + self.wall_size)
                    x2 = x1 + self.wall_size
                    y2 = y1 + self.wall_size
                    self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", tags=('corner'))
                elif i % 2 == 0 and j % 2 != 0:
                    x1 = (((j-1)//2) * (self.cell_size + self.wall_size)) + self.wall_size
                    y1 = i//2 * (self.cell_size + self.wall_size)
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.wall_size
                    self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="#f3f3f3", tags=('wall'), outline="#d8d8d8")
                elif i % 2 != 0 and j % 2 == 0:
                    x1 = j//2 * (self.cell_size + self.wall_size)
                    y1 = (((i-1)//2) * (self.cell_size + self.wall_size)) + self.wall_size
                    x2 = x1 + self.wall_size
                    y2 = y1 + self.cell_size
                    self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="#f3f3f3", tags=('wall'), outline="#d8d8d8")
                else:
                    x1 = (((j-1)//2) * (self.cell_size + self.wall_size)) + self.wall_size
                    y1 = (((i-1)//2) * (self.cell_size + self.wall_size)) + self.wall_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.grid[i][j] = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", tags=('cell'), outline="#d8d8d8")
                
                if i == 0 or j == 0 or i == self.grid_size - 1 or j == self.grid_size - 1:
                    self.canvas.itemconfig(self.grid[i][j], fill="black", outline="black")

    def coordinates(self):
        for i in range(self.grid_size):
            value = 0
            if i % 2 == 0:
                value = i//2 * (self.cell_size + self.wall_size) + self.wall_size
            else:
                value = (((i+1)//2) * (self.cell_size + self.wall_size))
            self.row_coordinates.append(value)
            self.col_coordinates.append(value)

    def coordinate_to_cell(self, x, y):
        row = 0
        col = 0
        for i in range(len(self.row_coordinates)):
            if y >= self.row_coordinates[i]:
                row = i + 1
        for i in range(len(self.col_coordinates)):
            if x >= self.col_coordinates[i]:
                col = i + 1
        return row, col

    def cell_center(self, row, col):
        x = 0
        y = 0
        if row % 2 == 0:
            y = row//2 * (self.cell_size + self.wall_size) + self.wall_size//2
        else:
            y = ((row + 1)//2 * (self.cell_size + self.wall_size)) - self.cell_size//2

        if col % 2 == 0:
            x = col//2 * (self.cell_size + self.wall_size) + self.wall_size//2
        else:
            x = ((col + 1)//2 * (self.cell_size + self.wall_size)) - self.cell_size//2
        return x, y

    def toggle_wall(self, row, col):
        current_color = self.canvas.itemcget(self.grid[row][col], "fill")
        new_color = "black" if current_color == "#f3f3f3" else "#f3f3f3"
        new_outline = "black" if current_color == "#f3f3f3" else "#d8d8d8"
        self.canvas.itemconfig(self.grid[row][col], fill=new_color, outline=new_outline)

    def on_click(self, event):
        x, y = event.x, event.y
        row, col = self.coordinate_to_cell(x, y)
        if row % 2 == 0 and col % 2 == 0:
            return
        if row % 2 == 0 or col % 2 == 0:
            self.toggle_wall(row, col)
        else:
            if not self.start_point:
                self.start_point = (row, col)
                self.canvas.itemconfig(self.grid[row][col], fill="green", outline="green")
            elif not self.end_point:
                self.end_point = (row, col)
                self.canvas.itemconfig(self.grid[row][col], fill="red", outline="red")
        self.path = []
        self.paths = []
        self.canvas.delete("line")

    def on_key_press(self, event):
        if event.keysym == "Escape":
            self.reset_path()
        elif event.keysym == "BackSpace":
            self.backtrack_path()
        elif event.keysym == "Return":
            self.save_path()
        elif event.keysym in ["Up", "Down", "Left", "Right"]:
            self.navigate_path(event.keysym)
        elif event.keysym in ["R", "r"]:
            self.random_grid()

    def reset_grid(self):
        self.canvas.delete("all")
        self.start_point = None
        self.end_point = None
        self.path = []
        self.paths = []
        self.grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.draw_grid()

    def reset_cells(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if "cell" in self.canvas.itemcget(self.grid[i][j], "tags"):
                    self.grid[i][j] = None
        self.start_point = None
        self.end_point = None
        self.canvas.delete("dot")
        self.path = []
        self.paths = []
        self.draw_grid()        

    def save_grid(self):
        grid_data = {
            "grid": [[self.canvas.itemcget(self.grid[i][j], "fill") for j in range(self.grid_size)] for i in range(self.grid_size)],
            "paths": self.paths
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as file:
                json.dump(grid_data, file)

    def load_grid(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                grid_data = json.load(file)
            self.reset_grid()
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.canvas.itemconfig(self.grid[i][j], fill=grid_data["grid"][i][j])
                    if grid_data["grid"][i][j] == "green":
                        self.start_point = (i, j)
                    elif grid_data["grid"][i][j] == "red":
                        self.end_point = (i, j)
            self.paths = grid_data["paths"]

    # def generate_paths(self):
    #     if not self.start_point or not self.end_point:
    #         messagebox.showwarning("Warning", "Please set start and end points.")
    #         return
    #     self.root.unbind("<Button-1>")
    #     self.root.bind("<Key>", self.on_key_press)

    def create_circle(self, x, y, r): #center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return self.canvas.create_oval(x0, y0, x1, y1, fill="pink", tags=('dot'))

    def navigate_path(self, direction):
        if not self.path:
            if self.start_point:
                self.path.append(self.start_point)
                self.create_circle(*self.cell_center(*self.start_point), 5)
            else:
                messagebox.showwarning("Warning", "Please set the start point.")
                return
        current_row, current_col = self.path[-1]
        if direction == "Up":
            new_row, new_col = current_row - 2, current_col
        elif direction == "Down":
            new_row, new_col = current_row + 2, current_col
        elif direction == "Left":
            new_row, new_col = current_row, current_col - 2
        elif direction == "Right":
            new_row, new_col = current_row, current_col + 2
        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            if self.canvas.itemcget(self.grid[(current_row + new_row) // 2][(current_col + new_col) // 2], "fill") == "black":
                print("Wall detected")
                self.print_current_node(current_row, current_col)
                return
            self.print_current_node(new_row, new_col)
            self.path.append((new_row, new_col))
            current_cell_x, current_cell_y = self.cell_center(current_row, current_col)
            new_cell_x, new_cell_y = self.cell_center(new_row, new_col)
            # self.canvas.create_line(current_col * self.cell_size + self.cell_size // 2, current_row * self.cell_size + self.cell_size // 2,
                                    # new_col * self.cell_size + self.cell_size // 2, new_row * self.cell_size + self.cell_size // 2, fill="blue")
            self.canvas.create_line(current_cell_x, current_cell_y, new_cell_x, new_cell_y, fill="blue", tags="line")
            self.canvas.delete("dot")
            self.create_circle(new_cell_x, new_cell_y, 5)

    def backtrack_path(self):
        if self.path:
            self.path.pop()
            self.canvas.delete("line")
            for i, (row, col) in enumerate(self.path):
                if i > 0:
                    prev_row, prev_col = self.path[i - 1]
                    prev_cell_x, prev_cell_y = self.cell_center(prev_row, prev_col)
                    cell_x, cell_y = self.cell_center(row, col)

                    self.canvas.create_line(prev_cell_x, prev_cell_y, cell_x, cell_y, fill="blue", tags="line")

    def print_current_node(self, row, col):
        cell_color = self.canvas.itemcget(self.grid[row][col], "fill")[0]
        north_wall = 1 if row > 0 and self.canvas.itemcget(self.grid[row - 1][col], "fill") == "black" else 0
        east_wall = 1 if col < self.grid_size - 1 and self.canvas.itemcget(self.grid[row][col + 1], "fill") == "black" else 0
        west_wall = 1 if col > 0 and self.canvas.itemcget(self.grid[row][col - 1], "fill") == "black" else 0
        south_wall = 1 if row < self.grid_size - 1 and self.canvas.itemcget(self.grid[row + 1][col], "fill") == "black" else 0
        print(f"Cell: ({cell_color[0]}, {north_wall}, {east_wall}, {west_wall}, {south_wall})")
        self.root.clipboard_clear()
        self.root.clipboard_append(f"({cell_color[0]}, {north_wall}, {east_wall}, {west_wall}, {south_wall})")

    def save_path(self):
        path_data = []
        for i, (row, col) in enumerate(self.path):
            if i > 0:
                prev_row, prev_col = self.path[i - 1]
                if row < prev_row:
                    path_data.append("up")
                elif row > prev_row:
                    path_data.append("down")
                elif col < prev_col:
                    path_data.append("left")
                elif col > prev_col:
                    path_data.append("right")
            cell_color = self.canvas.itemcget(self.grid[row][col], "fill")[0]
            north_wall = 1 if row > 0 and self.canvas.itemcget(self.grid[row - 1][col], "fill") == "black" else 0
            east_wall = 1 if col < self.grid_size - 1 and self.canvas.itemcget(self.grid[row][col + 1], "fill") == "black" else 0
            west_wall = 1 if col > 0 and self.canvas.itemcget(self.grid[row][col - 1], "fill") == "black" else 0
            south_wall = 1 if row < self.grid_size - 1 and self.canvas.itemcget(self.grid[row + 1][col], "fill") == "black" else 0
            path_data.append((cell_color, north_wall, east_wall, west_wall, south_wall))
        self.paths.append(path_data)
        self.save_path_in_collection(path_data, row, col)
        self.reset_path()
    
    def save_path_in_collection(self, path_data, row, col):
        formatted_path = ""
        for i in range(len(path_data)):
            if isinstance(path_data[i], tuple):
                cell_color = path_data[i][0]
                if cell_color == "r":
                    cell_color = "red"
                elif cell_color == "g":
                    cell_color = "green"
                elif cell_color == "w":
                    cell_color = "white"
                formatted_path += f"({cell_color[0]}, {path_data[i][1]}, {path_data[i][2]}, {path_data[i][3]}, {path_data[i][4]})"
            else:
                formatted_path += f" -[{path_data[i]}]-> "
        with open(paths_file, "a") as file:
            if self.canvas.itemcget(self.grid[row][col], "fill") == "red":
                formatted_path += ", complete"
            else:
                formatted_path += ", incomplete"
            file.write(json.dumps(formatted_path) + "\n")

    def reset_path(self):
        self.path = []
        self.canvas.delete("dot")
        for item in self.canvas.find_all():
            if "line" in self.canvas.gettags(item):
                self.canvas.delete(item)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python traceGenerator4.py <grid_size> <paths_file>")
        sys.exit(1)
    n = int(sys.argv[1])
    paths_file = sys.argv[2]
    root = tk.Tk()
    root.title("Grid Generator")
    app = GridGenerator(root, n, paths_file)
    root.mainloop()