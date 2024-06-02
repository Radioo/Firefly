import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from firefly import FireflyAlgorithm

matplotlib.use('TkAgg')

class LinearProgrammingVisualizer:
    def __init__(self, constraints, objective):
        self.constraints = constraints
        self.objective = objective
        self.max_x, self.max_y = self.calculate_max_axis()

    def calculate_max_axis(self):
        max_x = max_y = 0
        for (a, b, _, rhs) in self.constraints:
            max_x = max(max_x, rhs / a if a != 0 else 0)
            max_y = max(max_y, rhs / b if b != 0 else 0)
        return max_x, max_y

    def plot_constraints(self):
        plt.figure(figsize=(8, 8))
        x = np.linspace(0, self.max_x, 400)

        for (a, b, sign, rhs) in self.constraints:
            # sign can be either '<=', '>=', or '='
            if b == 0:
                y = np.where(a * x <= rhs, np.inf, -np.inf)
            else:
                y = (rhs - a * x) / b
            if sign == '<=':
                plt.fill_between(x, y, self.max_y, alpha=0.3, where=(y <= self.max_y))
            elif sign == '>=':
                plt.fill_between(x, y, 0, alpha=0.3, where=(y >= 0))
            elif sign == '=':
                y1 = np.where(b != 0, (rhs - a * x) / b, np.inf)
                y2 = np.where(b != 0, (rhs - a * x) / b, np.inf)
                plt.fill_between(x, y1, y2, alpha=0.3)

    def visualize(self):
        self.plot_constraints()
        # self.plot_objective_function()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0)
        plt.ylim(0)
        plt.title('Linear Programming Visualization')
        return plt


class LinearProgrammingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Firefly Algorithm for Linear Programming")
        self.root.geometry("600x800")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.second_frame = tk.Frame(self.canvas)
        self.second_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.canvas.create_window((0, 0), window=self.second_frame, anchor="nw" , width=500)

        # Objective function inputs
        ttk.Label(self.second_frame, text="Objective Function to maximize").pack(anchor="center", pady=(10, 5))

        obj_frame = tk.Frame(self.second_frame)
        obj_frame.pack(anchor="center")
        self.obj_a = ttk.Entry(obj_frame, width=10)
        self.obj_a.pack(side="left", padx=5)
        ttk.Label(obj_frame, text="x1 +").pack(side="left")
        self.obj_b = ttk.Entry(obj_frame, width=10)
        self.obj_b.pack(side="left", padx=5)
        ttk.Label(obj_frame, text="x2").pack(side="left")

        # Constraints
        self.constraints = []
        button_frame = tk.Frame(self.second_frame)
        button_frame.pack(anchor="center", pady=(10, 5))
        self.add_constraint_button = ttk.Button(button_frame, text="Add Constraint", command=self.add_constraint)
        self.add_constraint_button.pack(side="left", padx=5)
        self.delete_constraint_button = ttk.Button(button_frame, text="Delete Constraint", command=self.delete_constraint)
        self.delete_constraint_button.pack(side="left", padx=5)

        # Bounds
        ttk.Label(self.second_frame, text="Bounds").pack(anchor="center", pady=(10, 5))
        bounds_frame = tk.Frame(self.second_frame)
        bounds_frame.pack(anchor="center")
        self.bounds = []
        for i in range(2):
            bound_frame = tk.Frame(bounds_frame)
            bound_frame.pack(anchor="center")
            ttk.Label(bound_frame, text="x" + str(i + 1) + ": ").pack(side="left")
            min_bound = ttk.Entry(bound_frame, width=10)
            min_bound.pack(side="left", padx=5)
            ttk.Label(bound_frame, text="to").pack(side="left")
            max_bound = ttk.Entry(bound_frame, width=10)
            max_bound.pack(side="left", padx=5)
            self.bounds.append((min_bound, max_bound))

        # Number of iterations
        ttk.Label(self.second_frame, text="Number of Iterations").pack(anchor="center", pady=(5, 0))
        self.num_iterations = ttk.Entry(self.second_frame)
        self.num_iterations.pack(anchor="center", padx=5, pady=(5, 10))

        # Number of fireflies
        ttk.Label(self.second_frame, text="Number of Fireflies").pack(anchor="center", pady=(5, 0))
        self.num_fireflies = ttk.Entry(self.second_frame)
        self.num_fireflies.pack(anchor="center", padx=5, pady=(5, 10))

        # Alpha parameter
        ttk.Label(self.second_frame, text="Alpha").pack(anchor="center", pady=(5, 0))
        self.alpha = ttk.Entry(self.second_frame)
        self.alpha.pack(anchor="center", padx=5, pady=(5, 10))
        self.alpha.insert(0, "0.5")  # Insert the default value

        # Beta parameter
        ttk.Label(self.second_frame, text="Beta").pack(anchor="center", pady=(5, 0))
        self.beta = ttk.Entry(self.second_frame)
        self.beta.pack(anchor="center", padx=5, pady=(5, 10))
        self.beta.insert(0, "0.2")  # Insert the default value

        # Gamma parameter
        ttk.Label(self.second_frame, text="Gamma").pack(anchor="center", pady=(5, 0))
        self.gamma = ttk.Entry(self.second_frame)
        self.gamma.pack(anchor="center", padx=5, pady=(5, 10))
        self.gamma.insert(0, "1.0")  # Insert the default value

        # Start button
        self.start_button = ttk.Button(self.second_frame, text="Start", command=self.start_optimization)
        self.start_button.pack(anchor="center", pady=(10, 10))

        self.constraint_entries = []

    def on_canvas_resize(self, event):
        self.second_frame.place(relx=0.5, rely=0.5, anchor="center", width=500)

    def add_constraint(self):
        row_frame = tk.Frame(self.second_frame)
        row_frame.pack(anchor="center", pady=(5, 10))
        coeffs = [ttk.Entry(row_frame, width=10) for _ in range(2)]
        for coeff in coeffs:
            coeff.pack(side="left", padx=5)
        sign = tk.StringVar()
        sign.set("<=")
        signs = ["<=","<=", ">=", "="]
        ttk.OptionMenu(row_frame, sign, *signs).pack(side="left", padx=5)
        rhs = ttk.Entry(row_frame, width=10)
        rhs.pack(side="left", padx=5)
        self.constraint_entries.append((coeffs, sign, rhs))

    def delete_constraint(self):
        if self.constraint_entries:
            last_row_frame = self.constraint_entries[-1][0][0].master
            last_row_frame.destroy()
            self.constraint_entries.pop()

    def start_optimization(self):
        num_iterations = int(self.num_iterations.get())
        num_fireflies = int(self.num_fireflies.get())
        alpha = float(self.alpha.get())  # Get the alpha parameter from the user's input
        beta = float(self.beta.get())  # Get the beta parameter from the user's input
        gamma = float(self.gamma.get())  # Get the gamma parameter from the user's input

        objective = (float(self.obj_a.get()), float(self.obj_b.get()))
        constraints = []
        for (coeffs, sign, rhs) in self.constraint_entries:
            a = float(coeffs[0].get())
            b = float(coeffs[1].get())
            rhs = float(rhs.get())
            constraints.append((a, b, sign.get(), rhs))

        # constraints = [
        #     (20, 10, '<=', 200),
        #     (10, 20, '<=', 120),
        #     (10, 30, '<=', 150),
        #     (1, 0, '>=', 0),
        #     (0, 1, '>=', 0)
        # ]
        #
        # objective = (5, 12)

        # Calculate bounds
        print("Constraints:", constraints)
        bounds = [(float(min_bound.get()), float(max_bound.get())) for (min_bound, max_bound) in self.bounds]

        vis = LinearProgrammingVisualizer(constraints, objective)
        plt = vis.visualize()
        obj_func = lambda x: objective[0] * x[0] + objective[1] * x[1]

        firefly = FireflyAlgorithm(obj_func, constraints, bounds, self.root, plt, n_fireflies=num_fireflies, max_iter=num_iterations, alpha=alpha, beta=beta, gamma=gamma)
        firefly.optimize()
        plt.show()



# Example usage
# constraints = [
#     (20, 10, '<=', 200),
#     (10, 20, '<=', 120),
#     (10, 30, '<=', 150),
#     (1, 0, '>=', 0),
#     (0, 1, '>=', 0)
# ]

# constraints = [
#     (2, 1, '<=', 600),
#     (0, 0, '<=', 225),
#     (5, 4, '<=', 1000),
#     (0, 2, '>=', 150),
#     (0, 0, '>=', 0)
# ]
#
# # objective = (5, 12)
# objective = (3, 4)
#
# lp_visualizer = LinearProgrammingVisualizer(constraints, objective)
# lp_visualizer.visualize()

root = tk.Tk()
app = LinearProgrammingGUI(root)
root.mainloop()

"""alpha: This is the randomness factor. It controls the randomness in the movement of the fireflies. This randomness 
can help the algorithm avoid getting stuck in local optima by allowing the fireflies to explore the search space. 

beta: This is the attractiveness at zero distance. It controls the attractiveness of fireflies at zero distance. The 
attractiveness decreases as the distance between the fireflies increases. 

gamma: This is the light absorption 
coefficient. It controls the decrease in the light intensity (and thus attractiveness) of a firefly with increasing 
distance."""