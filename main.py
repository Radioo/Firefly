import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import matplotlib
from tkinter import ttk
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")

class FireflyAlgorithm:
    def __init__(self, obj_func, constraints, bounds, n_fireflies=200, max_iter=50, alpha=0.5, beta=0.2, gamma=1.0):
        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.fireflies = np.zeros((self.n_fireflies, 2))
        print("Generating fireflies...")
        for i in range(self.n_fireflies):
            while True:
                firefly = np.random.rand(2)
                firefly[0] = firefly[0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
                firefly[1] = firefly[1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
                if self._satisfy_constraints(firefly):
                    self.fireflies[i] = firefly
                    print(i + 1, "/", self.n_fireflies, "fireflies generated")
                    break

        self.intensities = np.apply_along_axis(self.obj_func, 1, self.fireflies)

    def optimize(self):
        best_firefly = None
        best_intensity = -np.inf

        for t in range(self.max_iter):
            print("Iteration", t, "Best solution found:", best_firefly)

            i = 0
            while i < len(self.fireflies):
                for j in range(len(self.fireflies)):
                    if self.intensities[j] < self.intensities[i]:
                        distance = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                        beta = self.beta * np.exp(-self.gamma * distance ** 2)
                        self.fireflies[i] = self.fireflies[i] + beta * (self.fireflies[j] - self.fireflies[i]) + \
                                            self.alpha * (np.random.rand(2) - 0.5)
                        self.fireflies[i] = np.clip(self.fireflies[i], [b[0] for b in self.bounds],
                                                    [b[1] for b in self.bounds])
                        if self._satisfy_constraints(self.fireflies[i]):
                            self.intensities[i] = self.obj_func(self.fireflies[i])
                            if self.intensities[i] > best_intensity:
                                best_intensity = self.intensities[i]
                                best_firefly = self.fireflies[i]
                        else:
                            # Remove firefly that does not satisfy all constraints
                            self.fireflies = np.delete(self.fireflies, i, axis=0)
                            self.intensities = np.delete(self.intensities, i)
                            i -= 1
                            break
                i += 1

            yield self.fireflies

        print("Best solution found:", best_firefly)

    def _satisfy_constraints(self, firefly):
        for constraint in self.constraints:
            left = np.dot(constraint['coeffs'], firefly)
            if constraint['sign'] == '<=' and not (left <= constraint['rhs']):
                return False
            elif constraint['sign'] == '>=' and not (left >= constraint['rhs']):
                return False
            elif constraint['sign'] == '=' and not (left == constraint['rhs']):
                return False
        return True


class LinearProgrammingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Firefly Algorithm for Linear Programming")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.second_frame = tk.Frame(self.canvas)

        self.canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        # Objective function inputs
        ttk.Label(self.second_frame, text="Objective Function").grid(column=0, row=0)
        self.obj_a = ttk.Entry(self.second_frame)
        self.obj_a.grid(column=1, row=0)
        ttk.Label(self.second_frame, text="x +").grid(column=2, row=0)
        self.obj_b = ttk.Entry(self.second_frame)
        self.obj_b.grid(column=3, row=0)
        ttk.Label(self.second_frame, text="y").grid(column=4, row=0)

        # Maximize or minimize
        self.opt_dir = tk.StringVar()
        ttk.Radiobutton(self.second_frame, text="Maximize", variable=self.opt_dir, value="max").grid(column=0, row=1)
        ttk.Radiobutton(self.second_frame, text="Minimize", variable=self.opt_dir, value="min").grid(column=1, row=1)

        # Constraints
        self.constraints = []
        self.add_constraint_button = ttk.Button(self.second_frame, text="Add Constraint", command=self.add_constraint)
        self.add_constraint_button.grid(column=0, row=2, columnspan=2)

        # Number of iterations
        ttk.Label(self.second_frame, text="Number of Iterations").grid(column=0, row=3)
        self.num_iterations = ttk.Entry(self.second_frame)
        self.num_iterations.grid(column=1, row=3)

        # Number of fireflies
        ttk.Label(self.second_frame, text="Number of Fireflies").grid(column=0, row=4)
        self.num_fireflies = ttk.Entry(self.second_frame)
        self.num_fireflies.grid(column=1, row=4)

        # Alpha parameter
        ttk.Label(self.second_frame, text="Alpha").grid(column=0, row=5)
        self.alpha = ttk.Entry(self.second_frame)
        self.alpha.grid(column=1, row=5)
        self.alpha.insert(0, "0.5")  # Insert the default value

        # Beta parameter
        ttk.Label(self.second_frame, text="Beta").grid(column=0, row=6)
        self.beta = ttk.Entry(self.second_frame)
        self.beta.grid(column=1, row=6)
        self.beta.insert(0, "0.2")  # Insert the default value

        # Gamma parameter
        ttk.Label(self.second_frame, text="Gamma").grid(column=0, row=7)
        self.gamma = ttk.Entry(self.second_frame)
        self.gamma.grid(column=1, row=7)
        self.gamma.insert(0, "1.0")  # Insert the default value

        # Start button
        self.start_button = ttk.Button(self.second_frame, text="Start", command=self.start_optimization)
        self.start_button.grid(column=0, row=8, columnspan=2)

        self.constraint_entries = []

    def add_constraint(self):
        row = len(self.constraint_entries) + 9
        coeffs = [tk.Entry(self.second_frame) for _ in range(2)]
        for i, coeff in enumerate(coeffs):
            coeff.grid(column=i, row=row)
        sign = tk.StringVar()
        sign.set("<=")
        signs = ["<=", ">=", "="]
        ttk.OptionMenu(self.second_frame, sign, *signs).grid(column=2, row=row)
        rhs = tk.Entry(self.second_frame)
        rhs.grid(column=3, row=row)
        self.constraint_entries.append((coeffs, sign, rhs))

    def start_optimization(self):
        obj_func = lambda x: float(self.obj_a.get()) * x[0] + float(self.obj_b.get()) * x[1]
        if self.opt_dir.get() == "min":
            obj_func = lambda x: -obj_func(x)

        constraints = []
        for coeffs, sign, rhs in self.constraint_entries:
            constraints.append({
                'coeffs': [float(coeff.get()) for coeff in coeffs],
                'sign': sign.get(),
                'rhs': float(rhs.get())
            })

        num_iterations = int(self.num_iterations.get())
        num_fireflies = int(self.num_fireflies.get())
        alpha = float(self.alpha.get())  # Get the alpha parameter from the user's input
        beta = float(self.beta.get())  # Get the beta parameter from the user's input
        gamma = float(self.gamma.get())  # Get the gamma parameter from the user's input

        fa = FireflyAlgorithm(obj_func, constraints, bounds=[(0, 1000), (0, 1000)], n_fireflies=num_fireflies, max_iter=num_iterations, alpha=alpha, beta=beta, gamma=gamma)

        fig, ax = plt.subplots()
        scat = ax.scatter([], [], c='red')

        # Create a grid of points
        x = np.linspace(0, 1000, 1000)
        y = np.linspace(0, 1000, 1000)
        X, Y = np.meshgrid(x, y)

        # Check if each point satisfies all the constraints
        for constraint in constraints:
            if constraint['sign'] == '<=':
                region = constraint['coeffs'][0] * X + constraint['coeffs'][1] * Y <= constraint['rhs']
            elif constraint['sign'] == '>=':
                region = constraint['coeffs'][0] * X + constraint['coeffs'][1] * Y >= constraint['rhs']
            else:  # constraint['sign'] == '='
                region = constraint['coeffs'][0] * X + constraint['coeffs'][1] * Y == constraint['rhs']
            ax.imshow(region, extent=(0, 1000, 0, 1000), origin='lower', alpha=0.3, aspect='auto')

        fireflies_list = list(fa.optimize())

        min_x = min(firefly[0] for fireflies in fireflies_list for firefly in fireflies)
        max_x = max(firefly[0] for fireflies in fireflies_list for firefly in fireflies)
        min_y = min(firefly[1] for fireflies in fireflies_list for firefly in fireflies)
        max_y = max(firefly[1] for fireflies in fireflies_list for firefly in fireflies)

        def update(i):
            fireflies = fireflies_list[i]  # Use the i-th element of fireflies_list
            scat.set_offsets(fireflies)
            ax.set_xlim(min_x, max_x)  # Set xlim to the minimum and maximum x values
            ax.set_ylim(min_y, max_y)  # Set ylim to the minimum and maximum y values
            return scat,

        ani = FuncAnimation(fig, update, frames=len(fireflies_list), repeat=False, interval=500)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearProgrammingGUI(root)
    root.mainloop()
