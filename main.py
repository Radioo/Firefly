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

        # Center the main frame in the root window
        self.main_frame = tk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20)

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Objective function inputs
        ttk.Label(self.main_frame, text="Objective Function").grid(column=0, row=0, columnspan=4, pady=(0,20))
        self.obj_a = ttk.Entry(self.main_frame, width=10)
        self.obj_a.grid(column=0, row=1)
        ttk.Label(self.main_frame, text="x1 +").grid(column=1, row=1)
        self.obj_b = ttk.Entry(self.main_frame, width=10)
        self.obj_b.grid(column=2, row=1)
        ttk.Label(self.main_frame, text="x2").grid(column=3, row=1)

        # Maximize or minimize
        self.opt_dir = tk.StringVar()
        ttk.Radiobutton(self.main_frame, text="Maximize", variable=self.opt_dir, value="max").grid(column=0, row=2, columnspan=2, pady=(10,5))
        ttk.Radiobutton(self.main_frame, text="Minimize", variable=self.opt_dir, value="min").grid(column=2, row=2, columnspan=2, pady=(10,5))

        # Constraints
        self.constraints = []
        self.add_constraint_button = ttk.Button(self.main_frame, text="Add Constraint", command=self.add_constraint)
        self.add_constraint_button.grid(column=0, row=3, columnspan=2, pady=(5,10))

        # Delete constraint button
        self.delete_constraint_button = ttk.Button(self.main_frame, text="Delete Constraint", command=self.delete_constraint)
        self.delete_constraint_button.grid(column=2, row=3, columnspan=2, pady=(5,10))

        # Number of iterations
        ttk.Label(self.main_frame, text="Number of Iterations").grid(column=0, row=4, columnspan=2, pady=(5,5))
        self.num_iterations = ttk.Entry(self.main_frame)
        self.num_iterations.grid(column=2, row=4, columnspan=2)

        # Number of fireflies
        ttk.Label(self.main_frame, text="Number of Fireflies").grid(column=0, row=5, columnspan=2, pady=(5,5))
        self.num_fireflies = ttk.Entry(self.main_frame)
        self.num_fireflies.grid(column=2, row=5, columnspan=2)

        # Alpha parameter
        ttk.Label(self.main_frame, text="Alpha").grid(column=0, row=6, columnspan=2, pady=(5,5))
        self.alpha = ttk.Entry(self.main_frame)
        self.alpha.grid(column=2, row=6, columnspan=2)
        self.alpha.insert(0, "0.5")  # Insert the default value

        # Beta parameter
        ttk.Label(self.main_frame, text="Beta").grid(column=0, row=7, columnspan=2, pady=(5,5))
        self.beta = ttk.Entry(self.main_frame)
        self.beta.grid(column=2, row=7, columnspan=2)
        self.beta.insert(0, "0.2")  # Insert the default value

        # Gamma parameter
        ttk.Label(self.main_frame, text="Gamma").grid(column=0, row=8, columnspan=2, pady=(5,5))
        self.gamma = ttk.Entry(self.main_frame)
        self.gamma.grid(column=2, row=8, columnspan=2)
        self.gamma.insert(0, "1.0")  # Insert the default value

        # Start button
        self.start_button = ttk.Button(self.main_frame, text="Start", command=self.start_optimization)
        self.start_button.grid(column=0, row=9, columnspan=4, pady=(15,5))

        self.constraint_entries = []

    def add_constraint(self):
        row = len(self.constraint_entries) + 10
        coeffs = [tk.Entry(self.main_frame, width=5) for _ in range(2)]
        for i, coeff in enumerate(coeffs):
            coeff.grid(column=i, row=row)
        sign = tk.StringVar()
        sign.set("<=")
        signs = [ "<=","<=", ">=", "="]
        sign_option_menu = ttk.OptionMenu(self.main_frame, sign, *signs)
        sign_option_menu.grid(column=2, row=row, pady=(5,5))
        rhs = tk.Entry(self.main_frame, width=5)
        rhs.grid(column=3, row=row)
        #ttk.Label(self.main_frame, text="x2").grid(column=4, row=row)
        self.constraint_entries.append((coeffs, sign_option_menu, rhs))

    def delete_constraint(self):
        if self.constraint_entries:
            coeffs, sign_option_menu, rhs = self.constraint_entries.pop()
            for coeff in coeffs:
                coeff.destroy()
            sign_option_menu.destroy()  # This will destroy the OptionMenu widget
            rhs.destroy()

    def start_optimization(self):
        obj_func = lambda x: float(self.obj_a.get()) * x[0] + float(self.obj_b.get()) * x[1]
        if self.opt_dir.get() == "min":
            obj_func = lambda x: -obj_func(x)

        constraints = []
        for coeffs, sign, rhs in self.constraint_entries:
            constraints.append({
                'coeffs': [float(coeff.get()) for coeff in coeffs],
                'sign': sign.cget("textvariable").get(),
                'rhs': float(rhs.get())
            })

        num_iterations = int(self.num_iterations.get())
        num_fireflies = int(self.num_fireflies.get())
        alpha = float(self.alpha.get())  # Get the alpha parameter
        beta = float(self.beta.get())    # Get the beta parameter
        gamma = float(self.gamma.get())  # Get the gamma parameter

        algorithm = FireflyAlgorithm(obj_func, constraints, [(0, 10), (0, 10)], n_fireflies=num_fireflies,
                                     max_iter=num_iterations, alpha=alpha, beta=beta, gamma=gamma)

        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter([], [])

        def update(frame):
            self.scat.set_offsets(frame)
            return self.scat,

        self.ani = FuncAnimation(self.fig, update, frames=algorithm.optimize, repeat=False)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearProgrammingGUI(root)
    root.mainloop()
