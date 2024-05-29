from tkinter import ttk
import matplotlib.animation as animation
import numpy as np
import tkinter as tk


class FireflyAlgorithm:
    def __init__(self, obj_func, constraints, bounds, root, plot, n_fireflies=200, max_iter=50, alpha=0.5, beta=0.2,
                 gamma=1.0):
        self.obj_func = obj_func
        self.constraints = constraints
        self.bounds = bounds
        self.root = root
        self.plot = plot
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.best_firefly = None
        self.best_fireflies = []

        self.fireflies = np.zeros((self.n_fireflies, 2))
        print("Generating fireflies...")
        print("Bounds:", bounds)
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
        best_intensity = -np.inf
        ax = self.plot.gca()
        new_best = None

        for t in range(self.max_iter):
            print("Iteration", t, "Best solution found:", self.best_firefly)

            i = 0
            while i < len(self.fireflies):
                for j in range(len(self.fireflies)):
                    if self.intensities[j] < self.intensities[i]:
                        distance = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                        beta = self.beta * np.exp(-self.gamma * distance ** 2)
                        old_position = self.fireflies[i].copy()  # Store the old position
                        self.fireflies[i] = self.fireflies[i] + beta * (self.fireflies[j] - self.fireflies[i]) + \
                                            self.alpha * (np.random.rand(2) - 0.5)
                        self.fireflies[i] = np.clip(self.fireflies[i], [b[0] for b in self.bounds],
                                                    [b[1] for b in self.bounds])
                        if self._satisfy_constraints(self.fireflies[i]):
                            self.intensities[i] = self.obj_func(self.fireflies[i])
                            if self.intensities[i] > best_intensity:
                                best_intensity = self.intensities[i]
                                self.best_firefly = self.fireflies[i]
                                new_best = self.best_firefly
                        else:
                            # Revert to old position if the new position does not satisfy all constraints
                            self.fireflies[i] = old_position
                i += 1

            if new_best is not None:
                ax.scatter(new_best[0], new_best[1], c='black', s=100, marker='x')
                new_best = None

            # Scatter plot without clearing the axes
            scatter = ax.scatter(self.fireflies[:, 0], self.fireflies[:, 1], c=self.intensities, cmap='viridis')

            scatter.set_clim(vmin=self.intensities.min(), vmax=self.intensities.max())

            # Sleep for one second
            self.plot.pause(1)
            scatter.remove()

        print("Best solution found:", self.best_firefly)
        solution_window = tk.Toplevel(self.root)
        solution_window.title("Best Solution")
        solution_window.geometry("250x100")
        ttk.Label(solution_window, text="Best solution found:").pack()
        ttk.Label(solution_window, text=str(self.best_firefly)).pack()
        ttk.Label(solution_window, text="Objective function value:").pack()
        ttk.Label(solution_window, text=str(best_intensity)).pack()

    def _satisfy_constraints(self, firefly):
        for (a, b, sign, rhs) in self.constraints:
            if sign == '<=':
                if a * firefly[0] + b * firefly[1] > rhs:
                    return False
            elif sign == '>=':
                if a * firefly[0] + b * firefly[1] < rhs:
                    return False
            elif sign == '=':
                if a * firefly[0] + b * firefly[1] != rhs:
                    return False
        return True
