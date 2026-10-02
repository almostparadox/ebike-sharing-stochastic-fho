from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scslp.config import FHOConfig
from scslp.problem import TwoStageProblem


@dataclass
class FHOResult:
    best_solution: np.ndarray
    best_fitness: float
    fitness_history: list[float]
    decoded: dict[str, Any]


class FireHawkOptimizer:
    """Fire Hawk Optimizer (FHO) metaheuristic for the discrete SCSLP problem."""

    def __init__(self, problem: TwoStageProblem, config: FHOConfig) -> None:
        self.problem = problem
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def _initialize_population(self) -> np.ndarray:
        """Create initial binary population with repair for station openings."""
        pop = self.rng.integers(0, 2, size=(self.config.pop_size, self.problem.dimension))
        for i in range(self.config.pop_size):
            # Ensure at least 1 station is open
            if np.sum(pop[i, :self.problem.num_stations]) == 0:
                random_st = self.rng.integers(0, self.problem.num_stations)
                pop[i, random_st] = 1
        return pop

    def _discretize(self, continuous_vector: np.ndarray) -> np.ndarray:
        """Sigmoid / threshold binarization for discrete variable updates."""
        prob = 1.0 / (1.0 + np.exp(-np.clip(continuous_vector, -10.0, 10.0)))
        binary = (self.rng.random(size=continuous_vector.shape) < prob).astype(int)
        if np.sum(binary[:self.problem.num_stations]) == 0:
            binary[self.rng.integers(0, self.problem.num_stations)] = 1
        return binary

    def optimize(self) -> FHOResult:
        """Execute Fire Hawk Optimizer iterations."""
        pop = self._initialize_population()
        fitnesses = np.array([self.problem.evaluate(ind) for ind in pop])

        best_idx = int(np.argmax(fitnesses))
        best_sol = pop[best_idx].copy()
        best_fit = float(fitnesses[best_idx])
        history: list[float] = []

        num_fh = max(1, min(self.config.num_fire_hawks, self.config.pop_size // 2))

        for _ in range(self.config.max_generations):
            # Sort population by descending fitness
            sorted_indices = np.argsort(fitnesses)[::-1]
            pop = pop[sorted_indices]
            fitnesses = fitnesses[sorted_indices]

            fire_hawks = pop[:num_fh].copy()
            preys = pop[num_fh:].copy()

            new_pop = list(fire_hawks)

            for fh_idx in range(num_fh):
                fh = fire_hawks[fh_idx]
                # Fire Hawk position update
                r1 = self.rng.random(size=self.problem.dimension)
                r2 = self.rng.random(size=self.problem.dimension)
                updated_fh_cont = fh + (r1 * (best_sol - fh)) + (r2 * (fire_hawks[self.rng.integers(0, num_fh)] - fh))
                updated_fh = self._discretize(updated_fh_cont)
                fit_fh = self.problem.evaluate(updated_fh)
                if fit_fh > fitnesses[fh_idx]:
                    new_pop[fh_idx] = updated_fh
                    fitnesses[fh_idx] = fit_fh

            # Preys update
            for prey in preys:
                selected_fh = fire_hawks[self.rng.integers(0, num_fh)]
                r3 = self.rng.random(size=self.problem.dimension)
                updated_prey_cont = prey + r3 * (selected_fh - prey)
                updated_prey = self._discretize(updated_prey_cont)
                new_pop.append(updated_prey)

            pop = np.array(new_pop[:self.config.pop_size])
            fitnesses = np.array([self.problem.evaluate(ind) for ind in pop])

            current_best_idx = int(np.argmax(fitnesses))
            if fitnesses[current_best_idx] > best_fit:
                best_fit = float(fitnesses[current_best_idx])
                best_sol = pop[current_best_idx].copy()

            history.append(best_fit)

        return FHOResult(
            best_solution=best_sol,
            best_fitness=best_fit,
            fitness_history=history,
            decoded=self.problem.decode_solution(best_sol),
        )
