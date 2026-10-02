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

    def _repair(self, sol: np.ndarray) -> np.ndarray:
        """Repair discrete solution to satisfy station, capacity, and route constraints."""
        s = sol.copy()
        y = s[: self.problem.num_stations]
        if np.sum(y) == 0:
            y[self.rng.integers(0, self.problem.num_stations)] = 1
            s[: self.problem.num_stations] = y

        open_cap = int(np.dot(self.problem.capacities, y))
        b = s[self.problem.dim_y : self.problem.dim_y + self.problem.dim_b]
        bike_indices = np.where(b == 1)[0]
        if len(bike_indices) > open_cap:
            drop = self.rng.choice(
                bike_indices, size=len(bike_indices) - open_cap, replace=False
            )
            b[drop] = 0
            s[self.problem.dim_y : self.problem.dim_y + self.problem.dim_b] = b

        x = s[self.problem.dim_y + self.problem.dim_b :].reshape(
            self.problem.num_scenarios, self.problem.max_trips
        )
        for sc in range(self.problem.num_scenarios):
            origs = self.problem.scenario_trip_origins[sc]
            dests = self.problem.scenario_trip_destinations[sc]
            num_t = min(len(origs), self.problem.max_trips)
            for t in range(num_t):
                if y[origs[t]] == 0 or y[dests[t]] == 0:
                    x[sc, t] = 0
                elif x[sc, t] == 0 and self.rng.random() < 0.6:
                    x[sc, t] = 1
        s[self.problem.dim_y + self.problem.dim_b :] = x.flatten()
        return s

    def _initialize_population(self) -> np.ndarray:
        """Create initial binary population with repair for station openings."""
        pop = []
        for _ in range(self.config.pop_size):
            ind = np.zeros(self.problem.dimension, dtype=int)
            k_st = self.rng.integers(8, self.problem.num_stations + 1)
            open_sts = self.rng.choice(
                self.problem.num_stations, size=k_st, replace=False
            )
            ind[open_sts] = 1
            open_cap = int(
                np.dot(self.problem.capacities, ind[: self.problem.num_stations])
            )
            n_bikes = min(
                open_cap, self.rng.integers(20, min(56, open_cap + 1))
            )
            ind[self.problem.dim_y : self.problem.dim_y + n_bikes] = 1
            x = ind[self.problem.dim_y + self.problem.dim_b :].reshape(
                self.problem.num_scenarios, self.problem.max_trips
            )
            for sc in range(self.problem.num_scenarios):
                origs = self.problem.scenario_trip_origins[sc]
                dests = self.problem.scenario_trip_destinations[sc]
                num_t = min(len(origs), self.problem.max_trips)
                for t in range(num_t):
                    if (
                        ind[origs[t]] == 1
                        and ind[dests[t]] == 1
                        and self.rng.random() < 0.8
                    ):
                        x[sc, t] = 1
            ind[self.problem.dim_y + self.problem.dim_b :] = x.flatten()
            pop.append(self._repair(ind))
        return np.array(pop)

    def _discretize(self, continuous_vector: np.ndarray) -> np.ndarray:
        """Sigmoid / threshold binarization for discrete variable updates."""
        prob = 1.0 / (1.0 + np.exp(-np.clip(continuous_vector, -10.0, 10.0)))
        binary = (self.rng.random(size=continuous_vector.shape) < prob).astype(
            int
        )
        return self._repair(binary)

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
