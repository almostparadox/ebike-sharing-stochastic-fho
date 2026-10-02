from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scslp.config import ProblemConfig


class TwoStageProblem:
    """Mathematical formulation of Two-Stage Stochastic E-Bike Sharing System (SCSLP)."""

    def __init__(
        self,
        stations_df: pd.DataFrame,
        scenarios_df: pd.DataFrame,
        distance_matrix: pd.DataFrame,
        config: ProblemConfig,
    ) -> None:
        self.stations_df = stations_df.reset_index(drop=True)
        self.scenarios_df = scenarios_df.reset_index(drop=True)
        self.distance_matrix = distance_matrix
        self.config = config

        self.num_stations = config.num_stations
        self.num_scenarios = len(config.scenario_probabilities)

        scenario_arr = np.array(self.scenarios_df["scenario"])
        counts = [
            int(np.sum(scenario_arr == sc)) for sc in range(self.num_scenarios)
        ]
        self.max_trips = max(counts) if counts else 0

        self.t_max = config.t_max
        self.bike_cost = config.bike_cost
        self.budget_fraction = config.budget_fraction

        # Precompute NumPy arrays for fast evaluation and clean typing
        self.capacities = np.array(self.stations_df["capacity"], dtype=int)
        self.opening_costs = np.array(
            self.stations_df["opening_cost"], dtype=float
        )
        self.max_total_bikes = int(np.sum(self.capacities))

        # Vector dimensions:
        # y: [num_stations] (station opening binary)
        # b: [max_total_bikes] (bikes allocated binary)
        # x: [num_scenarios * max_trips] (trip accepted binary)
        self.dim_y = self.num_stations
        self.dim_b = self.max_total_bikes
        self.dim_x = self.num_scenarios * self.max_trips
        self.dimension = self.dim_y + self.dim_b + self.dim_x

        # Precompute budget limit
        total_possible_station_cost = float(np.sum(self.opening_costs))
        total_possible_bike_cost = float(self.bike_cost * self.max_total_bikes)
        self.budget_limit = float(
            self.budget_fraction
            * (total_possible_station_cost + total_possible_bike_cost)
        )

        # Pre-cache trip data per scenario for vectorized fast checks
        self.scenario_trip_origins: list[np.ndarray] = []
        self.scenario_trip_destinations: list[np.ndarray] = []
        self.scenario_trip_profits: list[np.ndarray] = []

        all_origs = np.array(self.scenarios_df["origin_station"], dtype=int)
        all_dests = np.array(
            self.scenarios_df["destination_station"], dtype=int
        )
        all_profits = np.array(self.scenarios_df["profit"], dtype=float)

        for sc in range(self.num_scenarios):
            mask = scenario_arr == sc
            self.scenario_trip_origins.append(all_origs[mask])
            self.scenario_trip_destinations.append(all_dests[mask])
            self.scenario_trip_profits.append(all_profits[mask])

    def decode_solution(self, solution: np.ndarray) -> dict[str, Any]:
        """Decode flat 1D solution vector into decision variables."""
        sol = np.asarray(solution, dtype=int)
        y = sol[: self.dim_y]
        b = sol[self.dim_y : self.dim_y + self.dim_b]
        x = sol[self.dim_y + self.dim_b :].reshape(
            self.num_scenarios, self.max_trips
        )
        return {
            "y": y,
            "b": b,
            "x": x,
            "stations_open": int(np.sum(y)),
            "bikes_allocated": int(np.sum(b)),
            "trips_accepted": int(np.sum(x)),
        }

    def is_feasible(self, solution: np.ndarray) -> bool:
        """Validate all first-stage and second-stage operational constraints."""
        decoded = self.decode_solution(solution)
        y, b, x = decoded["y"], decoded["b"], decoded["x"]

        # Must open at least one station
        if np.sum(y) == 0:
            return False

        # Budget constraint
        total_cost = float(np.dot(self.opening_costs, y)) + float(
            self.bike_cost * np.sum(b)
        )
        if total_cost > self.budget_limit:
            return False

        # Bike capacity constraint: total bikes cannot exceed total capacity of opened stations
        open_capacity = int(np.dot(self.capacities, y))
        total_bikes = int(np.sum(b))
        if total_bikes > open_capacity:
            return False

        # Second-stage feasibility: accepted trips must start and end at open stations
        for sc in range(self.num_scenarios):
            origs = self.scenario_trip_origins[sc]
            dests = self.scenario_trip_destinations[sc]
            num_t = min(len(origs), self.max_trips)
            accepted = x[sc, :num_t] == 1
            if np.any(accepted):
                accepted_origs = origs[:num_t][accepted]
                accepted_dests = dests[:num_t][accepted]
                if np.any(y[accepted_origs] == 0) or np.any(
                    y[accepted_dests] == 0
                ):
                    return False

        return True

    def evaluate(self, solution: np.ndarray) -> float:
        """Calculate net expected profit in IDR. Returns 0.0 if solution violates constraints."""
        if not self.is_feasible(solution):
            return 0.0

        decoded = self.decode_solution(solution)
        y, b, x = decoded["y"], decoded["b"], decoded["x"]

        # Calculate expected trip revenue across scenarios
        expected_revenue = 0.0
        for sc in range(self.num_scenarios):
            prob = self.config.scenario_probabilities[sc]
            profits = self.scenario_trip_profits[sc]
            num_t = min(len(profits), self.max_trips)
            sc_profit = float(np.sum(profits[:num_t] * x[sc, :num_t]))
            expected_revenue += prob * sc_profit

        # Calculate capital expenditure
        station_cost = float(np.dot(self.opening_costs, y))
        bike_cost = float(self.bike_cost * np.sum(b))

        net_profit = expected_revenue - station_cost - bike_cost
        return max(0.0, net_profit)
