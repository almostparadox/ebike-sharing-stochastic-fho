from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class ProblemConfig:
    num_stations: int = 15
    budget_fraction: float = 0.8
    bike_cost: float = 4_000_000.0
    charging_time: int = 4
    max_trip_duration: int = 3
    max_battery_distance: float = 20.0
    scenario_probabilities: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.5])
    t_max: int = 15
    trip_profit_multiplier: float = 10_000_000.0
    max_walking_time: float = 10.0
    walking_speed_kmph: float = 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_stations": self.num_stations,
            "budget_fraction": self.budget_fraction,
            "bike_cost": self.bike_cost,
            "charging_time": self.charging_time,
            "max_trip_duration": self.max_trip_duration,
            "max_battery_distance": self.max_battery_distance,
            "scenario_probabilities": self.scenario_probabilities,
            "t_max": self.t_max,
            "trip_profit_multiplier": self.trip_profit_multiplier,
            "max_walking_time": self.max_walking_time,
            "walking_speed_kmph": self.walking_speed_kmph,
        }

    def save_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> ProblemConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class FHOConfig:
    pop_size: int = 50
    max_generations: int = 100
    num_fire_hawks: int = 5
    random_seed: int | None = 42


@dataclass
class Station:
    station_id: int
    name: str
    latitude: float
    longitude: float
    capacity: int
    opening_cost: float


@dataclass
class Trip:
    scenario: int
    trip_id: int
    origin: tuple[float, float]
    destination: tuple[float, float]
    start_time: int
    end_time: int
    profit: float
    distance: float
    battery_consumption: float
    probability: float
