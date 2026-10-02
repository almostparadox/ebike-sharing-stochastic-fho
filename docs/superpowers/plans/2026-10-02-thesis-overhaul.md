# Thesis Code and Document Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the undergraduate mathematics thesis workspace into a clean, reproducible, modular Python package (`scslp`), a pristine LaTeX document workspace (`thesis/`), canonical datasets (`data/`), and a publication-grade portfolio showcase (`README.md`).

**Architecture:** Monorepo with distinct decoupled components: `code/` contains a modern Python package managed by `uv` implementing the Two-Stage Stochastic E-Bike Sharing model and Fire Hawk Optimizer metaheuristic; `data/` houses canonical Yogyakarta station and scenario datasets; `thesis/` holds modular LaTeX chapters compiled out-of-source into `build/main.pdf`.

**Tech Stack:** Python 3.10+, `uv`, NumPy, Pandas, Geopy, Folium, Matplotlib, Pytest, LaTeX / `latexmk`, BibTeX.

**Spec:** `docs/superpowers/specs/2026-10-02-thesis-overhaul-design.md`

## Global Constraints
- Target Python 3.10+ compatibility using standard `uv` dependency management.
- All code in `code/src/scslp` must be typed, formatted, and covered by unit tests in `code/tests`.
- Random processes must accept an explicit `random_state` or `seed` for deterministic reproducibility.
- Zero hardcoded `/kaggle/input/` or Colab-specific magic commands (`!pip`).
- LaTeX build must isolate all artifacts into `thesis/build/` without leaving clutter in the source tree.
- Original examination conclusions and results (IDR 274,030,876 net profit, 12 stations, 55 bikes, 80 trips accepted) must be accurately represented and preserved.

---

### Task 1: Environment & Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `code/pyproject.toml`
- Create: `code/src/scslp/__init__.py`
- Test: `code/tests/test_scaffold.py`

**Interfaces:**
- Produces: `scslp.__version__ == "0.1.0"`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_scaffold.py
import scslp

def test_version():
    assert scslp.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_scaffold.py -v
```
Expected: FAIL (ModuleNotFoundError or AttributeError)

- [ ] **Step 3: Write minimal implementation**

Create `.gitignore` at root:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.venv
venv/
ENV/
.pytest_cache/

# OS
.DS_Store
Thumbs.db

# LaTeX build artifacts
thesis/build/
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.lof
*.log
*.lot
*.out
*.synctex.gz
*.toc
```

Create `code/pyproject.toml`:
```toml
[project]
name = "scslp"
version = "0.1.0"
description = "Two-Stage Stochastic E-Bike Sharing System Optimization in Yogyakarta with Fire Hawk Optimizer"
authors = [{ name = "Saddam Aditya Hartanto" }]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "geopy>=2.4.0",
    "folium>=0.16.0",
    "matplotlib>=3.8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
scslp-cli = "scslp.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/scslp"]
```

Create `code/src/scslp/__init__.py`:
```python
"""SCSLP: Two-Stage Stochastic E-Bike Sharing Optimization with Fire Hawk Optimizer."""

__version__ = "0.1.0"
```

Create `code/README.md`:
```markdown
# SCSLP Python Package

Two-Stage Stochastic Capacitated Station Location Problem (SCSLP) solver for Electric Bike-Sharing Systems in Yogyakarta using the Fire Hawk Optimizer (FHO) metaheuristic.
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_scaffold.py -v
```
Expected: PASS (`1 passed in ...`)

- [ ] **Step 5: Commit**

```bash
git add .gitignore code/pyproject.toml code/src/scslp/__init__.py code/README.md code/tests/test_scaffold.py
git commit -m "feat(code): scaffold scslp package with uv and pyproject.toml"
```

---

### Task 2: Configuration Dataclasses (`scslp/config.py`)

**Files:**
- Create: `code/src/scslp/config.py`
- Create: `code/tests/test_config.py`

**Interfaces:**
- Consumes: Nothing
- Produces:
  - `ProblemConfig(num_stations: int, budget_fraction: float, bike_cost: float, charging_time: int, max_trip_duration: int, max_battery_distance: float, scenario_probabilities: list[float], trip_profit_multiplier: float)`
  - `FHOConfig(pop_size: int, max_generations: int, num_fire_hawks: int, random_seed: int | None)`
  - `Station(station_id: int, name: str, latitude: float, longitude: float, capacity: int, opening_cost: float)`
  - `Trip(scenario: int, trip_id: int, origin: tuple[float, float], destination: tuple[float, float], start_time: int, end_time: int, profit: float, distance: float, battery_consumption: float, probability: float)`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_config.py
from scslp.config import ProblemConfig, FHOConfig, Station, Trip

def test_default_problem_config():
    cfg = ProblemConfig()
    assert cfg.num_stations == 15
    assert cfg.budget_fraction == 0.8
    assert cfg.bike_cost == 4_000_000.0
    assert cfg.scenario_probabilities == [0.2, 0.3, 0.5]
    assert len(cfg.scenario_probabilities) == 3

def test_default_fho_config():
    fho_cfg = FHOConfig(pop_size=50, max_generations=100, random_seed=42)
    assert fho_cfg.pop_size == 50
    assert fho_cfg.max_generations == 100
    assert fho_cfg.random_seed == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_config.py -v
```
Expected: FAIL (ImportError: cannot import name 'ProblemConfig')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/config.py
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
        with open(path, "r", encoding="utf-8") as f:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_config.py -v
```
Expected: PASS (`2 passed in ...`)

- [ ] **Step 5: Commit**

```bash
git add code/src/scslp/config.py code/tests/test_config.py
git commit -m "feat(code): add ProblemConfig, FHOConfig, Station, and Trip dataclasses"
```

---

### Task 3: Deterministic Data Generator & Canonical Datasets (`scslp/generator.py`)

**Files:**
- Create: `code/src/scslp/generator.py`
- Create: `code/tests/test_generator.py`
- Create: `data/stations_data.csv`
- Create: `data/scenarios_data.csv`
- Create: `data/distance_matrix.csv`
- Create: `data/config.json`

**Interfaces:**
- Consumes: `ProblemConfig`, `Station`, `Trip` from `scslp.config`
- Produces:
  - `generate_synthetic_data(config: ProblemConfig, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - `save_datasets(stations_df: pd.DataFrame, scenarios_df: pd.DataFrame, distance_matrix_df: pd.DataFrame, output_dir: Path)`
  - `load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ProblemConfig]`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_generator.py
from pathlib import Path
from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data, save_datasets, load_datasets

def test_generate_synthetic_data():
    cfg = ProblemConfig(num_stations=15)
    stations_df, scenarios_df, dist_matrix = generate_synthetic_data(cfg, seed=42)
    
    assert len(stations_df) == 15
    assert set(stations_df.columns) >= {"station_id", "name", "latitude", "longitude", "capacity", "opening_cost"}
    assert len(scenarios_df) > 0
    assert set(scenarios_df.columns) >= {"scenario", "trip_id", "origin", "destination", "profit", "distance"}
    assert dist_matrix.shape == (15, 15)
    # Diagonal distance must be zero
    for i in range(15):
        assert dist_matrix.iloc[i, i] == 0.0

def test_save_and_load_datasets(tmp_path: Path):
    cfg = ProblemConfig(num_stations=15)
    stations_df, scenarios_df, dist_matrix = generate_synthetic_data(cfg, seed=42)
    save_datasets(stations_df, scenarios_df, dist_matrix, cfg, tmp_path)
    
    st_df, sc_df, dm_df, loaded_cfg = load_datasets(tmp_path)
    assert len(st_df) == 15
    assert len(sc_df) == len(scenarios_df)
    assert dm_df.shape == (15, 15)
    assert loaded_cfg.num_stations == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_generator.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'scslp.generator')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/generator.py
from __future__ import annotations

import json
from pathlib import Path
import random
import ast
from geopy.distance import geodesic
import numpy as np
import pandas as pd

from scslp.config import ProblemConfig

# Historical candidate station names and strategic anchor coordinates around Yogyakarta
YOGYAKARTA_STATIONS = [
    {"name": "Sidomoyo", "lat": -7.7601, "lon": 110.3245},
    {"name": "Monjali", "lat": -7.7547, "lon": 110.3695},
    {"name": "Kentungan", "lat": -7.7554, "lon": 110.3802},
    {"name": "Seturan", "lat": -7.7681, "lon": 110.4089},
    {"name": "UGM", "lat": -7.7707, "lon": 110.3777},
    {"name": "Tugu Yogyakarta", "lat": -7.7829, "lon": 110.3671},
    {"name": "Stasiun Tugu", "lat": -7.7891, "lon": 110.3614},
    {"name": "Cik Di Tiro", "lat": -7.7812, "lon": 110.3755},
    {"name": "Malioboro Mall", "lat": -7.7931, "lon": 110.3656},
    {"name": "Titik Nol KM", "lat": -7.8002, "lon": 110.3648},
    {"name": "Kraton Yogyakarta", "lat": -7.8053, "lon": 110.3642},
    {"name": "Simpang Empat Pelem", "lat": -7.8094, "lon": 110.3956},
    {"name": "Ngestiharjo", "lat": -7.7967, "lon": 110.3421},
    {"name": "Mergangsan", "lat": -7.8182, "lon": 110.3734},
    {"name": "Bandara Adisutjipto", "lat": -7.7882, "lon": 110.4318},
]


def generate_synthetic_data(
    config: ProblemConfig,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate deterministic candidate stations, scenarios, and distance matrix."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 1. Generate candidate stations
    stations = []
    for i in range(config.num_stations):
        meta = YOGYAKARTA_STATIONS[i % len(YOGYAKARTA_STATIONS)]
        capacity = rng.randint(5, 10)
        opening_cost = rng.randint(10_000_000, 50_000_000)
        stations.append({
            "station_id": i,
            "name": meta["name"],
            "latitude": meta["lat"],
            "longitude": meta["lon"],
            "capacity": capacity,
            "opening_cost": opening_cost,
        })
    stations_df = pd.DataFrame(stations)

    # 2. Distance matrix
    n_st = len(stations_df)
    dist_matrix = pd.DataFrame(
        np.zeros((n_st, n_st)),
        index=stations_df["station_id"],
        columns=stations_df["station_id"],
    )
    for i in range(n_st):
        coord_i = (stations_df.loc[i, "latitude"], stations_df.loc[i, "longitude"])
        for j in range(n_st):
            if i != j:
                coord_j = (stations_df.loc[j, "latitude"], stations_df.loc[j, "longitude"])
                dist_matrix.iloc[i, j] = geodesic(coord_i, coord_j).kilometers
            else:
                dist_matrix.iloc[i, j] = 0.0

    # 3. Generate demand scenarios (Weekdays, Saturday, Sunday)
    trips_per_scenario = [30] * len(config.scenario_probabilities)
    scenarios = []

    for sc_idx, num_trips in enumerate(trips_per_scenario):
        prob = config.scenario_probabilities[sc_idx]
        for trip_id in range(num_trips):
            while True:
                orig_st, dest_st = rng.sample(range(n_st), 2)
                orig_coord = (
                    stations_df.loc[orig_st, "latitude"] + rng.uniform(-0.005, 0.005),
                    stations_df.loc[orig_st, "longitude"] + rng.uniform(-0.005, 0.005),
                )
                dest_coord = (
                    stations_df.loc[dest_st, "latitude"] + rng.uniform(-0.005, 0.005),
                    stations_df.loc[dest_st, "longitude"] + rng.uniform(-0.005, 0.005),
                )
                trip_dist = geodesic(orig_coord, dest_coord).kilometers

                # Check walking feasibility to nearest station
                walk_orig = geodesic(orig_coord, (stations_df.loc[orig_st, "latitude"], stations_df.loc[orig_st, "longitude"])).kilometers
                walk_dest = geodesic(dest_coord, (stations_df.loc[dest_st, "latitude"], stations_df.loc[dest_st, "longitude"])).kilometers
                time_orig = (walk_orig / config.walking_speed_kmph) * 60
                time_dest = (walk_dest / config.walking_speed_kmph) * 60

                if time_orig <= config.max_walking_time and time_dest <= config.max_walking_time:
                    sk = rng.randint(0, max(0, config.t_max - config.max_trip_duration - 1))
                    ek = rng.randint(sk + 1, min(config.t_max, sk + config.max_trip_duration))
                    duration = ek - sk
                    profit = config.trip_profit_multiplier * duration
                    battery_cons = min(1.0, trip_dist / config.max_battery_distance)

                    scenarios.append({
                        "scenario": sc_idx,
                        "trip_id": trip_id,
                        "origin_station": orig_st,
                        "destination_station": dest_st,
                        "origin": str(orig_coord),
                        "destination": str(dest_coord),
                        "start_time": sk,
                        "end_time": ek,
                        "profit": profit,
                        "distance": trip_dist,
                        "battery_consumption": battery_cons,
                        "probability": prob,
                    })
                    break

    scenarios_df = pd.DataFrame(scenarios)
    return stations_df, scenarios_df, dist_matrix


def save_datasets(
    stations_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    dist_matrix_df: pd.DataFrame,
    config: ProblemConfig,
    output_dir: Path | str,
) -> None:
    """Save generated data and configuration to directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stations_df.to_csv(out / "stations_data.csv", index=False)
    scenarios_df.to_csv(out / "scenarios_data.csv", index=False)
    dist_matrix_df.to_csv(out / "distance_matrix.csv")
    config.save_json(out / "config.json")


def load_datasets(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ProblemConfig]:
    """Load canonical datasets from directory."""
    path = Path(data_dir)
    stations_df = pd.read_csv(path / "stations_data.csv")
    scenarios_df = pd.read_csv(path / "scenarios_data.csv")
    dist_matrix_df = pd.read_csv(path / "distance_matrix.csv", index_col=0)
    config = ProblemConfig.from_json(path / "config.json")
    return stations_df, scenarios_df, dist_matrix_df, config
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_generator.py -v
```
Expected: PASS (`2 passed in ...`)

- [ ] **Step 5: Generate canonical data files in `data/`**

Run one-liner to generate `data/` folder files:
```bash
uv run --project code python -c "
from pathlib import Path
from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data, save_datasets
cfg = ProblemConfig()
st, sc, dm = generate_synthetic_data(cfg, seed=42)
save_datasets(st, sc, dm, cfg, Path('data'))
print('Canonical data generated in data/')
"
```
Expected output: `Canonical data generated in data/`

- [ ] **Step 6: Commit**

```bash
git add code/src/scslp/generator.py code/tests/test_generator.py data/
git commit -m "feat(code): implement deterministic data generator and produce canonical datasets"
```

---

### Task 4: Problem Formulation & Constraint Validator (`scslp/problem.py`)

**Files:**
- Create: `code/src/scslp/problem.py`
- Create: `code/tests/test_problem.py`

**Interfaces:**
- Consumes: `ProblemConfig` from `scslp.config`
- Produces:
  - `TwoStageProblem(stations_df, scenarios_df, dist_matrix, config)`
    - `dimension: int` property
    - `evaluate(solution: np.ndarray) -> float` (returns net expected profit in IDR or 0.0 if infeasible)
    - `is_feasible(solution: np.ndarray) -> bool`
    - `decode_solution(solution: np.ndarray) -> dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_problem.py
import numpy as np
from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data
from scslp.problem import TwoStageProblem

def test_problem_initialization_and_evaluation():
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, dm_df = generate_synthetic_data(cfg, seed=42)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)

    assert problem.dimension > 0

    # Trivial solution with no stations open must yield 0 profit
    zero_sol = np.zeros(problem.dimension, dtype=int)
    profit = problem.evaluate(zero_sol)
    assert profit == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_problem.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'scslp.problem')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/problem.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from scslp.config import ProblemConfig


class TwoStageProblem:
    """Mathematical formulation of the Two-Stage Stochastic E-Bike Sharing System (SCSLP)."""

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
        self.max_trips = max(self.scenarios_df.groupby("scenario")["trip_id"].count())
        self.t_max = config.t_max
        self.bike_cost = config.bike_cost
        self.budget_fraction = config.budget_fraction

        # Bike capacity upper bound across system
        self.max_total_bikes = int(self.stations_df["capacity"].sum())

        # Vector dimensions:
        # y: [num_stations] (station opening binary)
        # b: [max_total_bikes] (bikes allocated binary)
        # x: [num_scenarios * max_trips] (trip accepted binary)
        self.dim_y = self.num_stations
        self.dim_b = self.max_total_bikes
        self.dim_x = self.num_scenarios * self.max_trips
        self.dimension = self.dim_y + self.dim_b + self.dim_x

        # Precompute budget limit
        total_possible_station_cost = self.stations_df["opening_cost"].sum()
        total_possible_bike_cost = self.bike_cost * self.max_total_bikes
        self.budget_limit = self.budget_fraction * (total_possible_station_cost + total_possible_bike_cost)

    def decode_solution(self, solution: np.ndarray) -> dict[str, Any]:
        """Decode flat 1D solution vector into decision variables."""
        sol = np.asarray(solution, dtype=int)
        y = sol[:self.dim_y]
        b = sol[self.dim_y:self.dim_y + self.dim_b]
        x = sol[self.dim_y + self.dim_b:].reshape(self.num_scenarios, self.max_trips)
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
        total_cost = np.dot(self.stations_df["opening_cost"], y) + self.bike_cost * np.sum(b)
        if total_cost > self.budget_limit:
            return False

        # Bike capacity constraint: total bikes cannot exceed total capacity of opened stations
        open_capacity = int(np.dot(self.stations_df["capacity"], y))
        total_bikes = int(np.sum(b))
        if total_bikes > open_capacity:
            return False

        # Second-stage feasibility: accepted trips must start and end at open stations
        for sc in range(self.num_scenarios):
            sc_trips = self.scenarios_df[self.scenarios_df["scenario"] == sc].reset_index(drop=True)
            for trip_idx, row in sc_trips.iterrows():
                if trip_idx >= self.max_trips:
                    break
                if x[sc, trip_idx] == 1:
                    orig_st = int(row["origin_station"])
                    dest_st = int(row["destination_station"])
                    if y[orig_st] == 0 or y[dest_st] == 0:
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
            sc_trips = self.scenarios_df[self.scenarios_df["scenario"] == sc].reset_index(drop=True)
            num_t = min(len(sc_trips), self.max_trips)
            sc_profit = float(np.sum(sc_trips["profit"].iloc[:num_t].values * x[sc, :num_t]))
            expected_revenue += prob * sc_profit

        # Calculate capital expenditure
        station_cost = float(np.dot(self.stations_df["opening_cost"], y))
        bike_cost = float(self.bike_cost * np.sum(b))

        net_profit = expected_revenue - station_cost - bike_cost
        return max(0.0, net_profit)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_problem.py -v
```
Expected: PASS (`1 passed in ...`)

- [ ] **Step 5: Commit**

```bash
git add code/src/scslp/problem.py code/tests/test_problem.py
git commit -m "feat(code): implement TwoStageProblem with budget, capacity, and trip constraints"
```

---

### Task 5: Fire Hawk Optimizer Implementation (`scslp/fho.py`)

**Files:**
- Create: `code/src/scslp/fho.py`
- Create: `code/tests/test_fho.py`

**Interfaces:**
- Consumes: `TwoStageProblem` from `scslp.problem`, `FHOConfig` from `scslp.config`
- Produces:
  - `FireHawkOptimizer(problem: TwoStageProblem, config: FHOConfig)`
    - `optimize() -> FHOResult(best_solution: np.ndarray, best_fitness: float, fitness_history: list[float], decoded: dict[str, Any])`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_fho.py
import numpy as np
from scslp.config import ProblemConfig, FHOConfig
from scslp.generator import generate_synthetic_data
from scslp.problem import TwoStageProblem
from scslp.fho import FireHawkOptimizer

def test_fho_optimization_runs():
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, dm_df = generate_synthetic_data(cfg, seed=42)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)

    fho_cfg = FHOConfig(pop_size=10, max_generations=5, num_fire_hawks=2, random_seed=42)
    optimizer = FireHawkOptimizer(problem, fho_cfg)
    result = optimizer.optimize()

    assert len(result.fitness_history) == 5
    assert result.best_solution.shape == (problem.dimension,)
    assert isinstance(result.best_fitness, float)
    assert result.best_fitness >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_fho.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'scslp.fho')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/fho.py
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

        for gen in range(self.config.max_generations):
            # Sort population by descending fitness
            sorted_indices = np.argsort(fitnesses)[::-1]
            pop = pop[sorted_indices]
            fitnesses = fitnesses[sorted_indices]

            fire_hawks = pop[:num_fh].copy()
            preys = pop[num_fh:].copy()

            # Assign preys to nearest Fire Hawk territory based on hamming distance
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_fho.py -v
```
Expected: PASS (`1 passed in ...`)

- [ ] **Step 5: Commit**

```bash
git add code/src/scslp/fho.py code/tests/test_fho.py
git commit -m "feat(code): implement discrete Fire Hawk Optimizer algorithm with territory and prey updates"
```

---

### Task 6: Visualization Suite (`scslp/viz.py`)

**Files:**
- Create: `code/src/scslp/viz.py`
- Create: `code/tests/test_viz.py`

**Interfaces:**
- Consumes: `stations_df`, `scenarios_df`, `FHOResult`
- Produces:
  - `create_folium_map(stations_df: pd.DataFrame, scenarios_df: pd.DataFrame, solution_dict: dict[str, Any] | None = None) -> folium.Map`
  - `plot_convergence(fitness_history: list[float], output_path: str | Path | None = None) -> matplotlib.figure.Figure`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_viz.py
from pathlib import Path
import folium
import matplotlib.figure
from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data
from scslp.viz import create_folium_map, plot_convergence

def test_create_folium_map(tmp_path: Path):
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, _ = generate_synthetic_data(cfg, seed=42)
    m = create_folium_map(st_df, sc_df)
    assert isinstance(m, folium.Map)
    out_html = tmp_path / "test_map.html"
    m.save(str(out_html))
    assert out_html.exists()

def test_plot_convergence(tmp_path: Path):
    history = [100.0, 250.0, 400.0, 500.0]
    out_png = tmp_path / "convergence.png"
    fig = plot_convergence(history, output_path=out_png)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert out_png.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_viz.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'scslp.viz')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/viz.py
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
import folium
import matplotlib.pyplot as plt
import pandas as pd


def create_folium_map(
    stations_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    solution_dict: dict[str, Any] | None = None,
) -> folium.Map:
    """Create interactive Folium map showing candidate stations and trip trajectories."""
    center_lat = float(stations_df["latitude"].mean())
    center_lon = float(stations_df["longitude"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    y = solution_dict.get("y") if solution_dict else None

    # Add station markers
    for idx, row in stations_df.iterrows():
        is_built = y[idx] == 1 if y is not None else True
        color = "green" if is_built else "gray"
        icon = "bicycle" if is_built else "info-sign"

        status_text = "STATUS: BUILT" if is_built else "STATUS: NOT BUILT"
        popup_html = (
            f"<b>{row['name']}</b> (ID: {row['station_id']})<br>"
            f"{status_text}<br>"
            f"Capacity: {row['capacity']} bikes<br>"
            f"Opening Cost: IDR {row['opening_cost']:,}"
        )
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    # Add trip polylines
    sample_trips = scenarios_df.head(40)
    for _, trip in sample_trips.iterrows():
        orig = ast.literal_eval(trip["origin"]) if isinstance(trip["origin"], str) else trip["origin"]
        dest = ast.literal_eval(trip["destination"]) if isinstance(trip["destination"], str) else trip["destination"]
        folium.PolyLine(
            locations=[orig, dest],
            color="#FF5733",
            weight=1.5,
            opacity=0.7,
            popup=f"Scenario: {trip['scenario']}, Trip: {trip['trip_id']}",
        ).add_to(m)

    return m


def plot_convergence(
    fitness_history: list[float],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot optimization iteration vs. best net expected profit."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(fitness_history) + 1), fitness_history, marker="o", color="#D32F2F", linewidth=2)
    ax.set_title("Konvergensi Fire Hawk Optimizer (SCSLP)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iterasi (Generasi)", fontsize=11)
    ax.set_ylabel("Keuntungan Bersih Harapan (IDR)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_viz.py -v
```
Expected: PASS (`2 passed in ...`)

- [ ] **Step 5: Commit**

```bash
git add code/src/scslp/viz.py code/tests/test_viz.py
git commit -m "feat(code): implement interactive Folium mapping and Matplotlib convergence visualizer"
```

---

### Task 7: Command Line Interface & Simulation Scripts (`scslp/cli.py`)

**Files:**
- Create: `code/src/scslp/cli.py`
- Create: `code/scripts/run_simulation.py`
- Create: `code/tests/test_cli.py`

**Interfaces:**
- Produces: CLI commands `generate`, `solve`, `visualize` executable via `uv run scslp-cli <command>`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_cli.py
from pathlib import Path
from scslp.cli import cli_generate, cli_solve

def test_cli_generate_and_solve(tmp_path: Path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    # Test generate
    cli_generate(["--outdir", str(data_dir), "--seed", "42"])
    assert (data_dir / "stations_data.csv").exists()
    assert (data_dir / "config.json").exists()

    # Test solve
    cli_solve(["--datadir", str(data_dir), "--outdir", str(results_dir), "--generations", "5", "--pop-size", "10"])
    assert (results_dir / "results.json").exists()
    assert (results_dir / "convergence.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run --project code pytest code/tests/test_cli.py -v
```
Expected: FAIL (ModuleNotFoundError: No module named 'scslp.cli')

- [ ] **Step 3: Write minimal implementation**

```python
# code/src/scslp/cli.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from scslp.config import ProblemConfig, FHOConfig
from scslp.generator import generate_synthetic_data, save_datasets, load_datasets
from scslp.problem import TwoStageProblem
from scslp.fho import FireHawkOptimizer
from scslp.viz import create_folium_map, plot_convergence


def cli_generate(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for Yogyakarta SCSLP.")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed = parser.parse_args(args)

    cfg = ProblemConfig()
    stations_df, scenarios_df, dist_matrix = generate_synthetic_data(cfg, seed=parsed.seed)
    save_datasets(stations_df, scenarios_df, dist_matrix, cfg, parsed.outdir)
    print(f"Datasets generated successfully in {parsed.outdir}")


def cli_solve(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve SCSLP with Fire Hawk Optimizer.")
    parser.add_argument("--datadir", type=str, default="data", help="Directory with datasets")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for output")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed = parser.parse_args(args)

    st_df, sc_df, dm_df, cfg = load_datasets(parsed.datadir)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)
    fho_cfg = FHOConfig(pop_size=parsed.pop_size, max_generations=parsed.generations, random_seed=parsed.seed)

    print(f"Solving SCSLP (Dim: {problem.dimension}, Pop: {fho_cfg.pop_size}, Gen: {fho_cfg.max_generations})...")
    optimizer = FireHawkOptimizer(problem, fho_cfg)
    result = optimizer.optimize()

    out = Path(parsed.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_fitness_idr": result.best_fitness,
        "stations_opened": result.decoded["stations_open"],
        "bikes_allocated": result.decoded["bikes_allocated"],
        "trips_accepted": result.decoded["trips_accepted"],
        "fitness_history": result.fitness_history,
    }
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_convergence(result.fitness_history, out / "convergence.png")
    map_viz = create_folium_map(st_df, sc_df, result.decoded)
    map_viz.save(str(out / "solution_map.html"))

    print(f"Simulation completed. Best fitness: IDR {result.best_fitness:,.2f}")
    print(f"Results saved to {parsed.outdir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCSLP E-Bike Sharing Optimization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_p = subparsers.add_parser("generate", help="Generate synthetic datasets")
    gen_p.set_defaults(func=lambda args: cli_generate(args))

    solve_p = subparsers.add_parser("solve", help="Run FHO optimization")
    solve_p.set_defaults(func=lambda args: cli_solve(args))

    parsed, remaining = parser.parse_known_args()
    if parsed.command == "generate":
        cli_generate(remaining)
    elif parsed.command == "solve":
        cli_solve(remaining)


if __name__ == "__main__":
    main()
```

Create `code/scripts/run_simulation.py`:
```python
#!/usr/bin/env python3
"""Run full simulation replicating thesis parameters."""
from pathlib import Path
from scslp.cli import cli_solve

if __name__ == "__main__":
    cli_solve(["--datadir", "data", "--outdir", "results", "--generations", "100", "--pop-size", "50", "--seed", "42"])
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
uv run --project code pytest code/tests/test_cli.py -v
```
Expected: PASS (`1 passed in ...`)

- [ ] **Step 5: Run full test suite for `code/`**

Run:
```bash
uv run --project code pytest code/tests -v
```
Expected: All tests pass (100% green)

- [ ] **Step 6: Commit**

```bash
git add code/src/scslp/cli.py code/scripts/run_simulation.py code/tests/test_cli.py
git commit -m "feat(code): implement unified CLI and thesis simulation script"
```

---

### Task 8: Thesis Directory Restructuring & Out-of-Source Build Automation (`thesis/`)

**Files:**
- Create: `thesis/Makefile`
- Create: `thesis/.latexmkrc`
- Move & Rename:
  - `SCSLP__Copy_/skripsimathugm.cls` $\to$ `thesis/skripsimathugm.cls`
  - `SCSLP__Copy_/setspace.sty` $\to$ `thesis/setspace.sty`
  - `SCSLP__Copy_/Skripsi.tex` $\to$ `thesis/main.tex`
  - `SCSLP__Copy_/Bab1.tex` $\to$ `thesis/chapters/bab1_pendahuluan.tex`
  - `SCSLP__Copy_/Bab2.tex` $\to$ `thesis/chapters/bab2_landasan_teori.tex`
  - `SCSLP__Copy_/Bab3.tex` $\to$ `thesis/chapters/bab3_optimisasi_stokastik.tex`
  - `SCSLP__Copy_/Bab4.tex` $\to$ `thesis/chapters/bab4_pemodelan_simulasi.tex`
  - `SCSLP__Copy_/Bab5.tex` $\to$ `thesis/chapters/bab5_penutup.tex`
  - `SCSLP__Copy_/pengesahanskripsi.pdf` $\to$ `thesis/attachments/pengesahanskripsi.pdf`
  - `SCSLP__Copy_/pernyataan.pdf` $\to$ `thesis/attachments/pernyataan.pdf`
  - Images (`*.png`, `*.jpg`, `Gambar FHO/`) $\to$ `thesis/figures/`
- Delete: Redundant stub `Bab6.tex` and stray `Main.java`

- [ ] **Step 1: Create thesis subfolders**

```bash
mkdir -p thesis/chapters thesis/figures thesis/attachments thesis/appendices
```

- [ ] **Step 2: Copy assets, official attachments, and class files**

```bash
cp "SCSLP__Copy_/skripsimathugm.cls" thesis/
cp "SCSLP__Copy_/setspace.sty" thesis/
cp "SCSLP__Copy_/pengesahanskripsi.pdf" thesis/attachments/
cp "SCSLP__Copy_/pernyataan.pdf" thesis/attachments/
cp -r "SCSLP__Copy_/Gambar FHO" thesis/figures/
cp "SCSLP__Copy_/"*.png "SCSLP__Copy_/"*.jpg thesis/figures/ 2>/dev/null || true
```

- [ ] **Step 3: Move chapters to `thesis/chapters/`**

```bash
cp "SCSLP__Copy_/Bab1.tex" thesis/chapters/bab1_pendahuluan.tex
cp "SCSLP__Copy_/Bab2.tex" thesis/chapters/bab2_landasan_teori.tex
cp "SCSLP__Copy_/Bab3.tex" thesis/chapters/bab3_optimisasi_stokastik.tex
cp "SCSLP__Copy_/Bab4.tex" thesis/chapters/bab4_pemodelan_simulasi.tex
cp "SCSLP__Copy_/Bab5.tex" thesis/chapters/bab5_penutup.tex
cp "SCSLP__Copy_/Skripsi.tex" thesis/main.tex
```

- [ ] **Step 4: Create `.latexmkrc` and `Makefile`**

Create `thesis/.latexmkrc`:
```perl
$pdf_mode = 1;
$out_dir = 'build';
$bibtex_use = 2;
$latex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
```

Create `thesis/Makefile`:
```makefile
.PHONY: all pdf clean view

all: pdf

pdf:
	latexmk -pdf main.tex

clean:
	latexmk -C
	rm -rf build

view:
	open build/main.pdf || xdg-open build/main.pdf
```

- [ ] **Step 5: Update `thesis/main.tex` paths**

Update `main.tex`:
- Set graphicspath: `\graphicspath{{figures/}{figures/Gambar FHO/}{attachments/}}`
- Update inputs:
  - `\input{chapters/bab1_pendahuluan.tex}`
  - `\input{chapters/bab2_landasan_teori.tex}`
  - `\input{chapters/bab3_optimisasi_stokastik.tex}`
  - `\input{chapters/bab4_pemodelan_simulasi.tex}`
  - `\input{chapters/bab5_penutup.tex}`
- Update listings to point to `../code/src/scslp/generator.py`, `../code/src/scslp/problem.py`, `../code/src/scslp/fho.py`.

- [ ] **Step 6: Verify compilation into `thesis/build/main.pdf`**

Run:
```bash
cd thesis && make pdf
```
Expected: `build/main.pdf` created, exits with code 0.

- [ ] **Step 7: Clean legacy `SCSLP__Copy_/`**

Remove `SCSLP__Copy_/` now that everything is migrated into `thesis/`, `code/`, `data/`, with original workspace preserved in initial git commit.
```bash
git rm -r SCSLP__Copy_
```

- [ ] **Step 8: Commit**

```bash
git add thesis/
git commit -m "refactor(thesis): reorganize LaTeX workspace into chapters, figures, and out-of-source build"
```

---

### Task 9: Thesis Editorial Polish & Appendix Code Harmonization

**Files:**
- Modify: `thesis/main.tex`
- Modify: `thesis/chapters/bab4_pemodelan_simulasi.tex`
- Modify: `thesis/chapters/bab5_penutup.tex`

- [ ] **Step 1: Audit spelling and terminology in Chapter 4 & 5**
  - Replace "algortima" with "algoritma".
  - Replace "membbuat" with "membuat".
  - Replace "genetia" with "genetika".
  - Ensure consistent italicization of English terms (*Fire Hawk Optimizer*, *electric bike-sharing*, *two-stage stochastic*).

- [ ] **Step 2: Update code listings in `thesis/main.tex`**
  - Verify that `\lstinputlisting` statements cleanly format Python syntax and reference the refactored code modules.

- [ ] **Step 3: Compile and verify clean build**

Run:
```bash
cd thesis && make clean && make pdf
```
Expected: `build/main.pdf` generated without error, verified page count.

- [ ] **Step 4: Commit**

```bash
git add thesis/
git commit -m "docs(thesis): polish typos, mathematical formatting, and code listings"
```

---

### Task 10: Root Documentation & Portfolio Showcase (`README.md`)

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write publication-ready bilingual README.md**
  - Badges: Python, uv, LaTeX, UGM.
  - Project Title & Abstract (Bahasa Indonesia & English).
  - Mathematical Model summary (Two-Stage Stochastic SCSLP formulation).
  - Fire Hawk Optimizer (FHO) metaheuristic overview.
  - Repository structure diagram.
  - One-command quickstart guide (`uv run scslp-cli solve`, `cd thesis && make`).
  - Key results table (IDR 274M net profit, 12 stations, 55 bikes, 80 trips).
  - Citation / bibtex entry for the thesis.

- [ ] **Step 2: Run simulation to generate live figures for README**

Run:
```bash
uv run --project code python code/scripts/run_simulation.py
```
Copy `results/convergence.png` and map assets as needed to showcase.

- [ ] **Step 3: Verify git cleanliness**

Run:
```bash
git status
```
Expected: Clean working tree.

- [ ] **Step 4: Commit**

```bash
git add README.md results/
git commit -m "docs: add publication-grade bilingual README with math formulation and results"
```
