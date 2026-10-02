from __future__ import annotations

from pathlib import Path
import random

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

    # 1. Generate candidate stations
    stations = []
    for i in range(config.num_stations):
        meta = YOGYAKARTA_STATIONS[i % len(YOGYAKARTA_STATIONS)]
        capacity = rng.randint(5, 10)
        opening_cost = rng.randint(10_000_000, 50_000_000)
        stations.append(
            {
                "station_id": i,
                "name": meta["name"],
                "latitude": meta["lat"],
                "longitude": meta["lon"],
                "capacity": capacity,
                "opening_cost": opening_cost,
            }
        )
    stations_df = pd.DataFrame(stations)

    # 2. Distance matrix
    n_st = len(stations_df)
    station_ids = list(range(n_st))
    dist_matrix = pd.DataFrame(
        np.zeros((n_st, n_st)),
        index=station_ids,
        columns=station_ids,
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
                walk_orig = geodesic(
                    orig_coord,
                    (stations_df.loc[orig_st, "latitude"], stations_df.loc[orig_st, "longitude"]),
                ).kilometers
                walk_dest = geodesic(
                    dest_coord,
                    (stations_df.loc[dest_st, "latitude"], stations_df.loc[dest_st, "longitude"]),
                ).kilometers
                time_orig = (walk_orig / config.walking_speed_kmph) * 60
                time_dest = (walk_dest / config.walking_speed_kmph) * 60

                if time_orig <= config.max_walking_time and time_dest <= config.max_walking_time:
                    sk = rng.randint(0, max(0, config.t_max - config.max_trip_duration - 1))
                    ek = rng.randint(sk + 1, min(config.t_max, sk + config.max_trip_duration))
                    duration = ek - sk
                    profit = config.trip_profit_multiplier * duration
                    battery_cons = min(1.0, trip_dist / config.max_battery_distance)

                    scenarios.append(
                        {
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
                        }
                    )
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


def load_datasets(
    data_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ProblemConfig]:
    """Load canonical datasets from directory."""
    path = Path(data_dir)
    stations_df = pd.read_csv(path / "stations_data.csv")
    scenarios_df = pd.read_csv(path / "scenarios_data.csv")
    dist_matrix_df = pd.read_csv(path / "distance_matrix.csv", index_col=0)
    # Ensure index and columns are integer station IDs
    dist_matrix_df.index = dist_matrix_df.index.astype(int)
    dist_matrix_df.columns = dist_matrix_df.columns.astype(int)
    config = ProblemConfig.from_json(path / "config.json")
    return stations_df, scenarios_df, dist_matrix_df, config
