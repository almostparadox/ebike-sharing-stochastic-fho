from pathlib import Path

from scslp.config import ProblemConfig  # type: ignore
from scslp.generator import generate_synthetic_data, load_datasets, save_datasets  # type: ignore


def test_generate_synthetic_data():
    cfg = ProblemConfig(num_stations=15)
    stations_df, scenarios_df, dist_matrix = generate_synthetic_data(cfg, seed=42)

    assert len(stations_df) == 15
    assert set(stations_df.columns) >= {
        "station_id",
        "name",
        "latitude",
        "longitude",
        "capacity",
        "opening_cost",
    }
    assert len(scenarios_df) > 0
    assert set(scenarios_df.columns) >= {
        "scenario",
        "trip_id",
        "origin",
        "destination",
        "profit",
        "distance",
    }
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
