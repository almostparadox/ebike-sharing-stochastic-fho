from pathlib import Path

from scslp.config import FHOConfig, ProblemConfig, Station, Trip  # type: ignore


def test_default_problem_config() -> None:
    cfg = ProblemConfig()
    assert cfg.num_stations == 15
    assert cfg.budget_fraction == 0.8
    assert cfg.bike_cost == 4_000_000.0
    assert cfg.scenario_probabilities == [0.2, 0.3, 0.5]
    assert len(cfg.scenario_probabilities) == 3


def test_default_fho_config() -> None:
    fho_cfg = FHOConfig(pop_size=50, max_generations=100, random_seed=42)
    assert fho_cfg.pop_size == 50
    assert fho_cfg.max_generations == 100
    assert fho_cfg.random_seed == 42


def test_station_and_trip_dataclasses() -> None:
    st = Station(
        station_id=1,
        name="Tugu",
        latitude=-7.7829,
        longitude=110.3671,
        capacity=10,
        opening_cost=25_000_000.0,
    )
    assert st.name == "Tugu"
    assert st.capacity == 10

    tr = Trip(
        scenario=0,
        trip_id=101,
        origin=(-7.78, 110.36),
        destination=(-7.79, 110.37),
        start_time=2,
        end_time=5,
        profit=30_000_000.0,
        distance=1.5,
        battery_consumption=0.075,
        probability=0.2,
    )
    assert tr.trip_id == 101
    assert tr.profit == 30_000_000.0


def test_problem_config_json_roundtrip(tmp_path: Path) -> None:
    cfg = ProblemConfig(num_stations=12, budget_fraction=0.75)
    file_path = tmp_path / "cfg.json"
    cfg.save_json(file_path)
    loaded = ProblemConfig.from_json(file_path)
    assert loaded.num_stations == 12
    assert loaded.budget_fraction == 0.75
