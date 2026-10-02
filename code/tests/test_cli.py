from pathlib import Path

import pytest

from scslp.cli import cli_generate, cli_solve, main  # type: ignore


def test_cli_generate_and_solve(tmp_path: Path) -> None:
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
    assert (results_dir / "solution_map.html").exists()


def test_main_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data_main"
    monkeypatch.setattr("sys.argv", ["scslp-cli", "generate", "--outdir", str(data_dir), "--seed", "42"])
    main()
    assert (data_dir / "stations_data.csv").exists()
