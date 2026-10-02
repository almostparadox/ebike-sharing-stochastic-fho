from pathlib import Path

import folium
import matplotlib.figure
import numpy as np

from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data
from scslp.viz import create_folium_map, plot_convergence


def test_create_folium_map(tmp_path: Path):
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, _ = generate_synthetic_data(cfg, seed=42)

    # Test without solution_dict
    m = create_folium_map(st_df, sc_df)
    assert isinstance(m, folium.Map)
    out_html = tmp_path / "test_map.html"
    m.save(str(out_html))
    assert out_html.exists()

    # Test with solution_dict
    y = np.zeros(15, dtype=int)
    y[:5] = 1
    m_sol = create_folium_map(st_df, sc_df, solution_dict={"y": y})
    assert isinstance(m_sol, folium.Map)
    out_sol_html = tmp_path / "test_map_sol.html"
    m_sol.save(str(out_sol_html))
    assert out_sol_html.exists()


def test_plot_convergence(tmp_path: Path):
    history = [100.0, 250.0, 400.0, 500.0]
    out_png = tmp_path / "convergence.png"
    fig = plot_convergence(history, output_path=out_png)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert out_png.exists()
