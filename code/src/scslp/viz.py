from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import folium
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_folium_map(
    stations_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    solution_dict: dict[str, Any] | None = None,
) -> folium.Map:
    """Create interactive Folium map showing candidate stations and trip trajectories."""
    lats = np.asarray(stations_df["latitude"], dtype=float)
    lons = np.asarray(stations_df["longitude"], dtype=float)
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    y = solution_dict.get("y") if solution_dict else None

    # Add station markers
    station_records: list[dict[str, Any]] = stations_df.to_dict(orient="records")
    for idx, row in enumerate(station_records):
        is_built = bool(y[idx] == 1) if y is not None else True
        color = "green" if is_built else "gray"
        icon = "bicycle" if is_built else "info-sign"

        status_text = "STATUS: BUILT" if is_built else "STATUS: NOT BUILT"
        st_name = str(row["name"])
        st_id = int(row["station_id"])
        cap = int(row["capacity"])
        cost = int(row["opening_cost"])
        st_lat = float(row["latitude"])
        st_lon = float(row["longitude"])

        popup_html = (
            f"<b>{st_name}</b> (ID: {st_id})<br>"
            f"{status_text}<br>"
            f"Capacity: {cap} bikes<br>"
            f"Opening Cost: IDR {cost:,}"
        )
        folium.Marker(
            location=[st_lat, st_lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    # Add trip polylines
    trip_records: list[dict[str, Any]] = scenarios_df.head(40).to_dict(orient="records")
    for trip in trip_records:
        raw_orig = trip["origin"]
        raw_dest = trip["destination"]
        orig = ast.literal_eval(raw_orig) if isinstance(raw_orig, str) else raw_orig
        dest = ast.literal_eval(raw_dest) if isinstance(raw_dest, str) else raw_dest
        sc_id = trip["scenario"]
        t_id = trip["trip_id"]
        folium.PolyLine(
            locations=[orig, dest],
            color="#FF5733",
            weight=1.5,
            opacity=0.7,
            popup=f"Scenario: {sc_id}, Trip: {t_id}",
        ).add_to(m)

    return m


def plot_convergence(
    fitness_history: list[float],
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
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
