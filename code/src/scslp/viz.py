from __future__ import annotations

import ast
import io
import math
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import folium
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val: Any) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = _safe_int((lon_deg + 180.0) / 360.0 * n)
    ytile = _safe_int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def _num2deg(xtile: int, ytile: int, zoom: int) -> tuple[float, float]:
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def _fetch_stitched_osm_tiles(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, zoom: int = 13
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    """Fetch and stitch OpenStreetMap tiles for a given bounding box."""
    x_min, y_min = _deg2num(lat_max, lon_min, zoom)
    x_max, y_max = _deg2num(lat_min, lon_max, zoom)

    tiles: list[list[Image.Image]] = []
    headers = {"User-Agent": "ThesisMapVisualizer/1.0"}

    for y in range(y_min, y_max + 1):
        row: list[Image.Image] = []
        for x in range(x_min, x_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            if not url.startswith(("http://", "https://")):
                tile = Image.new("RGB", (256, 256), (242, 242, 242))
            else:
                req = urllib.request.Request(url, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:  # noqa: S310 # nosec B310
                        tile = Image.open(io.BytesIO(resp.read()))
                except (urllib.error.URLError, TimeoutError, OSError):
                    tile = Image.new("RGB", (256, 256), (242, 242, 242))
            row.append(tile)
        tiles.append(row)

    if not tiles or not tiles[0]:
        return None

    width = len(tiles[0]) * 256
    height = len(tiles) * 256
    stitched = Image.new("RGB", (width, height))

    for r_idx, r in enumerate(tiles):
        for c_idx, t in enumerate(r):
            stitched.paste(t, (c_idx * 256, r_idx * 256))

    top_left_lat, top_left_lon = _num2deg(x_min, y_min, zoom)
    bottom_right_lat, bottom_right_lon = _num2deg(x_max + 1, y_max + 1, zoom)
    extent = (top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat)
    return (stitched, extent)


def create_folium_map(
    stations_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    solution_dict: dict[str, Any] | None = None,
    routes_cache: dict[str, list[tuple[float, float]]] | None = None,
) -> folium.Map:
    """Create interactive Folium map showing stations and real street trip trajectories."""
    lats = np.asarray(stations_df["latitude"], dtype=float)
    lons = np.asarray(stations_df["longitude"], dtype=float)
    center_lat = _safe_float(np.mean(lats))
    center_lon = _safe_float(np.mean(lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    y = solution_dict.get("y") if solution_dict else None

    # Add station markers
    station_records: list[dict[str, Any]] = stations_df.to_dict(orient="records")
    for idx, row in enumerate(station_records):
        is_built = bool(y[idx] == 1) if y is not None else True
        color = "green" if is_built else "gray"
        icon = "bicycle" if is_built else "info-sign"

        status_text = "STATUS: TERBANGUN" if is_built else "STATUS: TIDAK DIBANGUN"
        st_name = str(row["name"])
        st_id = _safe_int(row["station_id"])
        cap = _safe_int(row["capacity"])
        cost = _safe_int(row["opening_cost"])
        st_lat = _safe_float(row["latitude"])
        st_lon = _safe_float(row["longitude"])

        popup_html = (
            f"<b>{st_name}</b> (ID: {st_id})<br>"
            f"{status_text}<br>"
            f"Kapasitas: {cap} sepeda<br>"
            f"Biaya Pembangunan: IDR {cost:,}"
        )
        folium.Marker(
            location=[st_lat, st_lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(m)

    # Add trip polylines with real street route coordinates when available
    trip_records: list[dict[str, Any]] = scenarios_df.head(45).to_dict(orient="records")
    for trip in trip_records:
        raw_orig = trip["origin"]
        raw_dest = trip["destination"]
        orig = ast.literal_eval(raw_orig) if isinstance(raw_orig, str) else raw_orig
        dest = ast.literal_eval(raw_dest) if isinstance(raw_dest, str) else raw_dest
        sc_id = trip["scenario"]
        t_id = trip["trip_id"]

        route_key = f"{orig}_{dest}"
        if routes_cache and route_key in routes_cache:
            polyline_points = routes_cache[route_key]
        else:
            polyline_points = [orig, dest]

        folium.PolyLine(
            locations=polyline_points,
            color="#E64A19",
            weight=2.5,
            opacity=0.75,
            popup=f"Skenario: {sc_id}, Trip: {t_id}",
        ).add_to(m)

    return m


def render_publication_map(
    stations_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    solution_dict: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    title: str = "Peta Sistem Electric Bike-Sharing di Yogyakarta",
    routes_cache: dict[str, list[tuple[float, float]]] | None = None,
) -> matplotlib.figure.Figure:
    """Render high-resolution 300-DPI publication map with real street network trajectories."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Yogyakarta bounding box
    lat_min, lat_max = -7.830, -7.740
    lon_min, lon_max = 110.315, 110.435

    # 1. Background map tiles
    tile_res = _fetch_stitched_osm_tiles(lat_min, lat_max, lon_min, lon_max, zoom=13)
    if tile_res is not None:
        img, extent = tile_res
        ax.imshow(img, extent=extent, alpha=0.65, interpolation="bilinear", zorder=1)
    else:
        ax.set_facecolor("#F4F4F6")

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # 2. Draw real street routes
    y = solution_dict.get("y") if solution_dict else None
    x = solution_dict.get("x") if solution_dict else None

    trip_records: list[dict[str, Any]] = scenarios_df.to_dict(orient="records")
    for trip in trip_records:
        sc_id = _safe_int(trip["scenario"])
        t_id = _safe_int(trip["trip_id"])

        # Check if trip is accepted in solution
        if x is not None and sc_id < len(x) and t_id < len(x[sc_id]) and x[sc_id][t_id] == 0:
            continue

        raw_orig = trip["origin"]
        raw_dest = trip["destination"]
        orig = ast.literal_eval(raw_orig) if isinstance(raw_orig, str) else raw_orig
        dest = ast.literal_eval(raw_dest) if isinstance(raw_dest, str) else raw_dest
        route_key = f"{orig}_{dest}"

        if routes_cache and route_key in routes_cache:
            coords = routes_cache[route_key]
        else:
            coords = [orig, dest]

        route_lats = [_safe_float(pt[0]) for pt in coords]
        route_lons = [_safe_float(pt[1]) for pt in coords]
        ax.plot(route_lons, route_lats, color="#E64A19", linewidth=1.4, alpha=0.55, zorder=2)

    # 3. Draw stations
    station_records: list[dict[str, Any]] = stations_df.to_dict(orient="records")
    built_lons: list[float] = []
    built_lats: list[float] = []
    unbuilt_lons: list[float] = []
    unbuilt_lats: list[float] = []

    for idx, row in enumerate(station_records):
        st_lon = _safe_float(row["longitude"])
        st_lat = _safe_float(row["latitude"])
        is_built = bool(y[idx] == 1) if y is not None else True
        if is_built:
            built_lons.append(st_lon)
            built_lats.append(st_lat)
        else:
            unbuilt_lons.append(st_lon)
            unbuilt_lats.append(st_lat)

        # Annotate major hubs
        st_name = str(row["name"])
        ax.annotate(
            st_name,
            (st_lon, st_lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold" if is_built else "normal",
            color="#1B5E20" if is_built else "#616161",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#BDBDBD",
                "linewidth": 0.6,
            },
            zorder=6,
        )

    if unbuilt_lons:
        ax.scatter(
            unbuilt_lons,
            unbuilt_lats,
            s=80,
            c="#9E9E9E",
            marker="o",
            edgecolors="white",
            linewidth=1.2,
            zorder=4,
            label="Stasiun Potensial (Tidak Dibangun)",
        )

    if built_lons:
        ax.scatter(
            built_lons,
            built_lats,
            s=110,
            c="#2E7D32",
            marker="^",
            edgecolors="white",
            linewidth=1.5,
            zorder=5,
            label="Stasiun Dibangun (Aktif)",
        )

    # Add legend line for trips
    ax.plot([], [], color="#E64A19", linewidth=2.0, alpha=0.8, label="Rute Perjalanan (Jaringan Jalan)")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Garis Bujur (Longitude)", fontsize=9.5)
    ax.set_ylabel("Garis Lintang (Latitude)", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4, color="gray", zorder=1)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92, facecolor="white", edgecolor="#BDBDBD")

    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig


def plot_convergence(
    fitness_history: list[float],
    output_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Plot optimization iteration vs. best net expected profit."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.plot(
        range(1, len(fitness_history) + 1),
        fitness_history,
        marker="o",
        markersize=3.5,
        color="#D32F2F",
        linewidth=2,
        label="Nilai Objektif Terbaik",
    )
    ax.set_title("Konvergensi Algoritma Fire Hawk Optimizer (SCSLP)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Iterasi (Generasi)", fontsize=10)
    ax.set_ylabel("Keuntungan Bersih Harapan (IDR)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)

    return fig
