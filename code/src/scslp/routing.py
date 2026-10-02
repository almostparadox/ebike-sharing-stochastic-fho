from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _parse_coord(raw: Any) -> tuple[float, float]:
    """Parse coordinate from tuple or string representation."""
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return (_safe_float(raw[0]), _safe_float(raw[1]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (tuple, list)) and len(parsed) == 2:
                return (_safe_float(parsed[0]), _safe_float(parsed[1]))
        except (ValueError, SyntaxError):
            pass
    return (0.0, 0.0)


def load_routes_cache(cache_path: str | Path = "data/routes_cache.json") -> dict[str, list[tuple[float, float]]]:
    """Load cached street routes from json file."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw_cache = json.load(f)
        cache: dict[str, list[tuple[float, float]]] = {}
        for k, v in raw_cache.items():
            cache[k] = [(_safe_float(pt[0]), _safe_float(pt[1])) for pt in v]
        return cache
    except (OSError, json.JSONDecodeError):
        return {}


def save_routes_cache(
    cache: dict[str, list[tuple[float, float]]], cache_path: str | Path = "data/routes_cache.json"
) -> None:
    """Save street routes cache to json file."""
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def _linear_fallback(
    orig_coord: tuple[float, float], dest_coord: tuple[float, float], n_points: int = 5
) -> list[tuple[float, float]]:
    """Generate linearly interpolated fallback coordinates."""
    lats = np.linspace(orig_coord[0], dest_coord[0], n_points)
    lons = np.linspace(orig_coord[1], dest_coord[1], n_points)
    return [(_safe_float(lat), _safe_float(lon)) for lat, lon in zip(lats, lons, strict=True)]


def fetch_bike_route(
    orig_coord: tuple[float, float],
    dest_coord: tuple[float, float],
    cache: dict[str, list[tuple[float, float]]] | None = None,
) -> list[tuple[float, float]]:
    """Fetch realistic bike route between two coordinates via public OSRM API with caching and fallback."""
    cache_key = f"{orig_coord}_{dest_coord}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    lat1, lon1 = orig_coord
    lat2, lon2 = dest_coord

    # OSRM expects {lon},{lat} format in query string
    url = f"http://router.project-osrm.org/route/v1/bike/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    if not url.startswith(("http://", "https://")):
        return _linear_fallback(orig_coord, dest_coord, n_points=5)

    req = urllib.request.Request(url, headers={"User-Agent": "ThesisResearch/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
        if "routes" in data and len(data["routes"]) > 0:
            raw_coords = data["routes"][0]["geometry"]["coordinates"]
            # Convert geojson [lon, lat] to (lat, lon)
            route: list[tuple[float, float]] = [
                (_safe_float(pt[1]), _safe_float(pt[0])) for pt in raw_coords
            ]
            if cache is not None:
                cache[cache_key] = route
            return route
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, KeyError):
        pass

    fallback_route = _linear_fallback(orig_coord, dest_coord, n_points=5)
    if cache is not None:
        cache[cache_key] = fallback_route
    return fallback_route


def get_all_scenario_routes(
    scenarios_df: pd.DataFrame, cache_path: str = "data/routes_cache.json"
) -> dict[str, list[tuple[float, float]]]:
    """Retrieve or compute street routes for all trips across scenarios in scenarios_df."""
    cache = load_routes_cache(cache_path)
    records: list[dict[str, Any]] = scenarios_df.to_dict(orient="records")

    for row in records:
        orig = _parse_coord(row["origin"])
        dest = _parse_coord(row["destination"])
        fetch_bike_route(orig, dest, cache=cache)

    save_routes_cache(cache, cache_path)
    return cache
