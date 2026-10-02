from __future__ import annotations

from pathlib import Path

from scslp.routing import (
    _linear_fallback,
    _parse_coord,
    fetch_bike_route,
    load_routes_cache,
    save_routes_cache,
)


def test_parse_coord():
    c1 = _parse_coord("(-7.7829, 110.3671)")
    assert round(c1[0], 4) == -7.7829
    assert round(c1[1], 4) == 110.3671

    c2 = _parse_coord((-7.7707, 110.3777))
    assert round(c2[0], 4) == -7.7707
    assert round(c2[1], 4) == 110.3777

    c_invalid = _parse_coord("invalid")
    assert c_invalid == (0.0, 0.0)


def test_linear_fallback():
    orig = (-7.0, 110.0)
    dest = (-8.0, 111.0)
    pts = _linear_fallback(orig, dest, n_points=5)
    assert len(pts) == 5
    assert pts[0] == orig
    assert pts[-1] == dest


def test_cache_save_and_load(tmp_path: Path):
    cache_file = tmp_path / "routes_cache.json"
    dummy_cache = {
        "(-7.0, 110.0)_(-8.0, 111.0)": [(-7.0, 110.0), (-7.5, 110.5), (-8.0, 111.0)]
    }
    save_routes_cache(dummy_cache, cache_file)
    loaded = load_routes_cache(cache_file)
    assert "(-7.0, 110.0)_(-8.0, 111.0)" in loaded
    assert len(loaded["(-7.0, 110.0)_(-8.0, 111.0)"]) == 3


def test_fetch_bike_route_from_cache():
    orig = (-7.1, 110.1)
    dest = (-7.2, 110.2)
    key = f"{orig}_{dest}"
    cached_pts = [orig, (-7.15, 110.15), dest]
    cache = {key: cached_pts}

    result = fetch_bike_route(orig, dest, cache=cache)
    assert result == cached_pts
