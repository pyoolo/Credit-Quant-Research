"""Batch pricing, scenario grids and driver-sensitivity sweeps."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from crv.pricing.engine import Artefacts, route_and_price


def batch_price(candidates: list[dict], art: Artefacts) -> pd.DataFrame:
    """Price a list of specs into a tidy comparison table."""
    rows = []
    for c in candidates:
        r = route_and_price(c, art)
        rows.append({
            "name": c.get("name", ""),
            "track": r.get("track"),
            "rating": c.get("rating_bucket", ""),
            "coupon": c.get("coupon"),
            "dur": c.get("duration"),
            "fair": round(r["fair"]) if r.get("fair") is not None else None,
            "pd40": round(r["pd_40"] * 100) if r.get("pd_40") is not None else None,
            "proposed": c.get("proposed_zs"),
            "gap": round(r["gap"]) if r.get("gap") is not None else None,
            "verdict": r.get("verdict", ""),
            "flags": "; ".join(r.get("flags", [])),
        })
    return pd.DataFrame(rows)


def scenario_grid(base: dict, grid: dict, art: Artefacts) -> pd.DataFrame:
    """Price every combination of the varied drivers in ``grid``."""
    keys = list(grid.keys())
    rows = []
    for combo in itertools.product(*grid.values()):
        spec = dict(base)
        for k, v in zip(keys, combo, strict=False):
            spec[k] = v
        r = route_and_price(spec, art)
        row = {k: spec[k] for k in keys}
        row["track"] = r.get("track")
        row["fair"] = round(r["fair"]) if r.get("fair") is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def sweep(base: dict, feat: str, lo: float, hi: float, n: int,
          art: Artefacts) -> tuple[np.ndarray, np.ndarray]:
    """Sweep one driver over ``[lo, hi]`` and return ``(xs, fair_values)``."""
    xs = np.linspace(lo, hi, n)
    ys = []
    for v in xs:
        spec = dict(base)
        spec[feat] = float(v)
        ys.append(route_and_price(spec, art).get("fair", np.nan))
    return xs, np.array(ys, float)
