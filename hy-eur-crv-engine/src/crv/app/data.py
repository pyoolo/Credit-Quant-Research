"""UI-agnostic loaders that assemble the monitor's per-track bond tables.

Reads the three ``rv_track*.csv`` outputs, attaches fundamentals where missing,
and normalises heterogeneous column names across tracks into one schema the UI
can render uniformly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import pandas as pd

from crv.config import paths
from crv.tracks.cross_section.fundamentals import build_fundamentals
from crv.utils.logging import get_logger

log = get_logger(__name__)

_TRACK_FILES = {1: paths.RV_T1, 2: paths.RV_T2, 3: paths.RV_T3}
_TRACK_NAMES = {1: "Track 1 — Cross-section", 2: "Track 2 — FRN / Perp", 3: "Track 3 — Distressed"}


@dataclass
class MonitorData:
    """The full payload the monitor view renders."""
    as_of: str
    tracks: dict = field(default_factory=dict)         # track_no -> {name, bonds}
    trajectories: dict = field(default_factory=dict)   # isin -> [[date, px], ...]


def _fund_min() -> pd.DataFrame:
    """Minimal fundamentals (net_lev/icr/ebitda_margin) for monitor enrichment."""
    try:
        f = build_fundamentals(paths.FINANCIALS)
        return f[["isin", "net_lev", "icr", "ebitda_margin"]].drop_duplicates("isin")
    except Exception as exc:  # noqa: BLE001 - financials optional for the monitor
        log.warning("fundamentals unavailable: %s", str(exc)[:60])
        return pd.DataFrame(columns=["isin", "net_lev", "icr", "ebitda_margin"])


def _num(x, d: int = 2):
    try:
        if x is None or pd.isna(x):
            return None
        return round(float(x), d)
    except Exception:  # noqa: BLE001
        return None


def _first_col(df: pd.DataFrame, *names):
    return next((n for n in names if n in df.columns), None)


def build_bonds(path, track: int, fund: pd.DataFrame) -> list[dict]:
    """Load one track's RV CSV into a list of normalised bond dicts."""
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if not {"net_lev", "icr", "ebitda_margin"}.issubset(df.columns):
        df = df.merge(fund, on="isin", how="left", suffixes=("", "_f"))
    c_alpha = _first_col(df, "rv_z", "rv_z_4w", "alpha_4w", "ds_score")
    c_z = _first_col(df, "z_spread", "zspread")
    c_fair = _first_col(df, "fair_zs")
    c_resid = _first_col(df, "resid_bps")
    c_sector = _first_col(df, "sector", "industry")

    out = []
    for _, r in df.iterrows():
        out.append(dict(
            isin=str(r.get("isin", "")),
            issuer=str(r.get("name", ""))[:34],
            sector=str(r.get(c_sector, "")) if c_sector else "",
            rating=str(r.get("rating_bucket", "")),
            track=track,
            zspread=_num(r.get(c_z), 0) if c_z else None,
            fair=_num(r.get(c_fair), 0) if (c_fair and track in (1, 2)) else None,
            resid=_num(r.get(c_resid), 0) if (c_resid and track in (1, 2)) else None,
            alpha=_num(r.get(c_alpha), 2) if c_alpha else None,
            dur=_num(r.get("duration"), 1),
            lev=_num(r.get("net_lev"), 1),
            cov=_num(r.get("icr"), 1),
            mgn=_num(r.get("ebitda_margin"), 4),
            regime=str(r.get("regime", "")) if "regime" in df.columns else "",
            pd40=_num(r.get("pd_40"), 2) if "pd_40" in df.columns else None,
            conf=_num(r.get("ds_conf"), 2) if "ds_conf" in df.columns else None,
            frontier=str(r.get("frontier_json", "[]")) if "frontier_json" in df.columns else "[]",
        ))
    return out


def load_monitor() -> MonitorData:
    """Assemble the full monitor payload across all three tracks."""
    fund = _fund_min()
    tracks = {}
    for k, fn in _TRACK_FILES.items():
        bonds = build_bonds(fn, k, fund)
        tracks[k] = {"name": _TRACK_NAMES[k], "bonds": bonds}
        log.info("%s: %d bonds", _TRACK_NAMES[k], len(bonds))

    try:
        import datetime as _dt
        as_of = _dt.date.fromtimestamp(os.path.getmtime(paths.RV_T1)).isoformat()
    except Exception:  # noqa: BLE001
        as_of = str(pd.Timestamp.today().date())

    traj = {}
    if os.path.exists(paths.T3_TRAJ):
        tj = pd.read_csv(paths.T3_TRAJ)
        if {"isin", "date", "px"}.issubset(tj.columns):
            tj["date"] = pd.to_datetime(tj["date"]).dt.strftime("%Y-%m-%d")
            for i, g in tj.sort_values("date").groupby("isin"):
                traj[str(i)] = [[d, float(p)] for d, p in zip(g["date"], g["px"], strict=False)]

    return MonitorData(as_of=as_of, tracks=tracks, trajectories=traj)


def bonds_dataframe(track_payload: dict) -> pd.DataFrame:
    """Convert a track payload's bond list into a display DataFrame."""
    return pd.DataFrame(track_payload.get("bonds", []))


def parse_frontier(frontier_json: str) -> pd.DataFrame:
    """Parse a stored ``frontier_json`` string into an (R, cum_pd) frame."""
    try:
        rows = json.loads(frontier_json)
    except Exception:  # noqa: BLE001
        rows = []
    return pd.DataFrame(rows, columns=["recovery", "cum_pd"]) if rows else pd.DataFrame(
        columns=["recovery", "cum_pd"])
