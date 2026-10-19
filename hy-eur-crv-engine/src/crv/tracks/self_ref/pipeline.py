"""Track-2 end-to-end pipeline: load panel -> fetch -> price -> model -> persist."""

from __future__ import annotations

from crv.config import paths
from crv.tracks.self_ref import artifacts, fetch, model, pricing
from crv.utils.io import load_track_cache
from crv.utils.logging import get_logger

log = get_logger(__name__)


def run() -> dict:
    """Run the full Track-2 pipeline and return its key artefacts."""
    panel_ts = load_track_cache(paths.CACHE_TS)
    if panel_ts is None:
        raise FileNotFoundError(
            f"Track-2 panel not found at {paths.CACHE_TS}. Run `crv fetch` first.")

    panel_ts = pricing.split_families(panel_ts)
    long_bonds, eur_curve, call_sched, perp_freq, frn_dm = fetch.fetch_all(panel_ts)
    long_bonds = pricing.split_families(long_bonds)

    panel_ts_priced = pricing.price(long_bonds, frn_dm, eur_curve, call_sched, perp_freq)
    rv_ts, rv_last = model.run(panel_ts_priced)
    artifacts.save(rv_ts, panel_ts_priced)

    return {"rv": rv_ts, "rv_last": rv_last, "priced": panel_ts_priced}
