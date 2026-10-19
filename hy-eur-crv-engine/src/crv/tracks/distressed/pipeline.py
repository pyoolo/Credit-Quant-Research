"""Track-3 end-to-end pipeline: load panel -> fetch -> levels -> persist."""

from __future__ import annotations

from crv.config import paths
from crv.tracks.distressed import artifacts, fetch, signal
from crv.utils.io import load_track_cache
from crv.utils.logging import get_logger

log = get_logger(__name__)


def run() -> dict:
    """Run the full Track-3 pipeline and return its key artefacts."""
    panel_ds = load_track_cache(paths.CACHE_DS)
    if panel_ds is None:
        raise FileNotFoundError(
            f"Track-3 panel not found at {paths.CACHE_DS}. Run `crv fetch` first.")

    panel_ds_long, ds_static = fetch.fetch_all(panel_ds)
    ds_signals, l3_traj, static_px = signal.run(panel_ds_long, ds_static)
    artifacts.save(ds_signals, ds_static, panel_ds_long)
    return {"signals": ds_signals, "trajectory": l3_traj,
            "static": static_px, "long": panel_ds_long}
