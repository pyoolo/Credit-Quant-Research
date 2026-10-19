"""Track-1 end-to-end pipeline: load panel -> features -> fit -> score -> persist.

Orchestrates the cross-section track. Returns the fitted model, the scored
latest snapshot, and the backtest report. Persists ``rv_track1.csv``, the fitted
model artefact and the company snapshot.
"""

from __future__ import annotations

from crv.config import paths
from crv.tracks.cross_section import artifacts, backtest, features, fundamentals, model, signal
from crv.utils.io import load_track_cache
from crv.utils.logging import get_logger

log = get_logger(__name__)


def run(run_backtest: bool = True) -> dict:
    """Run the full Track-1 pipeline and return its key artefacts."""
    panel = load_track_cache(paths.CACHE_XS)
    if panel is None:
        raise FileNotFoundError(
            f"Track-1 panel not found at {paths.CACHE_XS}. Run `crv fetch` first.")

    fund = fundamentals.build_fundamentals(paths.FINANCIALS)
    panel = features.add_structural_features(panel)
    panel = features.attach_fundamentals(panel, fund)
    panel = features.apply_near_maturity_filter(panel)
    panel, fund_params = features.build_fundamental_scores(panel)

    fitted, train = model.fit(panel)
    live, snapshot = signal.score(panel, fitted)
    snapshot.to_csv(paths.RV_T1, index=False)
    log.info("rv_track1.csv saved (%d bonds)", len(snapshot))

    artifacts.save_model(fitted, train, live, fund_params)
    artifacts.save_company_snapshot(panel)

    report = backtest.run(panel, fitted, train) if run_backtest else None
    return {"fitted": fitted, "snapshot": snapshot, "live": live,
            "train": train, "backtest": report}
