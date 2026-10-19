"""End-to-end data pipeline: fetch -> sanity -> route -> save 3 tracks.

This is the single entry point behind ``crv fetch`` on the CLI. It produces the
three disjoint panels each track consumes, persisting each both as a pickle
cache (dtypes preserved) and as a CSV.
"""

from __future__ import annotations

from crv.config import paths, settings
from crv.data import fetch, router, sanity
from crv.utils.io import save_track_cache
from crv.utils.logging import get_logger

log = get_logger(__name__)


def run(verbose_sanity: bool = True) -> router.RoutedPanels:
    """Run the full data pipeline and return the three routed panels."""
    df_panel, _static = fetch.fetch_panel()

    if verbose_sanity:
        sanity.sanity_check(df_panel)

    panels = router.route(df_panel)
    latest = str(df_panel["date"].max())

    save_track_cache(panels.cross_section, "XS_cross_section",
                     paths.CACHE_XS, paths.CSV_XS, settings.PANEL_WEEKS, latest)
    save_track_cache(panels.self_ref, "TS_self_ref",
                     paths.CACHE_TS, paths.CSV_TS, settings.PANEL_WEEKS, latest)
    save_track_cache(panels.distressed, "DS_distressed",
                     paths.CACHE_DS, paths.CSV_DS, settings.PANEL_WEEKS, latest)

    log.info("pipeline complete: 3 disjoint panels saved")
    return panels
