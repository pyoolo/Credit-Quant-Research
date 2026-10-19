"""Shared data layer: one Bloomberg fetch -> raw panel -> 3-track router.

    ``crv.data.universe``   — Bloomberg universe + field requests
    ``crv.data.fetch``      — assemble the weekly raw panel (cache-aware)
    ``crv.data.sanity``     — read-only data-quality diagnostics
    ``crv.data.router``     — split the corporate panel into xs / ts / ds tracks
    ``crv.data.ratings``    — rating-bucket / notch derivation
"""

from crv.data import fetch, ratings, router, sanity, universe

__all__ = ["fetch", "ratings", "router", "sanity", "universe"]
