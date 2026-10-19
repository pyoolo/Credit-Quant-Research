"""New-issue pricing engine.

Consumes the fitted track artefacts to price hypothetical / new-issue bonds. A
router classifies each spec into a track and dispatches to the matching pricer:

    Track 1  fair-spread Lasso (raw fundamental ratios -> fund scores -> fair Z)
    Track 2  per-family peer-distribution placement (FRN DM / PERP STW)
    Track 3  price-implied default-probability scenarios

On top of the single-bond pricer sit batch pricing, a scenario grid and a
driver-sensitivity sweep.

    ``engine``       — load artefacts, the three pricers, the router
    ``analytics``    — batch / scenario-grid / sensitivity helpers
"""

from crv.pricing import analytics, engine

__all__ = ["analytics", "engine"]
