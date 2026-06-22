"""Track 3 — distressed PD / recovery frontier (the ``ds`` panel).

Bonds wider than the distress cutoff trade on default probability and recovery,
not on a spread curve, so they are assessed through four complementary lenses:

    L1  price-implied default — a constant-hazard model solved for the implied
        default intensity at fixed recovery scenarios; reports the max implied
        recovery where the price sits below any plausible recovery
    L2  capital-structure coherence — on multi-bond issuers, implied
        recovery/PD should respect seniority; flags violations
    L3  trajectory / regime — recent price trend + acceleration -> regime label
        (deteriorating / stabilising / recovering) with a confidence weight
    L4  composite — a judgement score in [-1, +1] with an uncertainty band that
        widens where data is missing or the trajectory is uncertain

The output is explicitly a judgement with a lo-hi band, not a precise fair value.

Modules:

    ``solver``   — constant-hazard PV, implied lambda, PD frontier (shared by L1 & save)
    ``fetch``    — distressed static + long price/spread history
    ``levels``   — L1-L4 model levels
    ``signal``   — orchestrates the four levels into ds_signals
    ``pipeline`` — fetch -> levels -> persist (rv_track3.csv + params + trajectory)
"""

from crv.tracks.distressed import levels, signal, solver

__all__ = ["levels", "signal", "solver"]
