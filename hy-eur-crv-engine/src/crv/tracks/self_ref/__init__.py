"""Track 2 — structural / self-referential pricer (the ``ts`` panel).

Floating-rate and perpetual bonds cannot be priced by the fixed-coupon
cross-section, so each is priced on its own terms:

    FRN   -> Bloomberg discount margin (fallback: z-spread)
    PERP  -> spread-to-worst over the future call schedule (yield-to-call vs the
             EUR swap curve), worst-of the first ``MAX_CALLS`` calls

The structural spread is then assessed two ways: a self-referential z-score
within each bond's own history, and a hybrid cross-section fair-value model
(Huber or gradient-boosting, chosen by CV) pooled over a rolling window, whose
residual (``alpha``) becomes a robust MAD-based RV score. The self-ref z-score
is the fallback wherever the cross-section can't score a bond.

Modules:

    ``fetch``    — long history + EUR curve + call schedule + FRN discount margin
    ``pricing``  — structural spread (FRN DM / PERP YTC-worst) + self-ref z-score
    ``model``    — rolling hybrid cross-section fair value -> alpha -> RV score
    ``pipeline`` — orchestrates fetch -> price -> model -> persist
"""

from crv.tracks.self_ref import model, pricing

__all__ = ["model", "pricing"]
