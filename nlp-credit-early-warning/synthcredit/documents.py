"""Render quarterly management reports from the latent management signal.

The renderer verbalises ``mgmt`` (see world.py) as the *propensity to hedge*:
the probability that each content sentence is drawn from the hedged/negative
pool rather than the confident pool is ``sigmoid(k * mgmt)``. The feature
extractor never sees ``mgmt`` — it must recover it from the words alone.

Two deliberate traps are built in:

* RISK FACTORS grow append-only over an issuer's life and lengthen for
  everyone during stress. Length correlates with calendar time and the cycle,
  not with issuer-specific fate. A naive pipeline that scores uncertainty on
  the whole document, boilerplate included, gets poisoned.
* Section omission ("reduced disclosure") is tied to the mgmt signal, so
  *missing* text is itself informative — the coverage-degradation channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFIDENT = [
    "Trading in the period was solid and broadly ahead of plan.",
    "Liquidity remains ample, with committed facilities fully undrawn.",
    "We are comfortable with the current leverage trajectory.",
    "Cash conversion remained strong across all divisions.",
    "The order book gives us good visibility into the coming quarters.",
    "Covenant headroom remains substantial at the reporting date.",
    "We expect margin expansion to continue through the year.",
    "The refinancing completed on favourable terms and extends our maturities.",
    "Demand in our core categories remains robust.",
    "We are well positioned to fund growth from internally generated cash.",
]

HEDGED = [
    "Performance may come under pressure if current conditions persist.",
    "Liquidity could tighten depending on working capital developments.",
    "Covenant headroom has reduced and remains subject to trading performance.",
    "Demand weakened in several categories and visibility is limited.",
    "We might need to seek an amendment should conditions deteriorate further.",
    "Margins declined and the recovery remains uncertain.",
    "Refinancing options depend on market conditions, which remain volatile.",
    "The outlook is cautious and subject to significant uncertainty.",
    "Cash generation deteriorated relative to the prior period.",
    "Certain suppliers have tightened terms, which could affect liquidity.",
]

LIQUIDITY_DETAIL = [
    "As of the reporting date the revolving credit facility was undrawn.",
    "Cash and equivalents are held predominantly at investment grade banks.",
    "The maturity profile has no significant amortisation before 2028.",
    "Headroom under the springing covenant stood above the test threshold.",
]

BOILERPLATE = [
    "We operate in a highly competitive industry.",
    "Our business is exposed to macroeconomic conditions.",
    "We are subject to extensive regulation in the jurisdictions where we operate.",
    "Fluctuations in raw material prices may affect our results.",
    "We depend on key personnel and skilled employees.",
    "Our substantial indebtedness could adversely affect our financial health.",
    "Restrictive covenants may limit our operating flexibility.",
    "We may not be able to generate sufficient cash to service our debt.",
    "Interest rate movements may increase our cost of borrowing.",
    "Exchange rate fluctuations may affect reported results.",
    "Disruptions to our supply chain could adversely affect operations.",
    "Cybersecurity incidents could disrupt our business.",
    "Litigation and disputes may result in significant liabilities.",
    "Changes in tax law could increase our effective tax rate.",
    "Climate-related regulation may increase compliance costs.",
    "We may be unable to implement our strategy successfully.",
    "Impairment of goodwill could affect reported earnings.",
    "Labour disputes could disrupt our operations.",
]

_HEDGE_SCALE = 0.55


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def render_document(mgmt: float, age_q: int, stress: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    p_hedge = _sigmoid(_HEDGE_SCALE * mgmt)

    def pick(n: int) -> list[str]:
        out = []
        for _ in range(n):
            pool = HEDGED if rng.random() < p_hedge else CONFIDENT
            out.append(pool[rng.integers(len(pool))])
        return out

    parts = ["BUSINESS REVIEW", *pick(6), ""]

    parts.append("LIQUIDITY AND CAPITAL RESOURCES")
    parts.extend(pick(4))
    # reduced disclosure: the detail sub-block disappears more often when
    # the management signal is bad — missingness is informative
    if rng.random() > _sigmoid(-2.6 + 0.9 * mgmt):
        order = rng.permutation(len(LIQUIDITY_DETAIL))
        parts.extend(LIQUIDITY_DETAIL[i] for i in order[:3])
    parts.append("")

    parts.append("OUTLOOK")
    parts.extend(pick(3))
    parts.append("")

    # append-only boilerplate: a fixed per-issuer order, take the first k.
    # k grows with age and with the aggregate cycle — for everyone.
    issuer_order = np.random.default_rng(seed % 100_003).permutation(len(BOILERPLATE))
    k = min(6 + age_q // 4 + (2 if stress else 0), len(BOILERPLATE))
    parts.append("RISK FACTORS")
    parts.extend(BOILERPLATE[i] for i in issuer_order[:k])
    return "\n".join(parts)


def render_all(filings: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'text' column. Deterministic given the filing table."""
    texts = [
        render_document(r.mgmt, r.age_q, r.stress, r.doc_seed)
        for r in filings.itertuples()
    ]
    out = filings.copy()
    out["text"] = texts
    return out
