"""Synthetic HY credit world.

Design principle
----------------
The three "worlds" (signal / coincident / noise) share *identical* latent
dynamics, events, fundamentals and market prices — the random draws are the
same. The only thing that differs is the mapping from the latent state to the
management-disclosure signal that drives the generated text:

  signal      : management observes a pipeline shock ``xi_t`` one step before
                it hits the credit state (it lands at t+2, so the information
                survives a one-quarter publication lag). Text has genuine
                incremental content over the market price.
  coincident  : text echoes the market's own information set (``x_hat``).
                Text-alone predicts events — and adds nothing given the
                spread. This is the tautology trap.
  noise       : text is independent noise. Nothing to find.

A pipeline evaluated on all three worlds can be *falsified*: it must find the
signal where it exists, and must find nothing where it does not.

Modelling references: discrete-time hazard framing follows Shumway (2001);
the disclosure-tone channel is inspired by Loughran & McDonald (2011) and the
"change beats level" evidence of Cohen, Malloy & Nicolosi (Lazy Prices).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

WORLDS = ("signal", "coincident", "noise")


@dataclass
class WorldConfig:
    world: str = "signal"
    n_issuers: int = 260
    n_quarters: int = 88
    seed: int = 7

    # aggregate credit cycle (2-state Markov, quarterly)
    p_enter_stress: float = 0.04
    p_exit_stress: float = 0.22
    stress_level: float = 2.0          # target intensity in stress
    cycle_smooth: float = 0.55         # AR smoothing of intensity

    # issuer latent distress x (mean-reverting to mu_i + beta_i * cycle)
    rho: float = 0.76
    sigma_x: float = 0.22
    mu_sd: float = 0.40                # issuer quality heterogeneity
    beta_mean: float = 0.95
    beta_sd: float = 0.22
    # pipeline component: persistent AR(1), lands in x with a 2-quarter
    # delay. Persistence is what makes early knowledge of it valuable over
    # a multi-quarter label window; an iid shock would be worth almost
    # nothing even to an oracle (verified in docs/design_notes).
    phi_xi: float = 0.80
    sigma_nu: float = 0.48

    # hazard: P(event at t) = sigmoid(hz_a + hz_b * x_t)
    hz_a: float = -5.70
    hz_b: float = 1.70

    # market observation
    market_noise: float = 0.18         # sd of market's noise on x
    spread_base: float = 250.0         # bp at x_hat = 0
    spread_kappa: float = 0.85
    spread_quote_noise: float = 0.06   # lognormal quote noise
    rating_alpha: float = 0.16         # EMA speed of the rating filter

    # filings
    lag_p2: float = 0.15               # P(publication lag = 2q), else 1q
    fund_noise: float = 0.12

    # text channel
    candid_frac: float = 0.70          # share of issuers whose text leads
    psi: float = 1.50                  # loading of xi in the mgmt signal
    text_obs_noise: float = 0.12       # noise on the mgmt signal itself


def _simulate_cycle(cfg: WorldConfig, rng: np.random.Generator):
    state = np.zeros(cfg.n_quarters, dtype=int)
    intensity = np.zeros(cfg.n_quarters)
    s, c = 0, 0.0
    for t in range(cfg.n_quarters):
        if s == 0 and rng.random() < cfg.p_enter_stress:
            s = 1
        elif s == 1 and rng.random() < cfg.p_exit_stress:
            s = 0
        target = cfg.stress_level if s == 1 else 0.0
        c = cfg.cycle_smooth * c + (1 - cfg.cycle_smooth) * target
        state[t], intensity[t] = s, c
    return pd.DataFrame({"quarter": np.arange(cfg.n_quarters),
                         "stress": state, "intensity": intensity})


class _Roster:
    """Active issuers with entry on exit (keeps the panel free of
    survivorship bias by construction: departed names stay in the data)."""

    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.next_id = 0
        n = cfg.n_issuers
        self.ids = np.arange(n); self.next_id = n
        self.mu = rng.normal(0, cfg.mu_sd, n)
        self.beta = np.maximum(rng.normal(cfg.beta_mean, cfg.beta_sd, n), 0.05)
        stat_sd = cfg.sigma_x / np.sqrt(1 - cfg.rho ** 2)
        self.x = self.mu + rng.normal(0, stat_sd, n)
        self.candid = rng.random(n) < cfg.candid_frac
        self.r_ema = self.x.copy()
        stat_xi = cfg.sigma_nu / np.sqrt(1 - cfg.phi_xi ** 2)
        self.pend1 = rng.normal(0, stat_xi, n)  # lands next quarter
        self.pend2 = rng.normal(0, stat_xi, n)  # lands at t+2 (mgmt sees it)
        self.born = np.zeros(n, dtype=int)

    def replace(self, idx: np.ndarray, t: int):
        cfg, rng = self.cfg, self.rng
        k = len(idx)
        if k == 0:
            return
        self.ids[idx] = np.arange(self.next_id, self.next_id + k)
        self.next_id += k
        self.mu[idx] = rng.normal(0, cfg.mu_sd, k)
        self.beta[idx] = np.maximum(rng.normal(cfg.beta_mean, cfg.beta_sd, k), 0.05)
        stat_sd = cfg.sigma_x / np.sqrt(1 - cfg.rho ** 2)
        self.x[idx] = self.mu[idx] + rng.normal(0, stat_sd, k)
        self.candid[idx] = rng.random(k) < cfg.candid_frac
        self.r_ema[idx] = self.x[idx]
        self.pend1[idx] = 0.0
        self.pend2[idx] = rng.normal(0, cfg.sigma_nu, k)
        self.born[idx] = t


def simulate_world(cfg: WorldConfig) -> dict:
    """Returns {'cycle', 'panel', 'filings'} DataFrames.

    panel   : one row per alive issuer-quarter (market data, event flag)
    filings : one row per filing, with publication quarter and the latent
              management signal the document renderer will verbalise.
    """
    if cfg.world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}")
    rng = np.random.default_rng(cfg.seed)
    cycle = _simulate_cycle(cfg, rng)
    ros = _Roster(cfg, rng)

    panel_rows, filing_rows = [], []
    for t in range(cfg.n_quarters):
        c = cycle.intensity.iat[t]

        # --- latent state update (identical across worlds) ---------------
        eps = rng.normal(0, cfg.sigma_x, len(ros.ids))
        ros.x = (cfg.rho * ros.x
                 + (1 - cfg.rho) * (ros.mu + ros.beta * c)
                 + eps
                 + ros.pend1)                       # pipeline shock lands
        nu = rng.normal(0, cfg.sigma_nu, len(ros.ids))
        xi_new = cfg.phi_xi * ros.pend2 + nu        # persistent pipeline
        ros.pend1, ros.pend2 = ros.pend2, xi_new    # shift the queue

        # --- events -------------------------------------------------------
        p = 1.0 / (1.0 + np.exp(-(cfg.hz_a + cfg.hz_b * ros.x)))
        event = rng.random(len(ros.ids)) < p

        # --- market observation (identical across worlds) -----------------
        x_hat = ros.x + rng.normal(0, cfg.market_noise, len(ros.ids))
        spread = (cfg.spread_base * np.exp(cfg.spread_kappa * x_hat)
                  * np.exp(rng.normal(0, cfg.spread_quote_noise, len(ros.ids))))
        ros.r_ema = (1 - cfg.rating_alpha) * ros.r_ema + cfg.rating_alpha * ros.x
        rating = np.clip(np.round(3.5 + 1.15 * ros.r_ema), 1, 7).astype(int)

        # --- management signal: THE ONLY PLACE THE WORLDS DIFFER ----------
        # Both noise vectors are drawn in every world so that RNG consumption
        # is identical: same seed => bit-identical states, events and prices
        # across worlds. Only the text channel changes.
        obs_noise = rng.normal(0, cfg.text_obs_noise, len(ros.ids))
        indep = rng.normal(0, 1.0, len(ros.ids))
        if cfg.world == "signal":
            # pend2 lands at t+2. The filing for period t is published at
            # t+1 (mostly): by then the t+1 shock (pend1) is already in the
            # price, so pend1 carries no incremental information for the
            # reader — pend2 does. Publication lag eats exactly one quarter
            # of lead; the economics must supply at least two.
            lead = np.where(ros.candid, cfg.psi, 0.0) * ros.pend2
            mgmt = ros.x + lead + obs_noise
        elif cfg.world == "coincident":
            mgmt = x_hat + obs_noise            # text echoes the price's info
        else:                                   # noise
            mgmt = indep

        lag = 1 + (rng.random(len(ros.ids)) < cfg.lag_p2).astype(int)
        lev = 5.0 + 1.1 * ros.x + rng.normal(0, cfg.fund_noise, len(ros.ids))
        cov = np.maximum(3.2 - 0.9 * ros.x
                         + rng.normal(0, cfg.fund_noise, len(ros.ids)), 0.2)
        mar = 12.0 - 2.0 * ros.x + rng.normal(0, 4 * cfg.fund_noise, len(ros.ids))

        for j in range(len(ros.ids)):
            panel_rows.append((int(ros.ids[j]), t, float(ros.x[j]),
                               bool(event[j]), float(spread[j]),
                               int(rating[j])))
            filing_rows.append((int(ros.ids[j]), t, t + int(lag[j]),
                                float(mgmt[j]), float(lev[j]), float(cov[j]),
                                float(mar[j]), int(t - ros.born[j]),
                                int(cycle.stress.iat[t]),
                                int(rng.integers(0, 2 ** 31 - 1))))
        ros.replace(np.where(event)[0], t + 1)

    panel = pd.DataFrame(panel_rows, columns=[
        "issuer", "quarter", "x", "event", "spread", "rating"])
    filings = pd.DataFrame(filing_rows, columns=[
        "issuer", "period_q", "pub_q", "mgmt", "leverage", "coverage",
        "margin", "age_q", "stress", "doc_seed"])
    return {"cycle": cycle, "panel": panel, "filings": filings}
