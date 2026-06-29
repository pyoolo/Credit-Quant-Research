"""
models.py
---------
Convertible Bond Pricing Engine.

1. Black-Scholes benchmark
2. Tsiveriotis-Fernandes (1998) on a trinomial tree
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ConvertibleBond:
    face_value       : float = 1000.0
    maturity         : float = 3.0
    coupon_rate      : float = 0.04
    coupon_freq      : int   = 2
    conversion_ratio : float = 20.0
    call_schedule    : list  = field(default_factory=list)
    put_schedule     : list  = field(default_factory=list)

    @property
    def coupon(self) -> float:
        return self.face_value * self.coupon_rate / self.coupon_freq

    @property
    def coupon_times(self) -> np.ndarray:
        n = int(self.maturity * self.coupon_freq)
        return np.linspace(self.maturity / n, self.maturity, n)


@dataclass
class MarketParams:
    S0            : float = 50.0
    r             : float = 0.03
    sigma         : float = 0.30
    credit_spread : float = 0.05
    recovery      : float = 0.40
    q             : float = 0.00

    @property
    def hazard_rate(self) -> float:
        return self.credit_spread / (1.0 - self.recovery)


# ---------------------------------------------------------------------------
# Black-Scholes benchmark
# ---------------------------------------------------------------------------

def price_bs_benchmark(cb: ConvertibleBond, mkt: MarketParams) -> dict:
    from scipy.stats import norm
    S = mkt.S0; K = cb.face_value / cb.conversion_ratio
    T = cb.maturity; r = mkt.r; sig = mkt.sigma; q = mkt.q
    F = cb.face_value; c = cb.coupon; disc = r + mkt.credit_spread

    bond_floor = sum(c * np.exp(-disc * t) for t in cb.coupon_times)
    bond_floor += F * np.exp(-disc * T)

    d1 = (np.log(S/K) + (r - q + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    call    = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    opt_val = cb.conversion_ratio * call
    delta   = cb.conversion_ratio * np.exp(-q*T) * norm.cdf(d1)
    price   = bond_floor + opt_val
    conv    = cb.conversion_ratio * S
    return {"model": "Black-Scholes", "price": price, "bond_floor": bond_floor,
            "option_value": opt_val, "conversion_value": conv,
            "delta": delta, "premium": (price - conv) / (conv + 1e-12)}


# ---------------------------------------------------------------------------
# Tsiveriotis-Fernandes on a full grid
# ---------------------------------------------------------------------------

def price_tsiveriotis_fernandes(
    cb: ConvertibleBond, mkt: MarketParams, N: int = 300,
) -> dict:
    """
    TF (1998) pricing on a Kamrad-Ritchken trinomial tree.

    Discounting scheme (per Tsiveriotis & Fernandes 1998):
    -------------------------------------------------------
    At each tree node:
        u = total CB value
        v = cash-only component

    The key insight: the equity component of u is default-free and
    discounts at r; the cash component (v) discounts at r+h.

    Implemented as:
        u_discounted = (1-w) * E[u] * exp(-r*dt) + w * E[u] * exp(-(r+h)*dt)
    where w = v_children / u_children is the cash weight.

    Recovery is NOT added as a separate cash flow here -- it is already
    implicitly captured by the spread discounting on the cash component.
    Explicitly, using exp(-(r+h)*dt) ≈ exp(-r*dt)*(1-h*dt) means that
    in expectation the cash component loses h*dt per period (default prob),
    which matches the TF PDE's -h*v term. Adding an explicit recovery term
    on top would double-count.

    Tree indexing:
    --------------
    At step i (0..N): active absolute indices in {N-i, N-i+2, ..., N+i}.
    log-price at index k = (k - N) * dx.
    Children of node k at step i+1: k-1 (up), k (mid), k+1 (down).
    """
    h   = mkt.hazard_rate
    r   = mkt.r
    S0  = mkt.S0; sig = mkt.sigma; q = mkt.q
    T   = cb.maturity; F = cb.face_value
    cr  = cb.conversion_ratio; c = cb.coupon

    dt  = T / N
    dx  = sig * np.sqrt(3 * dt)
    mu  = (r - q - 0.5 * sig**2) * dt

    pu = 0.5 * ((sig**2*dt + mu**2) / dx**2 + mu / dx)
    pd = 0.5 * ((sig**2*dt + mu**2) / dx**2 - mu / dx)
    pm = 1.0 - pu - pd
    if not (pu > 0 and pd > 0 and pm > 0):
        raise ValueError(f"Negative probs pu={pu:.4f} pm={pm:.4f} pd={pd:.4f}")

    log_S_offsets = (np.arange(2*N+1) - N) * dx

    # Terminal condition
    S_T = S0 * np.exp(log_S_offsets)
    u   = np.maximum(cr * S_T, F)
    v   = np.where(cr * S_T <= F, F, 0.0)

    coupon_steps = {round(t / dt) for t in cb.coupon_times}
    call_map     = {round(t / dt): px for t, px in cb.call_schedule}
    put_map      = {round(t / dt): px for t, px in cb.put_schedule}

    disc_r  = np.exp(-r * dt)
    disc_rh = np.exp(-(r + h) * dt)

    for i in range(N - 1, -1, -1):
        active = np.arange(N - i, N + i + 1, 2)
        ku = active - 1; km = active; kd = active + 1

        # Risk-neutral expectations from children
        u_c = pu * u[ku] + pm * u[km] + pd * u[kd]
        v_c = pu * v[ku] + pm * v[km] + pd * v[kd]

        # Cash weight: what fraction is the cash component
        w = np.clip(v_c / (u_c + 1e-12), 0.0, 1.0)

        # Blended discount: equity part at r, cash part at r+h
        u_d = (1.0 - w) * u_c * disc_r + w * u_c * disc_rh
        v_d = w * u_c * disc_rh

        # Coupon (cash flow at step boundary i -> i+1)
        if i + 1 in coupon_steps:
            u_d += c
            v_d += c

        S_i    = S0 * np.exp(log_S_offsets[active])
        conv_v = cr * S_i

        # Call provision
        if i in call_map:
            call_px = call_map[i]
            mask = u_d > call_px
            holder_converts = conv_v >= call_px
            u_d = np.where(mask, np.where(holder_converts, conv_v, call_px), u_d)
            v_d = np.where(mask, np.where(holder_converts, 0.0, call_px), v_d)

        # Put provision
        if i in put_map:
            put_px = put_map[i]
            mask = u_d < put_px
            u_d  = np.where(mask, put_px, u_d)
            v_d  = np.where(mask, put_px, v_d)

        # Voluntary conversion
        mask = conv_v > u_d
        u_d  = np.where(mask, conv_v, u_d)
        v_d  = np.where(mask, 0.0,    v_d)

        u[active] = u_d
        v[active] = v_d

    price = u[N]; v0 = v[N]; conv = cr * S0

    # Greeks from ±1 nodes at step 1
    S_up = S0 * np.exp(log_S_offsets[N-1])
    S_dn = S0 * np.exp(log_S_offsets[N+1])
    u_up = u[N-1]; u_dn = u[N+1]
    delta = (u_up - u_dn) / (S_up - S_dn + 1e-12)
    gamma = (u_up - 2*price + u_dn) / ((0.5*(S_up - S_dn))**2 + 1e-12)

    return {
        "model"            : "Tsiveriotis-Fernandes",
        "price"            : price,
        "cash_component"   : v0,
        "equity_component" : price - v0,
        "conversion_value" : conv,
        "delta"            : delta,
        "gamma"            : gamma,
        "premium"          : (price - conv) / (conv + 1e-12),
        "bond_floor_proxy" : v0 / (price + 1e-12),
    }


# ---------------------------------------------------------------------------
# Sensitivity surface
# ---------------------------------------------------------------------------

def sensitivity_surface(
    cb: ConvertibleBond, base_mkt: MarketParams,
    param: str, param_range: np.ndarray, model: str = "tf",
) -> tuple[np.ndarray, np.ndarray]:
    prices = []
    for val in param_range:
        m = MarketParams(S0=base_mkt.S0, r=base_mkt.r, sigma=base_mkt.sigma,
                         credit_spread=base_mkt.credit_spread,
                         recovery=base_mkt.recovery, q=base_mkt.q)
        setattr(m, param, val)
        res = price_tsiveriotis_fernandes(cb, m, N=200) if model == "tf" \
              else price_bs_benchmark(cb, m)
        prices.append(res["price"])
    return param_range, np.array(prices)
