import numpy as np
import pytest

from at1_coco.processes import CET1Params, draw_randoms, simulate_cet1


def base_params(**kw):
    d = dict(
        c0=13.0, kappa=0.3, theta=13.0, sigma=1.2,
        jump_intensity=0.15, jump_median=2.0, jump_vol=0.5, ponv_intensity=0.005,
    )
    d.update(kw)
    return CET1Params(**d)


def test_shapes_and_grid():
    r = draw_randoms(1000, 120, seed=1)
    sim = simulate_cet1(base_params(), trigger_level=5.125, horizon=10.0, dt=1 / 12, randoms=r)
    assert sim.paths.shape == (1000, 121)
    assert sim.times[-1] == pytest.approx(10.0)
    assert sim.paths[:, 0].tolist() == [13.0] * 1000


def test_reproducible():
    r = draw_randoms(500, 120, seed=7)
    a = simulate_cet1(base_params(), 5.125, 10.0, 1 / 12, r)
    b = simulate_cet1(base_params(), 5.125, 10.0, 1 / 12, r)
    assert np.array_equal(a.paths, b.paths)
    assert np.array_equal(a.trigger_idx, b.trigger_idx)


def test_trigger_is_recorded_below_level():
    # a violent, low-capital bank should absorb on many paths
    p = base_params(c0=6.0, theta=6.0, sigma=3.0, jump_intensity=0.5, jump_median=3.0)
    r = draw_randoms(5000, 120, seed=3)
    sim = simulate_cet1(p, trigger_level=5.125, horizon=10.0, dt=1 / 12, randoms=r)
    absorbed = sim.trigger_idx >= 0
    assert absorbed.mean() > 0.3
    # mechanical triggers must have crossed the level at the recorded step
    mech = sim.trigger_type == 1
    idx = sim.trigger_idx[mech]
    vals = sim.paths[mech, idx]
    assert np.all(vals <= 5.125 + 1e-9)


def test_absorption_increases_with_ponv():
    r = draw_randoms(5000, 120, seed=11)
    lo = simulate_cet1(base_params(ponv_intensity=0.0), 5.125, 10.0, 1 / 12, r)
    hi = simulate_cet1(base_params(ponv_intensity=0.10), 5.125, 10.0, 1 / 12, r)
    assert (hi.trigger_idx >= 0).mean() > (lo.trigger_idx >= 0).mean()


def test_higher_capital_lowers_absorption():
    r = draw_randoms(5000, 120, seed=5)
    weak = simulate_cet1(base_params(c0=8.0, theta=8.0), 5.125, 10.0, 1 / 12, r)
    strong = simulate_cet1(base_params(c0=16.0, theta=16.0), 5.125, 10.0, 1 / 12, r)
    assert (strong.trigger_idx >= 0).mean() < (weak.trigger_idx >= 0).mean()


def test_randoms_step_guard():
    r = draw_randoms(10, 12, seed=0)
    with pytest.raises(ValueError):
        simulate_cet1(base_params(), 5.125, 10.0, 1 / 12, r)  # needs 120 steps
