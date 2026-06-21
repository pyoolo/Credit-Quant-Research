from bcu import data


def test_columns_and_size():
    df = data.generate(n=1000, seed=0)
    for c in ["rating", "cds_spread", "downgrade", "earnings_miss",
              "spread_widening", "default"]:
        assert c in df.columns
    assert len(df) == 1000


def test_signals_predict_default():
    df = data.generate(n=8000, seed=2)
    # default rate must be higher among downgraded names
    hi = df.loc[df.downgrade == 1, "default"].mean()
    lo = df.loc[df.downgrade == 0, "default"].mean()
    assert hi > lo
