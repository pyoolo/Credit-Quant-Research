# Data

Vendor price data for defaulted Venezuelan sovereign / PDVSA bonds cannot be
redistributed. Two options:

1. **Synthetic panel (default).** `political_option.data.simulate_default_panel()`
   generates a panel that reproduces the paper's stylised facts (level collapse at
   default, curve compression, jump repricing at political events, no macro channel).
   All scripts and tests run on it out of the box.

2. **Real data.** Drop a CSV here with a `date` column and one column per bond
   (prices in cents on the dollar), e.g. `venz.csv`, and load it with
   `political_option.data.load_price_panel("data/venz.csv")`. Macro variables
   (DXY, Brent, WTI, EMBI, EM HY, US 10Y, VIX, MOVE) should already be transformed
   into log returns / first differences, as in Appendix A.3 of the paper.
