#!/usr/bin/env bash
# Rebuild everything from the raw data: dataset, tests, analysis, figures,
# video and the PDF note.  Run from the at1-coco-pricer folder:
#
#     source .venv/bin/activate
#     bash run_all.sh            # full run (~25 min)
#     bash run_all.sh --quick    # skip the weekly calibration (figures from
#                                # spread_and_stress / uncertainty / video only)
#
# Logs go to results/*_log.txt.  The script stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

step() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

step "Checking inputs"
python - <<'PY'
import sys
from pathlib import Path
import pandas as pd
need = ["data/raw/ciq_capital_adequacy_quarterly.xlsx", "data/raw/ciq_basel_annual.xlsx",
        "data/raw/ciq_bnp_share.xlsx", "data/bonds_static.csv", "data/fred_treasury.csv",
        "data/bnp_srep_requirement.csv"]
missing = [f for f in need if not Path(f).exists()]
if missing:
    sys.exit("missing input files: " + ", ".join(missing))
d = pd.read_excel("data/raw/ciq_capital_adequacy_quarterly.xlsx", header=None)
r = d.index[d[0] == "Fiscal Period Ended"][0]
dates = pd.to_datetime(d.iloc[r, 1:]).dropna()
print(f"CET1 export: {dates.min().date()} -> {dates.max().date()} ({len(dates)} quarters)")
fred = pd.read_csv("data/fred_treasury.csv")
print(f"FRED curve up to {fred.iloc[:, 0].iloc[-1]}")
PY

step "1/7 Dataset";            python scripts/build_dataset.py > results/build_log.txt
step "2/7 Tests";              python -m pytest -q
step "3/7 Market analysis";    python -u scripts/market_analysis.py | tee results/market_log.txt
if [[ $QUICK -eq 0 ]]; then
  step "4/7 Calibration (~10 min)"; python -u scripts/calibrate_bnp.py | tee results/calibrate_log.txt
else
  step "4/7 Calibration skipped (--quick)"
fi
step "5/7 Spread, call, stress (~5 min)"; python -u scripts/spread_and_stress.py | tee results/stress_log.txt
step "6/7 Parameter uncertainty (~6 min)"; python -u scripts/parameter_uncertainty.py | tee results/uncertainty_log.txt
step "7/7 Video (~2 min)";     python scripts/make_video.py

if command -v latexmk >/dev/null 2>&1; then
  step "PDF note"
  (cd paper && latexmk -pdf -interaction=nonstopmode -quiet at1_coco_note.tex >/dev/null && latexmk -c >/dev/null)
  echo "paper/at1_coco_note.pdf rebuilt"
else
  echo "latexmk not found: PDF not rebuilt"
fi

step "ALL DONE"
ls -1 results/*.png results/*.mp4
