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
need = [f"data/raw/ciq_bond_{i}.xlsx" for i in
        ["US05602XQR25", "US05602XQQ42", "US05602XQS08", "US09659T2D24", "US09659W2X16", "US09659T2B67"]]
need += ["data/raw/ciq_cds_bnp_snrfor_eur.xlsx", "data/raw/ciq_capital_adequacy_quarterly.xlsx",
         "data/raw/ciq_basel_annual.xlsx", "data/raw/ciq_bnp_share.xlsx",
         "data/reference/bonds_static.csv", "data/reference/senior_t2_static.csv",
         "data/reference/fred_treasury.csv", "data/reference/bnp_srep_requirement.csv"]
missing = [f for f in need if not Path(f).exists()]
if missing:
    sys.exit("missing input files: " + ", ".join(missing))
print("all inputs present")
PY

step "1/8 Dataset";            python scripts/build_bnp_dataset.py > results/build_log.txt
step "2/8 Tests";              python -m pytest -q
step "3/8 Market analysis";    python -u scripts/market_analysis.py | tee results/market_log.txt
if [[ $QUICK -eq 0 ]]; then
  step "4/8 Calibration (~10 min)"; python -u scripts/calibrate_bnp.py | tee results/calibrate_log.txt
else
  step "4/8 Calibration skipped (--quick)"
fi
step "5/8 Capital structure: T2/SNP anchor (~2 min)"; python -u scripts/capital_structure.py | tee results/capital_structure_log.txt
step "6/8 Spread, call, stress (~5 min)"; python -u scripts/spread_and_stress.py | tee results/stress_log.txt
step "7/8 Parameter uncertainty (~6 min)"; python -u scripts/parameter_uncertainty.py | tee results/uncertainty_log.txt
step "8/8 Video (~2 min)";     python scripts/make_video.py

if command -v latexmk >/dev/null 2>&1; then
  step "PDF note"
  (cd paper && latexmk -pdf -interaction=nonstopmode -quiet at1_coco_note.tex >/dev/null && latexmk -c >/dev/null)
  echo "paper/at1_coco_note.pdf rebuilt"
else
  echo "latexmk not found: PDF not rebuilt"
fi

step "ALL DONE"
ls -1 results/*.png results/*.mp4
