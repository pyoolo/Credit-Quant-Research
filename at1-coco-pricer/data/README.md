# data/

```
data/
├── raw/         S&P Capital IQ exports, never edited by hand        (gitignored: licensed)
├── reference/   public inputs: term sheets, requirements, FRED      (committed)
└── processed/   built by scripts/build_bnp_dataset.py               (gitignored: contains CIQ data)
```

Rebuild `processed/` from the repository root:

```
python scripts/build_bnp_dataset.py
```

## raw/

| File | Content |
|---|---|
| `ciq_bond_US05602XQR25.xlsx` | AT1 6.875% NC33, CIQ evaluated prices ("Chart" sheet) |
| `ciq_bond_US05602XQQ42.xlsx` | AT1 7.45% NC35 |
| `ciq_bond_US05602XQS08.xlsx` | AT1 7.20% NC36 |
| `ciq_bond_US09659T2D24.xlsx` | T2 5.906% 2035NC34 |
| `ciq_bond_US09659W2X16.xlsx` | SNP 5.738% 2035NC34 |
| `ciq_bond_US09659T2B67.xlsx` | T2 2.588% 2035NC30 (resets on 5y CMT in 2030) |
| `ciq_cds_bnp_snrfor_eur.xlsx` | Senior CDS, EUR, MM14: 5y daily since 2023 + curve snapshots |
| `ciq_capital_adequacy_quarterly.xlsx` | CET1, Tier 1, leverage, RWA by quarter |
| `ciq_basel_annual.xlsx` | Year-end requirement stack |
| `ciq_bnp_share.xlsx` | BNP share price (EUR) |
| `ciq_capital_structure.xlsx` | Debt outstanding (not used: CIQ labels the AT1s "Senior Debt") |

The T2/SNP exports carry no ISIN; they were matched by "T0 Accrued Interest" at 27-Oct-2026
(2.592 / 1.068 / 0.539 = 5.906% / 5.738% / 2.588% coupons on 30/360).

## reference/

| File | Content |
|---|---|
| `bonds_static.csv` | AT1 terms from the BNP prospectuses |
| `senior_t2_static.csv` | T2 and SNP terms from the BNP debt issuance list / pricing supplements |
| `bnp_srep_requirement.csv` | CET1 (MDA) requirement by quarter, with source |
| `bnp_cet1_requirement_breakdown.csv` | Requirement stack and O-SII step-ups |
| `bnp_eba_stress_2025.csv` | EBA 2025 stress test |
| `fred_treasury.csv` | FRED DGS5 / DGS7 / DGS10 (update before extending the dates) |

## processed/

| File | Content |
|---|---|
| `bnp_market_daily.csv` | One row per business day and bond: prices, accrued, yield to first call, spread over FRED, CMT, CDS, share, published CET1 |
| `bnp_instruments.csv` | One row per instrument (6 bonds + CDS) |
| `bnp_capital_quarterly.csv` | CET1 history, MDA requirement, cushion, publication date |
| `bnp_capital_structure_dataset.xlsx` | All of the above in one workbook, with a README sheet and reference tables |

Coverage as built on 28-Oct-2026: AT1 prices to 25-Sep-2026; T2, SNP and CDS to 27-Oct-2026;
FRED and share price to 24-Sep-2026. 3Q26 CET1 (12.9%) entered by hand from the results release.
