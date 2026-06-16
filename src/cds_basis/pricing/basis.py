"""CDS-cash basis computation.

Basis = CDS par spread - cash spread (Z-spread or asset-swap spread).
Negative basis => bond cheap relative to CDS (classic negative-basis trade:
buy bond + buy protection).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .bond import Bond
from .cds import CDS
from .curves import DiscountCurve


class CashSpread(str, Enum):
    Z_SPREAD = "z_spread"
    ASW = "asw"


@dataclass
class BasisResult:
    cds_spread: float
    cash_spread: float
    basis: float
    cash_spread_type: str

    @property
    def basis_bps(self) -> float:
        return self.basis * 1e4


def compute_basis(
    curve: DiscountCurve,
    bond: Bond,
    bond_price: float,
    cds: CDS,
    cds_spread: float,
    cash_spread_type: CashSpread = CashSpread.ASW,
) -> BasisResult:
    """Compute the CDS-cash basis for one name on one date.

    Parameters
    ----------
    curve:
        Risk-free discount curve.
    bond, bond_price:
        The cash bond and its observed dirty price.
    cds, cds_spread:
        The CDS contract and its observed par spread (decimal, e.g. 0.012).
    cash_spread_type:
        Which cash spread to subtract from the CDS spread.
    """
    if cash_spread_type is CashSpread.ASW:
        cash = bond.asset_swap_spread(curve, bond_price)
    else:
        cash = bond.z_spread(curve, bond_price)
    return BasisResult(
        cds_spread=cds_spread,
        cash_spread=cash,
        basis=cds_spread - cash,
        cash_spread_type=cash_spread_type.value,
    )
