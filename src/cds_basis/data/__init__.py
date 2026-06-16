"""Data layer: panel schema, synthetic generator and validation."""
from .synthetic import (
    PANEL_COLUMNS,
    SyntheticConfig,
    generate_panel,
    validate_panel,
)

__all__ = [
    "PANEL_COLUMNS",
    "SyntheticConfig",
    "generate_panel",
    "validate_panel",
]
