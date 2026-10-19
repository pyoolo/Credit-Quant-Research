"""Thin logging wrapper so modules don't ``print``.

Every module does ``log = get_logger(__name__)``. The CLI configures the root
handler once; library code never touches global logging state.
"""

from __future__ import annotations

import logging as _logging
import sys

_CONFIGURED = False


def configure(level: int = _logging.INFO) -> None:
    """Configure the root logger once (idempotent). Called by the CLI."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = _logging.StreamHandler(sys.stdout)
    handler.setFormatter(_logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                                             datefmt="%H:%M:%S"))
    root = _logging.getLogger("crv")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> _logging.Logger:
    """Return a namespaced logger under the ``crv`` root."""
    if not name.startswith("crv"):
        name = f"crv.{name}"
    return _logging.getLogger(name)
