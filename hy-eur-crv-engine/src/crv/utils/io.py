"""Pickle / JSON IO helpers with metadata stamping.

The track caches are pickled dicts carrying a small metadata header
(``saved_at``, ``track``, row/bond counts) so a downstream loader can report
provenance without re-deriving it.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from crv.utils.logging import get_logger

log = get_logger(__name__)


def save_pickle(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path | str) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path: Path | str) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_track_cache(panel, name: str, cache_path: Path | str, csv_path: Path | str,
                     panel_weeks: int, latest: str) -> dict:
    """Persist one track panel both as a metadata-stamped pickle and as a CSV."""
    meta = {
        "panel": panel, "track": name, "saved_at": datetime.now().isoformat(),
        "panel_weeks": panel_weeks, "latest": latest,
        "n_bond": int(panel["isin"].nunique()), "n_rows": int(len(panel)),
    }
    save_pickle(meta, cache_path)
    panel.to_csv(csv_path, index=False)
    log.info("[%s] cache -> %s | csv -> %s (%d rows, %d bonds)",
             name, cache_path, csv_path, meta["n_rows"], meta["n_bond"])
    return meta


def load_track_cache(cache_path: Path | str):
    """Load one track panel from its metadata-stamped pickle (None if absent)."""
    d = load_pickle(cache_path)
    if d is None:
        return None
    log.info("[%s] %s | %d bonds, %d rows",
             d["track"], d["saved_at"][:16], d["n_bond"], d["n_rows"])
    return d["panel"]
