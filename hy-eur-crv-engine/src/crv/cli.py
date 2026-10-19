"""Command-line interface for the CRV engine.

    crv fetch              run the shared fetch + 3-track router
    crv track1             run the cross-section pipeline (Track 1)
    crv track2             run the self-ref pipeline (Track 2)
    crv track3             run the distressed pipeline (Track 3)
    crv all                fetch + all three tracks, in order
    crv price <spec.json>  price one or more new-issue specs from a JSON file
    crv app                launch the Streamlit app

Bloomberg-dependent steps (``fetch`` and the per-track fetches inside
``track2`` / ``track3``) require a terminal with the ``bql`` package available.
"""

from __future__ import annotations

import argparse
import json
import sys

from crv import __version__
from crv.utils.logging import configure, get_logger

log = get_logger(__name__)


def _cmd_fetch(_args) -> None:
    from crv.data import pipeline
    pipeline.run()


def _cmd_track1(_args) -> None:
    from crv.tracks.cross_section import pipeline
    pipeline.run(run_backtest=not _args.no_backtest)


def _cmd_track2(_args) -> None:
    from crv.tracks.self_ref import pipeline
    pipeline.run()


def _cmd_track3(_args) -> None:
    from crv.tracks.distressed import pipeline
    pipeline.run()


def _cmd_all(args) -> None:
    _cmd_fetch(args)
    _cmd_track1(args)
    _cmd_track2(args)
    _cmd_track3(args)


def _cmd_price(args) -> None:
    from crv.pricing import analytics, engine
    with open(args.spec) as f:
        specs = json.load(f)
    if isinstance(specs, dict):
        specs = [specs]
    art = engine.load_artefacts()
    df = analytics.batch_price(specs, art)
    print(df.to_string(index=False))


def _cmd_app(_args) -> None:
    import subprocess
    from pathlib import Path
    main_py = Path(__file__).parent / "app" / "main.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(main_py)], check=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crv", description="HY EUR relative-value engine")
    p.add_argument("--version", action="version", version=f"crv {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="shared fetch + 3-track router").set_defaults(func=_cmd_fetch)
    t1 = sub.add_parser("track1", help="cross-section pipeline")
    t1.add_argument("--no-backtest", action="store_true", help="skip the backtest")
    t1.set_defaults(func=_cmd_track1)
    sub.add_parser("track2", help="self-ref pipeline").set_defaults(func=_cmd_track2)
    sub.add_parser("track3", help="distressed pipeline").set_defaults(func=_cmd_track3)
    allp = sub.add_parser("all", help="fetch + all three tracks")
    allp.add_argument("--no-backtest", action="store_true")
    allp.set_defaults(func=_cmd_all)
    pr = sub.add_parser("price", help="price new-issue specs from a JSON file")
    pr.add_argument("spec", help="path to a JSON spec (object or list of objects)")
    pr.set_defaults(func=_cmd_price)
    sub.add_parser("app", help="launch the Streamlit app").set_defaults(func=_cmd_app)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import logging
    configure(logging.DEBUG if args.verbose else logging.INFO)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
