#!/usr/bin/env python3
"""Minimal test runner (pytest-compatible test files, no pytest needed)."""
import importlib.util, pathlib, sys, time, traceback

root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root))
failed = 0
for f in sorted((root / "tests").glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            t0 = time.time()
            try:
                getattr(mod, name)()
                print(f"PASS  {f.stem}::{name} ({time.time()-t0:.1f}s)")
            except Exception:
                failed += 1
                print(f"FAIL  {f.stem}::{name}")
                traceback.print_exc()
sys.exit(1 if failed else 0)
