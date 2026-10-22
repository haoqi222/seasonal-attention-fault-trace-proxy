#!/usr/bin/env python3
"""Regenerate Figure 1 with an external legend for the V4 release."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "05_figures_tables/formal_v4"
OUTPUT_NAME = "fig1_v4_spatial_cross_validation_design_0903-01.png"


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--font-dir", type=Path, default=Path(r"C:\Windows\Fonts"))
    args = parser.parse_args()
    source = Path(__file__).with_name("25_generate_formal_figures_tables_v3.py")
    spec = importlib.util.spec_from_file_location("figures_v3", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=True)
    module.configure_font(args.font_dir)
    module.figure_1_spatial_cv(args.root, args.output, OUTPUT_NAME)
    print(args.output / OUTPUT_NAME)


if __name__ == "__main__":
    main()
