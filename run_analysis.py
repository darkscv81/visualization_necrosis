"""Run the reproducible longitudinal rCBV workflow locally."""

import argparse
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from necrosis_analysis.pipeline import (
    analyze_inputs, export_results, inspect_inputs, load_config, register_inputs,
)
from necrosis_analysis.synthetic import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Analysis YAML file")
    source.add_argument("--synthetic", action="store_true", help="Generate an independent synthetic DICOM fixture")
    parser.add_argument("--output", required=True, type=Path, help="Directory for local generated results")
    parser.add_argument("--stage", choices=["inspect", "register", "full"], default="full")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        overrides = generate_synthetic_dataset(args.output / "synthetic_inputs") if args.synthetic else None
        config = load_config(args.config, overrides=overrides)
        inputs = inspect_inputs(config, args.output)
        logging.info("Inspected %d image volumes.", len(inputs.images))
        if args.stage == "inspect":
            return 0
        registered = register_inputs(config, inputs, args.output)
        logging.info("Registration status: %s", registered.report["review"]["status"])
        logging.info("Registration review fingerprint: %s", registered.report["fingerprint"])
        if args.stage == "register":
            return 0 if registered.report["review"]["status"] != "rejected" else 1
        analysis = analyze_inputs(config, inputs, registered, args.output)
        report_path = export_results(config, inputs, registered, analysis, args.output)
        logging.info("Saved %s", report_path)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
