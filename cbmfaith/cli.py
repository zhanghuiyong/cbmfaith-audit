"""Command-line adapter for conventional Paper 1 CSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import AuditInputs, audit_faithfulness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concepts", nargs="+")
    parser.add_argument("--combined", default="diagnosis_probability")
    parser.add_argument("--concept-route", default="concept_only_probability")
    parser.add_argument("--residual-route", default="raw_residual_probability")
    parser.add_argument("--blackbox")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--gauge-offset", type=float)
    parser.add_argument("--group-column", help="Biological-unit column for grouped leakage probes")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    names = args.concepts or [column.removeprefix("label_") for column in frame if column.startswith("label_")]
    labels = frame[[f"label_{name}" for name in names]].to_numpy(float) if names else None
    predictions = frame[[f"prob_{name}" for name in names]].to_numpy(float) if names else None
    replacement_columns = [f"replacement_logit_{name}" for name in names]
    replaced = frame[replacement_columns].to_numpy(float) if replacement_columns and all(c in frame for c in replacement_columns) else None
    inputs = AuditInputs(
        y_true=frame.target.to_numpy(int),
        combined_probability=frame[args.combined].to_numpy(float),
        concept_probability=frame[args.concept_route].to_numpy(float) if args.concept_route in frame else None,
        residual_probability=frame[args.residual_route].to_numpy(float) if args.residual_route in frame else None,
        concept_logit=frame.concept_logit_reconstructed.to_numpy(float) if "concept_logit_reconstructed" in frame else None,
        residual_logit=frame.residual_logit_gated.to_numpy(float) if "residual_logit_gated" in frame else None,
        concept_labels=labels,
        concept_predictions=predictions,
        concept_names=names,
        replaced_combined_logits=replaced,
        blackbox_probability=frame[args.blackbox].to_numpy(float) if args.blackbox else None,
        gauge_offset=args.gauge_offset,
        groups=frame[args.group_column].to_numpy() if args.group_column else None,
    )
    report = audit_faithfulness(inputs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
