"""Offline structural validator for CBMFaith JSON reports."""

from __future__ import annotations

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_report(report: dict) -> list[str]:
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = [f"{'.'.join(map(str, e.absolute_path)) or '$'}: {e.message}"
              for e in validator.iter_errors(report)]
    def finite(value, path='$'):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path}: non-finite numbers are not JSON values")
        elif isinstance(value, dict):
            for key, item in value.items(): finite(item, f'{path}.{key}')
        elif isinstance(value, list):
            for i, item in enumerate(value): finite(item, f'{path}[{i}]')
    finite(report)
    if isinstance(report, dict):
        for name in ('combined_diagnosis', 'concept_route_diagnosis', 'residual_route_diagnosis'):
            diagnostic = report.get(name)
            if isinstance(diagnostic, dict):
                n, positives = diagnostic.get('n'), diagnostic.get('positives')
                if type(n) is int and type(positives) is int and positives > n:
                    errors.append(f'{name}.positives exceeds n')
    return errors


def schema_path() -> Path:
    return Path(str(files("cbmfaith").joinpath("schema/benchmark-output.schema.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CBMFaith v0.3 JSON report")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = validate_report(report)
    if errors:
        raise SystemExit("invalid CBMFaith report:\n- " + "\n- ".join(errors))
    print(f"valid CBMFaith v0.3 report: {args.report}")
