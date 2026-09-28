# CBMFaith Audit

Repository: https://github.com/zhanghuiyong/cbmfaith-audit

CBMFaith Audit is a reference implementation for observable faithfulness audits of concept bottleneck models (CBMs). It reports diagnostic performance, semantic recovery, route reliance, controlled model-level intervention responsiveness, operational completeness, and cross-validated residual concept decodability.

These quantities are observational or model-level diagnostics unless a separate causal identification argument is supplied. They must not be interpreted automatically as causal mediation effects.

## Release

Current software release: **v0.3.1**.

The software package version is 0.3.1. The bundled machine-readable protocol and output schema remain at protocol/schema version **0.3.0**. This distinction is intentional.

## Installation

From a source checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check cbmfaith
```

A pre-built wheel may also be installed:

```bash
python -m pip install cbmfaith_audit-0.3.1-py3-none-any.whl
```

## Command-line interfaces

```bash
cbmfaith --help
cbmfaith-validate report.json
```

The `cbmfaith-validate` command validates exported JSON reports against the packaged schema and additional fail-closed checks.

## Python API

```python
from cbmfaith import fit_gauge_reference, gauge_fixed_route_reliance

reference = fit_gauge_reference(development_concept_logits, method="median")
report = gauge_fixed_route_reliance(
    test_concept_logits,
    test_residual_logits,
    test_combined_logits,
    offset=reference["offset"],
)
```

Models should expose combined, concept-route, and residual-route quantities separately when those routes are defined. Use `AuditInputs` directly when residual representations or replacement logits are available.

## Protocol and schemas

- `cbmfaith/BENCHMARK_PROTOCOL.md`: human-readable cross-architecture audit protocol.
- `cbmfaith/protocol/CBMFAITH_PROTOCOL_v0.3.md`: versioned protocol specification.
- `cbmfaith/protocol/audit-protocol-v0.3.0.json`: machine-readable protocol descriptor.
- `cbmfaith/schema/benchmark-output.schema.json`: output schema.
- `cbmfaith/schema/benchmark-manifest.schema.json`: benchmark manifest schema.

Metrics that are not defined for an architecture should be omitted and accompanied by an architecture-specific reason. Missing route fields mean that the quantity is not identifiable from that architecture; they do not mean zero.

## Scope and limitations

This repository is a reference implementation and protocol, not a community-standard benchmark. Independent adoption and governance would be required for such a designation.

The software does not include restricted patient data, model checkpoints, patient-level prediction matrices, or internal reader workbooks. Paper-specific analysis and figure-generation scripts may require external datasets or project-specific artifacts not distributed with this package.

## Citation

Citation metadata is provided in `CITATION.cff`. Once a DOI-backed archival release is created, add the DOI to that file and to the release notes.

## License

Released under the [MIT License](LICENSE).
