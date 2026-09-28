# CBMFaith Audit v0.3.1 — GitHub Release Notes

## Summary

CBMFaith Audit v0.3.1 is the paper-associated software release candidate for observable faithfulness audits of concept bottleneck models.

The package reports diagnostic performance, semantic concept recovery, route reliance, controlled model-level intervention responsiveness, operational completeness, and cross-validated residual concept decodability. These are not automatically causal mediation estimands.

## Versioning

- Software package: **0.3.1**
- Protocol descriptor: **0.3.0**
- Output schema: **0.3.0**
- Manifest schema: **0.3.0**

The software patch version and protocol/schema version intentionally differ.

## Release assets

Recommended GitHub Release tag: `v0.3.1`

Attach:
- `cbmfaith_audit-0.3.1-py3-none-any.whl`
- `SHA256SUMS.txt`

GitHub will automatically provide source-code ZIP and TAR.GZ archives for the tagged commit.

## Validation before release

The release should only be tagged after:
1. the full test suite passes;
2. the wheel is rebuilt from the tagged source;
3. wheel metadata reports version 0.3.1 and MIT licensing;
4. `cbmfaith --help` and `cbmfaith-validate --help` run successfully;
5. `SHA256SUMS.txt` is regenerated from the final wheel.

## Archival DOI

After publishing the GitHub Release, archive the release with Zenodo. Add the resulting version DOI to `CITATION.cff`, this release note, and the manuscript Code Availability statement.

## Known scope boundary

This software release does not include restricted patient data, model checkpoints, patient-level prediction matrices, internal reader workbooks, or the complete paper training environment. Paper-specific scripts outside this package may depend on external datasets and project-specific artifacts.
