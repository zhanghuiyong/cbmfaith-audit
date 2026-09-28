# CBM Faithfulness Reference Audit Protocol v0.3

This reference audit protocol reports observable behavior. Its controlled model-level
interventions are input replacements in a fitted model, not identified causal interventions
on disease biology.

## Required common evaluation design

1. Freeze the cohort split before fitting models. Keep patients, lesions, eyes,
   or other repeated biological units in one partition.
2. Give compared methods the same images, concept definitions, outcome, encoder
   policy, outer folds, and selection budget. Label architecture adaptations as
   adaptations rather than exact reproductions.
3. Report diagnosis, semantic concept recovery, and calibration separately.
4. For hybrid models, export additive concept and residual logits. The absolute
   logit route ratio is descriptive, gauge-dependent, and must not be called variance
   explained or causal attribution. Declare an offset convention learned from training/OOF
   data, report a reference-centered sensitivity analysis, and retain offset-invariant
   route-only predictions and rank concordance as primary evidence.
5. Replace concepts only with values inside the declared concept ontology. Call
   this a controlled model-level intervention unless a structural causal
   identification argument supports a stronger interpretation.
6. Fit residual-concept decoders out of sample. Their AUROC measures decodability,
   not causal leakage.
7. Repeat on a locked external cohort. State ontology mismatches rather than
   silently recoding them.

## Minimum report

- cohort size, outcome prevalence, biological grouping key, and split policy;
- combined, concept-only, and residual-only AUROC/AUPRC/Brier where defined;
- per-concept and macro semantic AUROC;
- normalized residual share and combined–residual rank correlation for additive
  hybrids;
- controlled model-level intervention response in logit and probability space;
- operational completeness relative to a fold-matched black box;
- out-of-sample residual concept decodability;
- missing metrics with an architecture-specific reason;
- analysis intent (prospective, confirmatory, exploratory, or post hoc).

Spatial alignment is a required extension when the model claims spatially
localized concepts, but v0.3 does not impose one mask metric across modalities.

Version 0.3 additionally requires an archived protocol hash, software version,
split identifiers, and checkpoint or model hashes sufficient to reproduce every
reported aggregate. Training seeds are independent stochastic replicates; folds
are blocks within a seed and must not be counted as independent replicates.
