# CBMFaith fixed reference protocol v0.3

This protocol audits observable behavior of concept bottleneck and hybrid residual
models. It does not identify causal mediation without an additional causal model and
its assumptions.

1. Freeze the cohort split before fitting. Keep repeated biological units together.
2. Match inputs, concept definitions, outcomes, folds, and selection budgets across
   models. Record architecture-specific adaptations.
3. Export combined, concept-only, and residual-only predictions when those routes are
   identifiable. Omit undefined quantities and record why.
4. Report diagnosis discrimination and calibration separately from semantic concept
   recovery.
5. Treat absolute additive-logit route shares as gauge-dependent. Freeze any gauge
   reference on development data and retain route-only predictions and rank agreement.
6. Describe concept replacement as a model-level intervention unless separate causal
   identification assumptions justify a stronger term.
7. Fit residual-concept decoders out of sample and preserve biological grouping.
8. Treat training seed as the independent replicate for stochastic training; treat
   folds as blocked repeated estimates.
9. Archive predictions, split identifiers, protocol hash, software version, and
   model/checkpoint hashes sufficient to reconstruct every aggregate.

The output contract is `schema/benchmark-output.schema.json`; the machine-readable
protocol descriptor is `protocol/audit-protocol-v0.3.0.json`.

