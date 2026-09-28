"""Reusable metrics for auditing concept-bottleneck faithfulness."""

from .audit import AuditInputs, audit_faithfulness
from .metrics import apply_gauge, fit_gauge_reference, gauge_fixed_route_reliance

__all__ = [
    "AuditInputs",
    "apply_gauge",
    "audit_faithfulness",
    "fit_gauge_reference",
    "gauge_fixed_route_reliance",
]
__version__ = "0.3.1"
