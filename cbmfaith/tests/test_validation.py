import copy

import pytest

from cbmfaith.validation import validate_report

VALID = {'schema_version': '0.3.0', 'causal_scope': 'model-level',
         'combined_diagnosis': {'n': 10, 'positives': 4, 'auroc': .7, 'auprc': .6, 'brier': .2}}

def test_valid_report():
    assert validate_report(VALID) == []

@pytest.mark.parametrize('field,value', [('n', 0), ('n', True), ('positives', 11),
                                         ('auroc', float('nan')), ('brier', None), ('auroc', 1.1)])
def test_invalid_diagnostics(field, value):
    r = copy.deepcopy(VALID); r['combined_diagnosis'][field] = value
    assert validate_report(r)

def test_optional_route_is_also_validated():
    r = copy.deepcopy(VALID); r['residual_route_diagnosis'] = {'n': 3}
    assert validate_report(r)

def test_invalid_top_level_and_extra_nonfinite():
    assert validate_report([])
    r = copy.deepcopy(VALID); r['extra'] = [float('inf')]
    assert validate_report(r)
