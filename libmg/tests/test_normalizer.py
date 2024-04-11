import pytest

from libmg.language.grammar import mg_parser
from libmg.normalizer.normalizer import mg_normalizer, var_occurs


@pytest.mark.parametrize('var, expr, occurs', [('X', 'X', True), ('Y', 'X', False), ('X', 'a', False),
                                               ('X', 'X || b', True), ('X', 'Y || b', False), ('X', 'a || b', False),
                                               ('X', 'X ; b', True), ('X', 'Y ; b', False), ('X', 'a ; b', False),
                                               ('X', 'let Z = a in X || Z', True), ('X', 'let Y = a in (Y;b)', False), ('X', 'let X = a in (X;b)', True),
                                               ('X', 'def f(Z){Z || X} in f(a)', True), ('X', 'def f(Z){Z || b} in f(X)', True),
                                               ('X', 'def f(Z){Z || a} in f(b)', False), ('X', 'def f(X){Z || a} in f(b)', False),
                                               ('X', 'if X then a else b', True), ('X', 'if a then X else b', True),
                                               ('X', 'if a then Y else b', False), ('X', 'X + b', True), ('X', 'a + b', False), ('X', 'a + X', True),
                                               ('X', 'X*', True), ('X', 'a*', False), ('X', 'X*5', True), ('X', 'a*5', False)])
def test_var_occurs(var, expr, occurs):
    assert var_occurs(mg_parser.parse(expr), var) is occurs
    assert var_occurs(expr, var) is occurs


# @pytest.mark.parametrize('expr, normalized', [('fix X = true in (true || false)', 'true || false'),
#                                               ('fix X = true in (X || false)', 'fix X = true in X || false'),
#                                               ('fix Y = a in (fix X = b in (Y || c))', 'fix Y = a in Y || c'),
#                                               ('fix Y = a in (fix X = b in (X || c))', 'fix X = b in X || c'),
#                                               ('fix Y = a in (fix X = b in (X || Y))', 'fix Y = a in fix X = b in X || Y'),
#                                               ('repeat X = true in (true || false) for 3', 'true || false'),
#                                               ('repeat X = true in (X || false) for 3', 'repeat X = true in X || false for 3'),
#                                               ('repeat Y = a in (repeat X = b in (Y || c) for 3) for 5', 'repeat Y = a in Y || c for 5'),
#                                               ('repeat Y = a in (repeat X = b in (X || c) for 3) for 5', 'repeat X = b in X || c for 3'),
#                                               ('repeat Y = a in (repeat X = b in (X || Y) for 3) for 5', 'repeat Y = a in repeat X = b in X || Y for 3 for 5')
#                                               ])
# def test_fixpoint_no_vars(expr, normalized):
#     tree = mg_normalizer.normalize(mg_parser.parse(expr))
#     string = mg_normalizer.normalize(expr)
#     expected_tree = mg_parser.parse(normalized)
#     assert tree == expected_tree
#     assert string == normalized


@pytest.mark.parametrize('expr, normalized', [('a *', 'a *'), ('a * *', 'a *'), ('(a + b || c) * *', '(a + b || c) *'),
                                              ('a * 5 *', 'a *'), ('(a + b || c) * 5 *', '(a + b || c) *')])
def test_double_fix(expr, normalized):
    tree = mg_normalizer.normalize(mg_parser.parse(expr))
    string = mg_normalizer.normalize(expr)
    expected_tree = mg_parser.parse(normalized)
    assert tree == expected_tree
    assert string == normalized


@pytest.mark.parametrize('expr, normalized', [('a * 5', 'a * 5'), ('a * 5 * 4', 'a * 20'), ('(a + b || c) * 3 * 2', '(a + b || c) * 6'),
                                              ('a * * 5', 'a *'), ('(a + b || c) * * 5', '(a + b || c) *')])
def test_double_repeat(expr, normalized):
    tree = mg_normalizer.normalize(mg_parser.parse(expr))
    string = mg_normalizer.normalize(expr)
    expected_tree = mg_parser.parse(normalized)
    assert tree == expected_tree
    assert string == normalized


# @pytest.mark.parametrize('expr, normalized', [('fix X = true in X;X', 'fix X = true in X'), ('fix X = true in X;X;not', 'fix X = true in X ; not'),
#                                               ('fix X = true in X;X;X', 'fix X = true in X'), ('fix X = true in X;not;X', 'fix X = true in X'),
#                                               ('fix X = true in (X;(not || true || X);or)', 'fix X = true in ((X ; not) || (X ; true) || X) ; or'),
#                                               ('fix X = true in (X;if X then true else false)', 'fix X = true in if X then X ; true else X ; false'),
#                                               ('fix X = true in (X;if true then X else false)', 'fix X = true in if X ; true then X else X ; false'),
#                                               ('fix X = true in (a ; ((X || not);and))', 'fix X = true in (X || (a ; not)) ; and'),
#                                               ('fix X = a in (fix Y = b in ((X;Y) || X))', 'fix X = a in fix Y = b in Y || X'),
#                                               ('fix X = a in (fix Y = b in (X;Y))', 'fix Y = b in Y')])
# def test_composition_fixvars(expr, normalized):
#     tree = mg_normalizer.normalize(mg_parser.parse(expr))
#     string = mg_normalizer.normalize(expr)
#     expected_tree = mg_parser.parse(normalized)
#     assert tree == expected_tree
#     assert string == normalized
