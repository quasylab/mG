import pytest

from libmg.language.grammar import mg_parser
from libmg.normalizer.normalizer import mg_normalizer, var_occurs, sub_exp_extract, subst, exp_len


@pytest.mark.parametrize('var, expr, occurs', [('X', 'X', True), ('Y', 'X', False), ('X', 'a', False),
                                               ('X', 'X || b', True), ('X', 'Y || b', False), ('X', 'a || b', False),
                                               ('X', 'X ; b', True), ('X', 'Y ; b', False), ('X', 'a ; b', False),
                                               ('X', 'let Z = a in X || Z', True), ('X', 'let Y = a in (Y;b)', False), ('X', 'let X = a in (X;b)', True),
                                               ('X', 'def f(Z){Z || X} in f(a)', True), ('X', 'def f(Z){Z || b} in f(X)', True),
                                               ('X', 'def f(Z){Z || a} in f(b)', False), ('X', 'def f(X){Z || a} in f(b)', False),
                                               ('X', 'if X then a else b', True), ('X', 'if a then X else b', True),
                                               ('X', 'if a then Y else b', False), ('X', 'X | b', True), ('X', 'a | b', False), ('X', 'a | X', True),
                                               ('X', 'X*', True), ('X', 'a*', False), ('X', 'rep X for 5', True), ('X', 'rep a for 5', False)])
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


@pytest.mark.parametrize('expr, normalized', [('a *', 'a *'), ('a * *', 'a *'), ('(a | b || c) * *', '(a|b || c) *'),
                                              ('(rep a for 5) *', 'a *'), ('(rep (a | b || c) for 5) *', '(a|b || c) *')])
def test_double_fix(expr, normalized):
    tree = mg_normalizer.normalize(mg_parser.parse(expr))
    string = mg_normalizer.normalize(expr)
    expected_tree = mg_parser.parse(normalized)
    assert tree == expected_tree
    assert string == normalized


@pytest.mark.parametrize('expr, normalized', [('rep a for 5', 'rep a for 5'), ('rep rep a for 5 for 4', 'rep a for 20'),
                                              ('rep rep (a | b || c) for 3 for 2', 'rep (a|b || c) for 6'),
                                              ('rep a * for 5', 'a *'), ('rep (a | b || c) * for 5', '(a|b || c) *')])
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

@pytest.mark.parametrize('expr, sub_exp_list', [('X', []), ('i', ['i']), ('a', ['a']), ('|b>a', ['|b>a']), ('<b|a', ['<b|a']), ('(c || X;|>b);and', ['c']),
                                                ('X;a', []), ('a;b', ['a;b']), ('X;b;c', []), ('a;b;c', ['a;b;c']),
                                                ('a || b || c', ['a || b || c']), ('a || X || c', ['a', 'c']),
                                                ('(a || X || c) ; b', ['a', 'c']), ('a || X;b || c', ['a', 'c']),
                                                ('a | b', ['a|b']), ('a | X', ['a']), ('X | a', ['a']), ('X;b | X;a', []),
                                                ('(a | X);b', ['a']), ('a*', ['a*']), ('X*', []), ('(X || a)*', ['a']),
                                                ('if a then b else c', ['if a then b else c']), ('if X then a else b', ['a', 'b']),
                                                ('if a then X else c', ['a', 'c']), ('rep a for 2', ['rep a for 2']),
                                                ('rep X for 2', []), ('rep (X || a) for 2', ['a'])])
def test_sub_exp(expr, sub_exp_list):
    assert sub_exp_extract(expr, 'X') == [mg_parser.parse(x) for x in sub_exp_list]


@pytest.mark.parametrize('expr1, expr2', [('let X = a in X', 'a'), ('let Y = b || c in a;X;Y', 'a;X;(b || c)'),
                                          ('a;(let X = b in (c;X*));d', 'a;(c;b*);d'), ('let X = a, Y = b in (X || Y);c', '(a || b) ; c'),
                                          ('def f(X,Y){(X || Y) ; a} in f(b, c)', '(b || c) ; a'),
                                          ('def test(X, Y){(Y || X);and;not} in test(test(a, b), b)', '(b || (b || a);and;not);and;not')])
def test_preprocessor(expr1, expr2):
    assert mg_normalizer.normalize(mg_parser.parse(expr1)) == mg_parser.parse(expr2)


@pytest.mark.parametrize('expr1, bindings, expr2', [('a;b || c;d', {mg_parser.parse('a'): mg_parser.parse('p1')}, 'p1;b || c;d'),
                                                    ('a;b || c;d', {mg_parser.parse('a;b'): mg_parser.parse('p1')}, 'p1 || c;d'),
                                                    ('a;b || c;d', {mg_parser.parse('a;b || c;d'): mg_parser.parse('p1')}, 'p1'),
                                                    ('a;b || c;d', {mg_parser.parse('a'): mg_parser.parse('p1'),
                                                                    mg_parser.parse('d'): mg_parser.parse('p2')}, 'p1;b || c;p2')])
def test_subst(expr1, bindings, expr2):
    assert subst(mg_parser.parse(expr1), bindings) == mg_parser.parse(expr2)


@pytest.mark.parametrize('expr, ln', [('a', 1), ('|>a', 1), ('<|a', 1), ('i', 1), ('a;b', 1), ('a || b', 2), ('a || b || c', 3),
                                      ('(a || b);c', 1), ('(a || b);i', 2), ('(a || b);(c || d)', 2), ('a;(b || c)', 2),
                                      ('(true || a) ; (b | c)', 1)])
def test_expr_len(expr, ln):
    assert exp_len(mg_parser.parse(expr)) == ln


@pytest.mark.parametrize('expr1, expr2', [('fix X = a in (c || X;|>b);and', '(c || a);(p1 || ((p1 || p2;|>b);and))*;p2'),
                                          ('fix X = a in (X;b)', 'a;(i;b)*'),
                                          ('fix X = a in (if true then X else b)', '(true || b || a);(p1 || p2 || (if p1 then p3 else p2))*;p3')])
def test_fix(expr1, expr2):
    assert mg_normalizer.normalize(mg_parser.parse(expr1)) == mg_parser.parse(expr2)


@pytest.mark.parametrize('expr1, expr2', [('repeat X = a in (c || X;|>b);and for 3', '(c || a);(rep (p1 || ((p1 || p2;|>b);and)) for 3);p2'),
                                          ('repeat X = a in (X;b) for 3', 'a;(rep (i;b) for 3)')])
def test_repeat(expr1, expr2):
    assert mg_normalizer.normalize(mg_parser.parse(expr1)) == mg_parser.parse(expr2)
