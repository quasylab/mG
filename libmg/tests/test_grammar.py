import pytest

from libmg.language.grammar import mg_parser, mg_reconstructor


@pytest.mark.parametrize('expr', ['a', '-', '!', 'b[5]', 'f[a]', '1'])
def test_atom_op(expr):
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'atom_op'
    assert rec_expr == expr
    assert rec_expr_tree == tree


@pytest.mark.parametrize('expr', ['<|a', '<|-', '<a|b', '<-|^', '<|a[1]', '<|-[b]', '<a[c]|b[1]', '<-[a]|^[1]'])
def test_lhd(expr):
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'lhd'
    assert rec_expr == expr
    assert rec_expr_tree == tree


@pytest.mark.parametrize('expr', ['|>a', '|>-', '|a>b', '|->^', '|>a[1]', '|>-[b]', '|a[c]>b[1]', '|-[a]>^[1]'])
def test_rhd(expr):
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'rhd'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_sequential_composition():
    expr = 'a ; b ; c ; d ; e'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'sequential_composition'
    assert tree == mg_parser.parse('(((a;b);c);d);e')
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_parallel_composition():
    expr = 'a || b || c || d || e'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)
    assert tree.data == 'parallel_composition'
    assert tree == mg_parser.parse('a || b || c || d || e')
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_parallel_nested_sequential():
    expr = 'a || b ; c ; d ; e || f'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'parallel_composition'
    assert tree == mg_parser.parse('a || (b ; c ; d ; e) || f')
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_sequential_nested_parallel():
    expr = 'a || b ; c || c || d ; e || e'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'parallel_composition'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_def_univariate_function():
    expr = 'def test(X) {(a || X) ; b} in test(c)'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'fun_def'
    assert len(tree.children) == 4
    assert tree.children[0].data == 'label_decl'
    assert tree.children[1].data == 'label_decl'
    assert tree.children[2].data == 'sequential_composition'
    assert tree.children[3].data == 'fun_call'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_def_multivariate_function():
    expr = 'def test(X, Y) {(Y || X) ; b} in test(c, a)'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'fun_def'
    assert len(tree.children) == 5
    assert tree.children[0].data == 'label_decl'
    assert tree.children[1].data == 'label_decl'
    assert tree.children[2].data == 'label_decl'
    assert tree.children[3].data == 'sequential_composition'
    assert tree.children[4].data == 'fun_call'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_local_var_expr():
    expr = 'let t1 = a ; b, t2 = c in t1 || t2'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'local_var_expr'
    assert len(tree.children) == 5
    assert tree.children[0].data == 'label_decl'
    assert tree.children[1].data == 'sequential_composition'
    assert tree.children[2].data == 'label_decl'
    assert tree.children[3].data == 'atom_op'
    assert tree.children[4].data == 'parallel_composition'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_choice():
    expr = "a|b"
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'choice'
    assert tree.children[0].data == 'atom_op'
    assert tree.children[1].data == 'atom_op'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_ite():
    expr = 'if a then b else c'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'ite'
    assert tree.children[0].data == 'atom_op'
    assert tree.children[1].data == 'atom_op'
    assert tree.children[2].data == 'atom_op'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_star():
    expr = '((a || b ; |>or) ; and) *'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'star'
    assert tree.children[0].data == 'sequential_composition'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_repeat():
    expr = 'rep ((a || b ; |>or) ; and) for 5'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'rep'
    assert tree.children[0].data == 'sequential_composition'
    assert int(tree.children[1]) == 5
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_fix():
    expr = 'fix X = b in (a || X ; |>or) ; and'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'fix'
    assert tree.children[2].data == 'sequential_composition'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_rep():
    expr = 'repeat X = b in (a || b ; |>or) ; and for 5'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'repeat'
    assert tree.children[2].data == 'sequential_composition'
    assert int(tree.children[-1]) == 5
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_id():
    expr = 'i'
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'id'
    assert rec_expr == expr
    assert rec_expr_tree == tree


def test_comment():
    expr = '''
    # this is a comment
    a || b;c # a parallel expression
    '''
    tree = mg_parser.parse(expr)
    rec_expr = mg_reconstructor.reconstruct(tree)
    rec_expr_tree = mg_parser.parse(rec_expr)

    assert tree.data == 'parallel_composition'
    assert rec_expr == 'a || b ; c'
    assert rec_expr_tree == tree
