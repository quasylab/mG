from lark import Transformer, v_args
from lark.visitors import Visitor


def is_free(tree, var=None):
    """
    Checks if the variable ``var`` is free in the given expression tree.

    :param tree: A ``ParseTree`` that represents a mG expression.
    :param var: A variable symbol in mG.
    :return: True if ``var`` is free in ``tree``, False otherwise.
    """
    class IsFixpoint(Transformer):
        @v_args(inline=True)
        def label(self, f):
            return str(f)

        @v_args(inline=True)
        def label_decl(self, var):
            return str(var)

        @v_args(inline=True)
        def atom_op(self, label):
            return label == var

        def lhd(self, _):
            return False

        def rhd(self, _):
            return False

        @v_args(inline=True)
        def composition(self, phi, psi):
            return phi or psi

        def parallel(self, args):
            for arg in args:
                if arg is True:
                    return True
            return False

        def fun_def(self, args):
            return args[-1]

        def var_def(self, args):
            return args[-1]

        def fun_call(self, args):
            for arg in args[1:]:  # first arg is the function name
                if arg is True:
                    return True
            return False

        def local_var_expr(self, args):
            return args[-1]

        @v_args(inline=True)
        def ite(self, test, iftrue, iffalse):
            return test or iftrue or iffalse

        @v_args(inline=True)
        def fix(self, variable_decl, initial_var_gnn, body):
            return initial_var_gnn or body

    return IsFixpoint().transform(tree) if var is not None else False


def fixpoint_no_vars(tree):
    """
    Transforms a mG fixpoint expression in which the fixpoint variable doesn't occur in its body.

    :param tree: A ``ParseTree`` representing a mG fixpoint expression.
    :return: A ``ParseTree`` consisting of the body of the input fixpoint expression.
    """
    return tree.children[-1]


class Normalizer(Visitor):
    """
    Transforms a mG expression in normal form, by acting on its ``ParseTree``. Currently the only normalization is the
    removal of the fixpoint expressions in which the fixpoint variable does not occur in the fixpoint body.
    """

    def __init__(self):
        super().__init__()

    @v_args(tree=True)
    def fix(self, tree):
        if not is_free(tree.children[-1], str(tree.children[0].children[0])):
            new_expr = fixpoint_no_vars(tree)
            tree.data = new_expr.data
            tree.children = new_expr.children

    def __default__(self, tree):
        return tree
