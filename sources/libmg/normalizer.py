from lark import Transformer, v_args
from lark.reconstruct import Reconstructor
from lark.visitors import Interpreter, Visitor


def is_fixpoint(tree, fix_var=None):
    class IsFixpoint(Transformer):
        @v_args(inline=True)
        def label(self, f):
            return str(f)

        @v_args(inline=True)
        def label_decl(self, var):
            return str(var)

        @v_args(inline=True)
        def atom_op(self, label):
            return label == fix_var

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

    return IsFixpoint().transform(tree) if fix_var is not None else False


def fixpoint_no_vars(tree):
    return tree.children[-1]


class Normalizer(Visitor):

    def __init__(self, parser):
        super().__init__()
        self.parser = parser
        self.reconstructor = Reconstructor(parser)
        self.fixpoint_vars = []

    @v_args(tree=True)
    def fix(self, tree):
        if not is_fixpoint(tree.children[-1], str(tree.children[0].children[0])):
            new_expr = fixpoint_no_vars(tree)
            tree.data = new_expr.data
            tree.children = new_expr.children

    def __default__(self, tree):
        return tree
