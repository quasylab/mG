from lark import Transformer, v_args
from lark.reconstruct import Reconstructor
from lark.visitors import Interpreter


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

        def fix(self, _):
            return False

    return IsFixpoint().transform(tree) if fix_var is not None else False


def fixpoint_no_vars(tree):
    return tree.children[-1]


class Normalizer(Interpreter):

    def __init__(self, parser):
        super().__init__()
        self.parser = parser
        self.reconstructor = Reconstructor(parser)
        self.fixpoint_vars = []

    @v_args(inline=True)
    def label_decl(self, var):
        return str(var)

    @v_args(tree=True)
    def fix(self, tree):
        if not is_fixpoint(tree.children[-1], self.visit(tree.children[0])):
            self.visit(fixpoint_no_vars(tree))
            return self.visit(fixpoint_no_vars(tree))
        else:
            return tree

    @v_args(tree=True)
    def fun_def(self, tree):
        tree.children[-1] = self.visit(tree.children[-1])
        return tree

    @v_args(tree=True)
    def var_def(self, tree):
        tree.children[-1] = self.visit(tree.children[-1])
        return tree

    def __default__(self, tree):
        return tree
