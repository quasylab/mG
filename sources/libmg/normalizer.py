from lark import Transformer, v_args
from lark.reconstruct import Reconstructor


def is_fixpoint(tree):
    class IsFixpoint(Transformer):
        def fun_app(self, _):
            return False

        def lhd(self, _):
            return False

        def rhd(self, _):
            return False

        def variable(self, _):
            return True

        @v_args(inline=True)
        def composition(self, phi, psi):
            if phi or psi:
                return True
            else:
                return False

        def mu_formula(self, _):
            return False

        def nu_formula(self, _):
            return False

        def parallel(self, args):
            for arg in args:
                if arg is True:
                    return True
            return False

    return IsFixpoint().transform(tree)


def fixpoint_no_vars(tree):
    return tree.children[2]


def seq_rhs_fixpoint(tree, reconstructor, parser):
    lhs = reconstructor.reconstruct(tree.children[0])
    if tree.children[1].data == 'composition':
        child = tree.children[1].children[0]
        rhs = '(' + reconstructor.reconstruct(child) + ')'
        if child.data == 'variable':
            new_expr = rhs
        else:
            new_expr = "(" + lhs + " ; " + rhs + ")"
        tree.children[1].children[0] = parser.parse(new_expr)
        return tree.children[1]
    else:
        new_expr = []
        for child in tree.children[1].children:
            rhs = reconstructor.reconstruct(child)
            if child.data == 'variable':
                new_expr.append(rhs)
            else:
                new_expr.append("(" + lhs + " ; " + rhs + ")")
        return parser.parse('||'.join(new_expr))


class Normalizer(Transformer):

    def __init__(self, parser):
        super().__init__()
        self.parser = parser
        self.reconstructor = Reconstructor(parser)

    @v_args(tree=True)
    def composition(self, tree):
        if is_fixpoint(tree.children[1]):
            return self.transform(seq_rhs_fixpoint(tree, self.reconstructor, self.parser))
        else:
            return tree

    @v_args(tree=True)
    def mu_formula(self, tree):
        if not is_fixpoint(tree.children[2]):
            return self.transform(fixpoint_no_vars(tree))
        else:
            return tree

    @v_args(tree=True)
    def nu_formula(self, tree):
        if not is_fixpoint(tree.children[2]):
            return self.transform(fixpoint_no_vars(tree))
        else:
            return tree
