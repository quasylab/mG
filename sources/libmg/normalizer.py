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


'''
def seq_rhs_fixpoint(tree, fix_var, reconstructor, parser):
    lhs = reconstructor.reconstruct(tree.children[0])
    if tree.children[1].data == 'composition':
        child = tree.children[1].children[0]
        rhs = '(' + reconstructor.reconstruct(child) + ')'
        if child.data == 'atom_op' and child.children[0].children[0] == fix_var:
            new_expr = rhs
        else:
            new_expr = "(" + lhs + " ; " + rhs + ")"
        tree.children[1].children[0] = parser.parse(new_expr)
        return tree.children[1]
    elif tree.children[1].data == 'parallel':
        new_expr = []
        for child in tree.children[1].children:
            rhs = reconstructor.reconstruct(child)
            if child.data == 'atom_op' and child.children[0].children[0] == fix_var:
                new_expr.append(rhs)
            else:
                new_expr.append("(" + lhs + " ; " + rhs + ")")
        return parser.parse('||'.join(new_expr))
    elif tree.children[1].data == 'fun_call':
        return tree.children[1]
    elif tree.children[1].data == 'local_var_expr':
        pass
    elif tree.children[1].data == 'ite':
        pass
    elif tree.children[1].data == 'atom_op' and tree.children[0].children[0] == fix_var:
        return tree.children[1]
    else:
        raise ValueError('Unexpected fixpoint expression:', tree.children[1].data)
'''


class Normalizer(Interpreter):

    def __init__(self, parser):
        super().__init__()
        self.parser = parser
        self.reconstructor = Reconstructor(parser)
        self.fixpoint_vars = []

    # @v_args(inline=True)
    # def label(self, f):
    #    return str(f)

    @v_args(inline=True)
    def label_decl(self, var):
        return str(var)

    # @v_args(tree=True)
    # def composition(self, tree):
    # if is_fixpoint(tree.children[1], self.fixpoint_vars[-1]):
    #   return self.visit(seq_rhs_fixpoint(tree, self.fixpoint_vars[-1], self.reconstructor, self.parser))
    # else:
    #    return tree

    @v_args(tree=True)
    def fix(self, tree):
        # self.fixpoint_vars.append(self.visit(tree.children[0]))
        if not is_fixpoint(tree.children[-1], self.visit(tree.children[0])):
            self.visit(fixpoint_no_vars(tree))
            # self.fixpoint_vars.pop()
            return self.visit(fixpoint_no_vars(tree))
        else:
            # body = self.visit(tree.children[-1])
            # self.fixpoint_vars.pop()
            # tree.children[-1] = body
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
