from lark import Transformer, v_args
from lark.visitors import Visitor, Interpreter

from libmg.grammar import mg_parser


def var_occurs(tree, var):
    """
    Checks if the variable ``var`` occurs in the given expression tree.

    :param tree: A ``ParseTree`` that represents a mG expression.
    :param var: A variable symbol in mG.
    :return: True if ``var`` occurs in ``tree``, False otherwise.
    """
    class IsFixpoint(Transformer):

        @v_args(inline=True)
        def label(self, f):
            return str(f)

        def label_decl(self, _):
            return False

        @v_args(inline=True)
        def atom_op(self, label):
            return label == var

        def lhd(self, _):
            return False

        def rhd(self, _):
            return False

        def __default__(self, data, children, meta):
            return any(children)

        '''
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
        '''

    return IsFixpoint().transform(tree)


class CompositionAnalyzer(Interpreter):
    def __init__(self, left, var):
        self.left = left
        self.var = var
        super().__init__()

    def atom_op(self, tree):
        if str(tree.children[0].children[0]) == self.var:
            return tree
        else:
            expr = mg_parser.parse('left;right')
            expr.children[0] = self.left
            expr.children[1] = tree
            return expr

    def composition(self, tree):
        left, right = tree.children
        left = self.visit(left)
        self.left = left
        return self.visit(right)

    def __default__(self, tree):
        tree.children = self.visit_children(tree)
        return tree


class Normalizer(Interpreter):
    """
    Transforms a mG expression in normal form, by acting on its ``ParseTree``. Currently the only normalization is the
    removal of the fixpoint expressions in which the fixpoint variable does not occur in the fixpoint body.
    """
    def current_fix_var(self):
        return next(reversed(self.fix_var))

    def __init__(self):
        self.fix_var = {}
        super().__init__()

    def composition(self, tree):
        left, right = tree.children
        if len(self.fix_var) > 0 and var_occurs(right, self.current_fix_var()):
            left, right = tree.children
            return CompositionAnalyzer(left, self.current_fix_var()).visit(right)
            # return composition_normalizer(tree, self.current_fix_var())
            # tree.children = self.visit_children(tree)
            # return tree
        else:
            return self.__default__(tree)
            # tree.children = self.visit_children(tree)
            # return tree

    def fix(self, tree):
        fix_var, init, body = tree.children
        fix_var_name = str(fix_var.children[0])
        if not var_occurs(body, fix_var_name):
            body = self.visit(body)
            return body
        else:
            parsed_init = self.visit(init)
            self.fix_var[fix_var_name] = True
            parsed_body = self.visit(body)
            self.fix_var.pop(fix_var_name)
            tree.children = [fix_var, parsed_init, parsed_body]
            return tree

    def __default__(self, tree):
        tree.children = self.visit_children(tree)
        return tree
