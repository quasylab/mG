"""Defines a normalizer to transform mG expressions in normal form.

This module defines the functions to normalize mG expressions when provided either as strings or trees. It also provides a function to check for the
occurrence of a variable in an expression.

The module contains the following functions:
- ``var_occurs(var, tree)``

The module contains the following classes:
- ``Normalizer``

The module contains the following objects:
- ``mg_normalizer``
"""
from copy import deepcopy
from typing import Any

from lark import Transformer, v_args, Tree, Token
from lark.visitors import Interpreter
from functools import singledispatchmethod, singledispatch

from libmg.language.grammar import mg_parser, mg_reconstructor


class Normalizer(Interpreter[Token, Tree]):
    """Normalizer for mG expression trees.

    Transforms a mG expression in normal form, by acting on its parse tree. Currently, this class only removes
    fixpoint/repeat expressions in which the fixpoint variable does not occur in the fixpoint body, and rewrites expressions so that fixpoint variables never
    occur on the right hand side of a sequential composition expression.

    Attributes:
        fix_var: The stack of fixpoint variables encountered during traversal of the tree.
    """

    class _CompositionAnalyzer(Interpreter):
        """Rewrites a sequential composition expression so that the current fixpoint variable does not occur on the left-hand side.

        Attributes:
            left: The tree of the left-hand side of the sequential composition expression.
            var: The fixpoint variable.
        """

        def __init__(self, left: Tree, var: str):
            """Initializes the instance with the left-hand side tree and the variable.

            Args:
                left: The tree of the left-hand side of the sequential composition expression.
                var: The fixpoint variable.
            """
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

        def sequential_composition(self, tree):
            left, right = tree.children
            left = self.visit(left)
            self.left = left
            return self.visit(right)

        def __default__(self, tree):
            tree.children = self.visit_children(tree)
            return tree

    def _current_fix_var(self) -> str:
        """Returns the current fixpoint variable.
        """
        return self.fix_var[-1]

    def __init__(self):
        """Initializes the instance.
        """
        self.fix_var: list[str] = []
        super().__init__()

    @singledispatchmethod
    def normalize(self, expr: Any) -> Any:
        """Normalizes a mG expression.

        If the expression is provided as a parse tree, a normalized parse tree is returned.
        If the expression is provided as string, a normalized string is returned.

        Examples:
            >>> mg_normalizer.normalize('fix Y = a in (fix X = b in (Y || c))')
            "fix Y = a in (Y || c)"

        Args:
            expr: The expression to normalize.

        Returns:
            The normalized expression.

        """
        raise NotImplementedError(f"Cannot format value of type {type(expr)}")

    @normalize.register
    def _(self, expr: Tree) -> Tree:
        prev_pass = None
        curr_pass = expr
        while prev_pass != curr_pass:
            prev_pass = deepcopy(curr_pass)
            curr_pass = self.visit(curr_pass)
        return curr_pass

    @normalize.register
    def _(self, expr: str) -> str:
        normalized_tree = self.normalize(mg_parser.parse(expr))
        assert isinstance(normalized_tree, Tree)
        return mg_reconstructor.reconstruct(normalized_tree)

    def sequential_composition(self, tree):
        left, right = tree.children
        if len(self.fix_var) > 0 and var_occurs(right, self._current_fix_var()):
            left, right = tree.children
            return self._CompositionAnalyzer(left, self._current_fix_var()).visit(right)
        else:
            return self.__default__(tree)

    def fix(self, tree):
        fix_var, init, body = tree.children
        fix_var_name = str(fix_var.children[0])
        if not var_occurs(body, fix_var_name):
            body = self.visit(body)
            return body
        else:
            parsed_init = self.visit(init)
            self.fix_var.append(fix_var_name)
            parsed_body = self.visit(body)
            self.fix_var.pop()
            tree.children = [fix_var, parsed_init, parsed_body]
            return tree

    def repeat(self, tree):
        fix_var, init, body, iters = tree.children
        fix_var_name = str(fix_var.children[0])
        if not var_occurs(body, fix_var_name):
            body = self.visit(body)
            return body
        else:
            parsed_init = self.visit(init)
            self.fix_var.append(fix_var_name)
            parsed_body = self.visit(body)
            self.fix_var.pop()
            tree.children = [fix_var, parsed_init, parsed_body, iters]
            return tree

    def __default__(self, tree):
        tree.children = self.visit_children(tree)
        return tree


@singledispatch
def var_occurs(expr: Tree | str, var: str) -> bool:
    """Checks if the variable occurs in the expression.

    The expression can be either a string or a parse tree.

    Examples:
        >>> var_occurs('let Y = a in (Y;b)','X')
        False
        >>> var_occurs('(a || X);b','X')
        True
        >>> var_occurs(mg_parser.parse('let Y = a in (Y;b)'),'X')
        False
        >>> var_occurs(mg_parser.parse('(a || X);b'),'X')
        True

    Args:
        expr: The expression to check.
        var: The variable.

    Returns:
        True if the variable occurs in the expression, false otherwise.
    """
    raise NotImplementedError(f"Cannot format value of type {type(expr)}")


@var_occurs.register
def _(expr: Tree, var: str) -> bool:
    class VarOccurs(Transformer[Token, bool]):
        """Checks if ``var`` occurs in the parse tree.
        """

        @v_args(inline=True)
        def label(self, f):
            return str(f)

        @v_args(inline=True)
        def atom_op(self, label):
            return label == var

        def lhd(self, _):
            return False

        def rhd(self, _):
            return False

        def __default__(self, data, children, meta):
            return any(filter(lambda x: isinstance(x, bool), children))

    return VarOccurs().transform(expr)


@var_occurs.register
def _(expr: str, var: str) -> bool:
    return var_occurs(mg_parser.parse(expr), var)


mg_normalizer = Normalizer()
"""
Normalizer instance on which to call ``normalize``.

Examples:
    >>> mg_normalizer.normalize('fix Y = a in (fix X = b in (Y || c))')
    "fix Y = a in (Y || c)"
"""
