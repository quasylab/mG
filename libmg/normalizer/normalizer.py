"""Defines a normalizer to transform mG expressions in normal form.

This module defines the functions to normalize mG expressions when provided either as strings or trees. It also provides a function to check for the
occurrence of a variable in an expression.

The module contains the following functions:
- ``var_occurs(var, tree)``
- ``sub_exp_extract(expr, var)``
- ``subst(expr, bindings)``
- ``exp_len(expr)``

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
        bound_vars: The dictionary of local variables encountered during traversal of the tree.
        bound_funs: The dictionary of local functions encountered during traversal of the tree.
    """

    def __init__(self):
        """Initializes the instance.
        """
        self.bound_vars = {}
        self.bound_funs = {}

    @singledispatchmethod
    def normalize(self, expr: Any) -> Any:
        """Normalizes a mG expression.

        If the expression is provided as a parse tree, a normalized parse tree is returned.
        If the expression is provided as string, a normalized string is returned.

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

    def label_decl(self, label):
        return str(label.children[0])

    def label(self, label):
        return str(label.children[0])

    def atom_op(self, op):
        parsed_op, = self.visit_children(op)
        if parsed_op in self.bound_vars:
            return self.bound_vars[parsed_op]
        else:
            return op

    def rhd(self, tree):
        return tree

    def lhd(self, tree):
        return tree

    def fun_call(self, tree):
        args = tree.children
        function_name = self.visit(args[0])
        arguments = [self.visit(arg) for arg in args[1:]]

        if function_name in self.bound_funs:
            function_args, function_body = self.bound_funs[function_name][0], deepcopy(self.bound_funs[function_name][1])
            matched_args = {vname: val for vname, val in zip(function_args, arguments)}

            self.bound_vars |= matched_args  # add the deferred function vars to var input
            eval_body = self.visit(function_body)  # now visit the function body
            for k in matched_args:  # eliminate the variables of this function from var_input
                self.bound_vars.pop(k)
            return eval_body
        else:  # is an operator, subst variables
            new_tree = deepcopy(tree)
            new_tree.children[1:] = arguments
            tree = subst(new_tree, {mg_parser.parse(var): val for var, val in self.bound_vars.items()})
            return tree

    def local_var_expr(self, tree):
        args = tree.children
        local_vars = []
        for i in range(len(args[0:-1]) // 2):
            var = self.visit(args[i * 2])
            val = args[i * 2 + 1]
            self.bound_vars[var] = val
            local_vars.append(var)
        eval_body = self.visit(tree.children[-1])
        for k in local_vars:  # eliminate the variables defined by this expression
            self.bound_vars.pop(k)
        return eval_body

    def fun_def(self, tree):
        args = tree.children
        function_name = self.visit(args[0])
        var_input = []
        for i in range(len(args[1:-2])):
            var_name = self.visit(args[1 + i])
            var_input.append(var_name)
        val = args[-2]  # we are not evaluating the function right now
        body = args[-1]
        self.bound_funs[function_name] = (var_input, val)
        return self.visit(body)

    def star(self, tree):
        body, = tree.children
        if body.data == 'star':
            body = self.visit(body)
            return body
        elif body.data == 'rep':
            body.data = 'star'
            body.children = body.children[:-1]
            body = self.visit(body)
            return body
        else:
            parsed_body = self.visit(body)
            tree.children = [parsed_body]
            return tree

    def rep(self, tree):
        body, iters = tree.children
        if body.data == 'star':
            body = self.visit(body)
            return body
        elif body.data == 'rep':
            body.children[-1] = Token('NUMBER', int(iters) * int(body.children[-1]))
            body = self.visit(body)
            return body
        else:
            parsed_body = self.visit(body)
            tree.children = [parsed_body, iters]
            return tree

    def fix(self, tree):
        var, init, body = tree.children
        var = self.visit(var)
        init = self.visit(init)
        body = self.visit(body)
        precomputed_terms = list(sub_exp_extract(body, var)) + [init]
        precomputed_terms_sizes = [exp_len(term) for term in precomputed_terms]
        total_size = sum(precomputed_terms_sizes)
        new_body = None
        if len(precomputed_terms) == 1:
            bindings = {mg_parser.parse(var): mg_parser.parse('i')}
            new_body = subst(body, bindings)
            return Tree(data='sequential_composition', children=[precomputed_terms[0], Tree(data='star', children=[new_body])])
        else:
            par_init = Tree(data='parallel_composition', children=precomputed_terms)
            i = 1
            p_list = []  # type: list[str]
            final_p = None
            while i <= total_size:
                next_p = 'p'
                next_size = precomputed_terms_sizes.pop(0)
                if next_size == 1:
                    next_p += str(i)
                else:
                    next_p += str(i) + '-' + str(i + next_size)
                if i == total_size:
                    final_p = next_p
                    bindings = {term: mg_parser.parse(proj) for term, proj in zip(precomputed_terms + [mg_parser.parse(var)], p_list + [final_p, final_p])}
                    new_body = subst(body, bindings)
                    break
                p_list.append(next_p)
                i = i + next_size
            assert final_p is not None
            par_middle = Tree(data='star', children=[Tree(data='parallel_composition', children=[mg_parser.parse(p) for p in p_list] + [new_body])])
            return Tree(data='sequential_composition',
                        children=[Tree(data='sequential_composition', children=[par_init, par_middle]), mg_parser.parse(final_p)])

    def repeat(self, tree):
        var, init, body, k = tree.children
        var = self.visit(var)
        init = self.visit(init)
        body = self.visit(body)
        precomputed_terms = list(sub_exp_extract(body, var)) + [init]
        precomputed_terms_sizes = [exp_len(term) for term in precomputed_terms]
        total_size = sum(precomputed_terms_sizes)
        if len(precomputed_terms) == 1:
            bindings = {mg_parser.parse(var): mg_parser.parse('i')}
            new_body = subst(body, bindings)
            return Tree(data='sequential_composition', children=[precomputed_terms[0], Tree(data='rep', children=[new_body, k])])
        else:
            par_init = Tree(data='parallel_composition', children=precomputed_terms)
            i = 1
            p_list = []  # type: list[str]
            final_p = None
            new_body = None
            while i <= total_size:
                next_p = 'p'
                next_size = precomputed_terms_sizes.pop(0)
                if next_size == 1:
                    next_p += str(i)
                else:
                    next_p += str(i) + '-' + str(i + next_size)
                if i == total_size:
                    final_p = next_p
                    bindings = {term: mg_parser.parse(proj) for term, proj in zip(precomputed_terms + [mg_parser.parse(var)], p_list + [final_p, final_p])}
                    new_body = subst(body, bindings)
                    break
                p_list.append(next_p)
                i = i + next_size
            assert final_p is not None
            par_middle = Tree(data='rep', children=[Tree(data='parallel_composition', children=[mg_parser.parse(p) for p in p_list] + [new_body]), k])
            return Tree(data='sequential_composition',
                        children=[Tree(data='sequential_composition', children=[par_init, par_middle]), mg_parser.parse(final_p)])

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


@singledispatch
def sub_exp_extract(expr: str | Tree, var: str) -> list[Tree]:
    """Obtains the list of the unique largest sub-expressions of ``expr`` that do not depend on the value of ``var``.

    The expression can be either a string or a parse tree.

    Examples:
        >>> sub_exp_extract('(a || X || c);and', 'X')
        [Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')])]), Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'c')])])]
        >>> sub_exp_extract('(a || b);c','X')
        [Tree('sequential_composition', [Tree('parallel_composition', [Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')])]), ...
        >>> sub_exp_extract(mg_parser.parse('(a || X || c);and'),'X')
        [Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')])]), Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'c')])])]
        >>> sub_exp_extract(mg_parser.parse('(a || b);c'),'X')
        [Tree('sequential_composition', [Tree('parallel_composition', [Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')])]), ...

    Args:
        expr: The expression to from which to obtain the sub-expressions.
        var: The variable.

    Returns:
        The list of (unique) sub-expressions.
    """
    raise NotImplementedError(f"Cannot format value of type {type(expr)}")


@sub_exp_extract.register
def _(expr: str, var: str) -> list[Tree]:
    return sub_exp_extract(mg_parser.parse(expr), var)


@sub_exp_extract.register
def _(expr: Tree, var: str) -> list[Tree]:
    class SubExpExtractor(Transformer):
        """
        Obtains all the sub-expressions of the given Tree that do not depend on the value of ``var``
        """
        class FixExp:
            def __init__(self, args):
                self.args = args

        class Exp:
            def __init__(self, args):
                self.args = args

        def __init__(self, var):
            """Initializes the instance.
            """
            self.var = var
            super().__init__()

        @v_args(tree=True)
        def id(self, tree):
            return self.Exp(tree)

        @v_args(tree=True)
        def atom_op(self, tree):
            label = str(tree.children[0].children[0])
            if label == self.var:
                return self.FixExp([])
            else:
                return self.Exp(tree)

        @v_args(tree=True)
        def lhd(self, tree):
            return self.Exp(tree)

        @v_args(tree=True)
        def rhd(self, tree):
            return self.Exp(tree)

        def sequential_composition(self, args):
            left, right = args
            if isinstance(left, self.FixExp):
                return left
            else:
                return self.Exp(Tree(data='sequential_composition', children=[left.args, right.args]))

        def parallel_composition(self, args):
            arglist = []  # type: list[Tree]
            isfix = False
            for exp in args:
                if isinstance(exp, self.FixExp):
                    isfix = True
                    arglist = arglist + exp.args
                else:
                    arglist.append(exp.args)

            if not isfix:
                return self.Exp(Tree(data='parallel_composition', children=arglist))
            else:
                return self.FixExp(arglist)

        def choice(self, args):
            left, right = args
            if isinstance(left, self.FixExp) or isinstance(right, self.FixExp):
                return self.FixExp((left.args if isinstance(left.args, list) else [left.args]) + (right.args if isinstance(right.args, list) else [right.args]))
            else:
                return self.Exp(Tree(data='choice', children=[left.args, right.args]))

        def star(self, args):
            body, = args
            if isinstance(body, self.FixExp):
                return self.FixExp(body.args)
            else:
                return self.Exp(Tree(data='star', children=[body.args]))

        def rep(self, args):
            body, k = args
            if isinstance(body, self.FixExp):
                return self.FixExp(body.args)
            else:
                return self.Exp(Tree(data='rep', children=[body.args, k]))

        # def fix(self, args):
        #     var, init, body = args
        #     var = str(var.children[0])
        #     if not isinstance(init, self.FixExp) and not isinstance(body, self.FixExp):
        #         return self.Exp(['fix ' + var + ' = ' + init.args[0] + ' in ' + body.args[0]])
        #     else:
        #         filtered_body_args = []
        #         extractor = SubExpExtractor(var)
        #         for arg in body.args:
        #             sub_exps = extractor.transform(mg_parser.parse(arg)).args
        #             filtered_body_args += sub_exps
        #         return self.FixExp(init.args + filtered_body_args)
        #
        # def rep(self, args):
        #     var, init, body, k = args
        #     var = str(var.children[0])
        #     if not isinstance(init, self.FixExp) and not isinstance(body, self.FixExp):
        #         return self.Exp(['repeat ' + var + ' = ' + init.args[0] + ' in ' + body.args[0] + ' for ' + k])
        #     else:
        #         filtered_body_args = []
        #         extractor = SubExpExtractor(var)
        #         for arg in body.args:
        #             sub_exps = extractor.transform(mg_parser.parse(arg)).args
        #             filtered_body_args += sub_exps
        #         return self.FixExp(init.args + filtered_body_args)

        def ite(self, args):
            test, iftrue, iffalse = args
            if not isinstance(iftrue, self.FixExp) and not isinstance(iffalse, self.FixExp) and not isinstance(test, self.FixExp):
                return self.Exp(Tree(data='ite', children=[test.args, iftrue.args, iffalse.args]))
            else:
                return self.FixExp((test.args if isinstance(test.args, list) else [test.args])
                                   + (iftrue.args if isinstance(iftrue.args, list) else [iftrue.args])
                                   + (iffalse.args if isinstance(iffalse.args, list) else [iffalse.args]))

    extractor = SubExpExtractor(var)
    sub_exps = extractor.transform(expr).args
    if isinstance(sub_exps, list):
        return list(dict.fromkeys(sub_exps))
    else:
        return [] if sub_exps is None else [sub_exps]


def subst(expr: Tree | Token, bindings: dict[Tree, Tree]) -> Any:
    """Substitutes the given ``bindings`` in ``expr``

    The function modifies ``expr`` in-place (return value may be tossed).

    Examples:
    >>> subst(mg_parser.parse('(a || c);and'), {mg_parser.parse('a'): mg_parser.parse('b')})
    Tree('sequential_composition', [Tree('parallel_composition', [Tree('atom_op', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'b')])]), ...

    Args:
        expr: The expression in to which substitute the bindings.
        bindings: The dictionary specifying what every ``key`` occurring in ``expr`` will be substituted with.

    Returns:
        The modified expression Tree.
    """
    if isinstance(expr, Token):
        return expr
    elif expr in bindings:
        return bindings[expr]
    else:
        expr.children = [subst(subexp, bindings) for subexp in expr.children]
        return expr


@singledispatch
def exp_len(expr: Tree | str) -> int:
    """Infers the length of the expression in terms of how many parallel compositions it contains.

    Examples:
    >>> exp_len('(a || c);and')
    1
    >>> exp_len('a || b || c')
    3

    Args:
        expr: The expression for which to compute the length.

    Returns:
        The number of labellings produced by the expression.
    """
    raise NotImplementedError()


@exp_len.register
def _(expr: Tree) -> int:
    class ExpLen(Interpreter):
        def __init__(self):
            super().__init__()
            self.current_len = 1

        def atom_op(self, tree):
            return 1

        def lhd(self, tree):
            return 1

        def rhd(self, tree):
            return 1

        @v_args(inline=True)
        def sequential_composition(self, left, right):
            self.current_len = self.visit(left)
            self.current_len = self.visit(right)
            return self.current_len

        def parallel_composition(self, tree):
            args = tree.children
            return sum([self.visit(arg) for arg in args])

        @v_args(inline=True)
        def choice(self, left, right):
            return self.visit(left)

        def __default__(self, tree):
            return self.current_len

    length = ExpLen().visit(expr)
    assert isinstance(length, int)
    return length


@exp_len.register
def _(expr: str) -> int:
    return exp_len(mg_parser.parse(expr))


mg_normalizer = Normalizer()
"""
Normalizer instance on which to call ``normalize``.

Examples:
    >>> mg_normalizer.normalize('fix Y = a in (fix X = b in (Y || c))')
    "fix Y = a in (Y || c)"
"""
