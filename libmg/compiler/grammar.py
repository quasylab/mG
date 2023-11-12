"""Defines the grammar, parser and reconstructor of the mG language

This module defines the objects holding the grammar, the LALR parser, and the (experimental) reconstructor.
The reserved words of the grammar are: || fix let if then else def in repeat for
The reserved symbols of the grammar are: , | < > = ( ) ; [ ] # { }
Variables should not be named with two initial underscores __, e.g. don't name variables such as __X.

The module contains the following classes:
- ``MGReconstructor``

The module contains the following objects:
- ``mg_grammar``
- ``mg_parser``
- ``mg_reconstructor``
"""
from typing import Iterable, Callable

# TODO: add white spaces and test string normalization (finish in test_explainer and refactor the class)
# TODO: tree to parse tree
# TODO: examples and doctest
from lark import Lark, ParseTree
from lark.reconstruct import Reconstructor
from lark.utils import is_id_continue

mg_grammar = r"""
                ?gnn_formula: label                                                                               -> atom_op
                         | "<" label? "|" label                                                                   -> lhd
                         | "|" label? ">" label                                                                   -> rhd
                         | label "(" (c_formula ",")* c_formula ")"                                               -> fun_call
                         | "let" (label_decl "=" c_formula ",")* label_decl "=" c_formula "in" c_formula          -> local_var_expr
                         | "def" label_decl "(" (label_decl ",")* label_decl ")" "{" c_formula "}" "in" c_formula -> fun_def
                         | "if" c_formula "then" c_formula "else" c_formula                                       -> ite
                         | "fix" label_decl "=" c_formula "in" c_formula                                          -> fix
                         | "repeat" label_decl "=" c_formula "in" c_formula "for" NUMBER                          -> repeat
                         | "(" c_formula ")"

                ?c_formula: gnn_formula
                         | gnn_formula ";" c_formula                                                              -> sequential_composition
                         | gnn_formula ( "||" gnn_formula )+                                                      -> parallel_composition

                ?start: c_formula

                label: /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/
                            |  FUNC_GEN

                FUNC_GEN: /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/ "[" /[^\]\[]+/ "]"

                label_decl: /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/

                COMMENT: "#" /[^\n]/*

                %ignore COMMENT
                %import common.WS
                %import common.NUMBER
                %import common.UCASE_LETTER
                %ignore WS
                """

mg_parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')


class MGReconstructor(Reconstructor):
    """Reconstructor for the mG language.

    The reconstructor transforms a parse tree into the corresponding string, reversing the operation of parsing. This implementation is a slight
    modification of Lark's ``Reconstructor`` class to allow the addition of white spaces between any tokens: even those that contain non alphanumeric symbols.
    """
    def reconstruct(self, tree: ParseTree, postproc: Callable[[Iterable[str]], Iterable[str]] | None = None, insert_spaces: bool = True) -> str:
        """Reconstructs a string from a parse tree.

        Args:
            tree: The tree to reconstruct.
            postproc: The post-processing function to apply to each word of the reconstructed string.
            insert_spaces: If true, add spaces between any two words of the reconstructed string.

        Returns:
            The reconstructed string.
        """
        x = self._reconstruct(tree)
        if postproc:
            x = postproc(x)
        y = []
        prev_item = ''
        for item in x:
            if insert_spaces and prev_item and item:
                if item == ';' or prev_item == ';':
                    y.append(' ')
                elif (prev_item == '<' or prev_item == '|' or item == '|' or prev_item == '>' or item == '>' or prev_item == '(' or item == ')'
                        or prev_item == '{' or item == '}' or (item == '(' and is_id_continue(prev_item) and prev_item != 'in') or item == ','):
                    pass
                else:
                    y.append(' ')
            y.append(item)
            prev_item = item
        return ''.join(y)


mg_reconstructor = MGReconstructor(mg_parser)
