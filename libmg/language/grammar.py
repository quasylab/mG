"""Defines the grammar, parser and reconstructor of the mG language

This module defines the objects holding the grammar, the LALR parser, and the (experimental) reconstructor.
The reserved words of the grammar are: || fix let if then else def in repeat for
The reserved symbols of the grammar are: , | < > = ( ) ; [ ] # { } *
Variables should not be named with two initial underscores __, e.g. don't name variables such as __X.

The module contains the following classes:
- ``MGReconstructor``

The module contains the following objects:
- ``mg_grammar``
- ``mg_reserved``
- ``mg_parser``
- ``mg_reconstructor``
"""
from typing import Iterable, Callable
from lark import Lark, ParseTree
from lark.reconstruct import Reconstructor

# top lowest priority
# bottom highest priority

mg_grammar = r"""
                ?formula: "if" formula "then" formula "else" formula                                            -> ite
                         | "let" (label_decl "=" formula ",")* label_decl "=" formula "in" formula              -> local_var_expr
                         | "def" label_decl "(" (label_decl ",")* label_decl ")" "{" formula "}" "in" formula   -> fun_def
                         | "fix" label_decl "=" formula "in" formula                                            -> fix
                         | "repeat" label_decl "=" formula "in" formula "for" NUMBER                            -> repeat
                         | expression

                ?expression:  choice

                ?choice: choice "|" parallel                                                                   -> choice
                        | parallel

                ?parallel: parallel ("||" sequential)+                                                         -> parallel_composition
                          | sequential

                ?sequential: sequential ";" unary                                                             -> sequential_composition
                            | unary

                ?unary:  unary "*"                                                                             -> star
                       | "rep" unary "for" NUMBER                                                              -> rep
                       | primary

                ?primary: label                                                                                -> atom_op
                          | "i"                                                                                -> id
                          | "<" label? "|" label                                                               -> lhd
                          | "|" label? ">" label                                                               -> rhd
                          | label "(" (formula ",")* formula ")"                                               -> fun_call
                          | "(" formula ")"

                label: /[a-zA-Z_0-9\^\-\!\%\&\+\~\/\@]+/
                            |  FUNC_GEN

                FUNC_GEN: /[a-zA-Z_0-9\^\-\!\%\+\&\~\/\@]+/ "[" /[^\]\[]+/ "]"

                label_decl: /[a-zA-Z_0-9\^\-\!\+\%\&\~\/\@]+/

                COMMENT: "#" /[^\n]/*

                ?start: formula

                %ignore COMMENT
                %import common.WS
                %import common.NUMBER
                %import common.UCASE_LETTER
                %ignore WS
                """

mg_reserved = {'||', 'fix', 'let', 'if', 'then', 'else', 'def', 'in', 'repeat', 'rep', 'for', ',', '|', '<', '>', '(', ')', ';', '[', ']', '#', '{', '}', '*'}

mg_parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')
"""
Parser instance on which to call ``parse``.

Examples:
    >>> mg_parser.parse('(a || b) ; c')
    Tree('sequential_composition', [Tree('parallel_composition', ...)])
"""


class MGReconstructor(Reconstructor):
    """Reconstructor for the mG language.

    The reconstructor transforms a parse tree into the corresponding string, reversing the operation of parsing. This implementation is a slight
    modification of Lark's ``Reconstructor`` class to allow the addition of white spaces between the appropriate tokens.
    """

    def reconstruct(self, tree: ParseTree, postproc: Callable[[Iterable[str]], Iterable[str]] | None = None, insert_spaces: bool = True) -> str:
        """Reconstructs a string from a parse tree.

        Args:
            tree: The tree to reconstruct.
            postproc: The post-processing function to apply to each word of the reconstructed string.
            insert_spaces: If true, add spaces between any two words of the reconstructed string.

        Examples:
            >>> from lark import Tree, Token
            >>> mg_reconstructor.reconstruct(Tree('rhd', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')]), Tree(Token('RULE', 'label'),
            ...                              [Token('__ANON_1', 'b')])]))
            '|a>b'

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
                if (prev_item not in {'<', '>', '(', '{', '|'} and item not in {'<', '>', ')', '}', ',', '|'}
                        and not (item == '(' and prev_item not in mg_reserved)
                        or item == ';' or prev_item == ';'):
                    y.append(' ')
            y.append(item)
            prev_item = item
        return ''.join(y)


mg_reconstructor = MGReconstructor(mg_parser)
"""
Reconstructor (unparser) instance on which to call ``reconstruct``.

Examples:
    >>> from lark import Tree, Token
    >>> mg_reconstructor.reconstruct(Tree('rhd', [Tree(Token('RULE', 'label'), [Token('__ANON_1', 'a')]), Tree(Token('RULE', 'label'),
    ...                              [Token('__ANON_1', 'b')])]))
    '|a>b'
"""
