# reserved words: || fix let if then else def in repeat for
# reserved symbols: , | < > = ( ) ; [ ] # { }
from lark import Lark

mg_grammar = r"""
                ?gnn_formula: label                                                                               -> atom_op
                         | "<" label? "|" label                                                                   -> lhd
                         | "|" label? ">" label                                                                   -> rhd
                         | label "(" (p_formula ",")* p_formula ")"                                               -> fun_call
                         | "let" (label_decl "=" p_formula ",")* label_decl "=" p_formula "in" p_formula          -> local_var_expr
                         | "def" label_decl "(" (label_decl ",")* label_decl "){" p_formula "} in " p_formula     -> fun_def
                         | "if" p_formula "then" p_formula "else" p_formula                                       -> ite
                         | "fix" label_decl "=" p_formula "in" p_formula                                          -> fix
                         | "repeat" label_decl "=" p_formula "in" p_formula "for" NUMBER                          -> repeat
                         | "(" p_formula ")"

                ?p_formula: gnn_formula
                         | gnn_formula ";" p_formula                                                              -> composition
                         | gnn_formula ( "||" gnn_formula )+                                                      -> parallel

                ?start: p_formula

                label: /[a-zA-Z_0-9\+\*\^\-\!\%\&\=\~\/\@]+/
                            |  FUNC_GEN

                FUNC_GEN: /[a-zA-Z_0-9\+\*\^\-\!\%\&\=\~\/\@]+/ "[" /[^\]\[]+/ "]"

                label_decl: /[a-zA-Z_0-9\+\*\^\-\!\%\&\=\~\/\@]+/
                
                COMMENT: "#" /[^\n]/*

                %ignore COMMENT
                %import common.WS
                %import common.NUMBER
                %import common.UCASE_LETTER
                %ignore WS
                """

mg_parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')
