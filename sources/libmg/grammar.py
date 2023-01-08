"""
The notation (gnn_formula)? ";" gnn_formula forces the parser to evaluate the sequential op left to right
"""


mg_grammar = r"""
                ?gnn_formula: function_name                                         -> fun_app
                         | "<" function_name? "|" function_name                     -> lhd
                         | "|" function_name? ">" function_name                     -> rhd
                         | UCASE_LETTER                                             -> variable
                         | "mu" variable_decl "," type_decl "."  gnn_formula        -> mu_formula
                         | "nu" variable_decl "," type_decl "."  gnn_formula        -> nu_formula
                         | function_name "(" p_formula ")"                          -> fun_call
                         | "(" start ")"

                ?p_formula: gnn_formula
                         | gnn_formula ";" p_formula                                -> composition
                         | gnn_formula ( "||" gnn_formula )+                        -> parallel
                         
                ?start: p_formula
                        | "def" function_name "(" variable_decl ":" type_decl "){" p_formula "}" start -> fun_def

                function_name: /[a-zA-Z_0-9\+\*\^\-\!\#\%\&\=\~\:\/]+/
                            |  FUNC_GEN

                FUNC_GEN: /[a-zA-Z_0-9\+\*\^\-\!\#\%\&\=\~\:\/]+/ "[" /[^\]\[]+/ "]"

                variable_decl: UCASE_LETTER
                

                type_decl: /[a-zA-Z_0-9][a-z_0-9]*/

                %import common.WS
                %import common.UCASE_LETTER
                %ignore WS
                """
