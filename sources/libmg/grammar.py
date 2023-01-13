"""
The notation (gnn_formula)? ";" gnn_formula forces the parser to evaluate the sequential op left to right
"""

mg_grammar = r"""
                ?gnn_formula: label                                         -> atom_op
                         | "<" label? "|" label                     -> lhd
                         | "|" label? ">" label                     -> rhd
                         | "mu" label_decl ":" type_decl "=" VALUE "." gnn_formula        -> mu_formula
                         | "nu" label_decl ":" type_decl "=" VALUE "." gnn_formula        -> nu_formula
                         | label "(" (p_formula ",")* p_formula ")"         -> fun_call
                         | "(" start ")"

                ?p_formula: gnn_formula
                         | gnn_formula ";" p_formula                                -> composition
                         | gnn_formula ( "||" gnn_formula )+                        -> parallel

                ?start: p_formula
                        | "def" label_decl "(" (label_decl ":" type_decl ",")* label_decl ":" type_decl "){" p_formula "}" start -> fun_def
                        | "let" label_decl "=" p_formula start -> var_def

                label: /[a-zA-Z_0-9\+\*\^\-\!\#\%\&\=\~\/]+/
                            |  FUNC_GEN

                FUNC_GEN: /[a-zA-Z_0-9\+\*\^\-\!\#\%\&\=\~\/]+/ "[" /[^\]\[]+/ "]"

                label_decl: /[a-zA-Z_0-9\+\*\^\-\!\#\%\&\=\~\/]+/

                type_decl: TYPE "[" NUMBER "]"

                TYPE: "bool" | "int" | "float" | "uint8" | "uint16" | "uint32" | "uint64" | "int8" | "int16" | "int32"
                      | "int64" | "float16" | "float32" | "float64" | "half" | "double"

                VALUE: "true" | "false" | NUMBER

                %import common.WS
                %import common.NUMBER
                %import common.UCASE_LETTER
                %ignore WS
                """
