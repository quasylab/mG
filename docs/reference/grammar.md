# Grammar Reference
The grammar that generates $\mu\mathcal{G}$ expressions is the following:

```
start ::= c_formula

c_formula ::=  gnn_formula
                | gnn_formula ";" c_formula 
                | gnn_formula ( "||" gnn_formula )  
                
gnn_formula ::= label                                                                              
                 | "<" label? "|" label                                                                   
                 | "|" label? ">" label                                                                   
                 | label "(" (c_formula ",")* c_formula ")"                                               
                 | "let" (label_decl "=" c_formula ",")* label_decl "=" c_formula "in" c_formula          
                 | "def" label_decl "(" (label_decl ",")* label_decl ")" "{" c_formula "}" "in" c_formula
                 | "if" c_formula "then" c_formula "else" c_formula                                      
                 | "fix" label_decl "=" c_formula "in" c_formula                                          
                 | "repeat" label_decl "=" c_formula "in" c_formula "for" NUMBER                   
                 | "(" c_formula ")"

label ::= /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/ |  FUNC_GEN

FUNC_GEN ::= /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/ "[" /[^\]\[]+/ "]"

label_decl ::= /[a-zA-Z_0-9\+\*\^\-\!\%\&\~\/\@]+/

COMMENT ::= "#" /[^\n]/*

%ignore COMMENT
%import common.WS
%import common.NUMBER
%import common.UCASE_LETTER
%ignore WS
```

The reserved symbols and words of $\mu\mathcal{G}$, which therefore cannot be used as function labels or variable names, are:

```
|| fix let if then else def in repeat for | < > , ( ) ; [ ] # { }
```
!!! warning 
    
    Variables should not be named with two initial underscores __, e.g. don't name variables such as __X.
    Such variables names are used internally for operators.