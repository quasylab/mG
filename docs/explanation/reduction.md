# Normalization

A $\mu\mathcal{G}$ expression as generated from its abstract syntax may be simplified (i.e. normalized) to a reduced, simpler form.
This simplification is useful in practice because it may result in less computations to be performed for the same output,
or it may make it easier to compile. All $\mu\mathcal{G}$ expressions in <span style="font-variant:small-caps;">libmg</span>
are automatically normalized before compilation.

For the time being, only two types of normalizations are performed by <span style="font-variant:small-caps;">libmg</span>, 
the first, is the simplification of fixpoint expressions where the variable does not occur in the expression body, the second
is the rewriting of sequential composition expressions where the fixpoint variable occurs on the right hand side.

## Simplification of unoccurring fixpoint variables

An expression of the form `fix X = N1 in N2` where `X` does not occur in `N2` is equivalent to just `N2`.
For example, the expression `fix X = 0 in (a || b);+`. Clearly, the value of `X` is not necessary to compute the sum `a + b` 
and we can simplify it to `(a || b);+`.

!!! warning
    
    This simplification is also applied in the case of the `repeat` macro.

## Simplification of sequential composition of fixpoint variables

Expressions of the form `fix X = N in N1(X) ; N2(X)` or `fix X = N in N1 ; N2(X)` where the fixpoint variable `X` 
occurs on the right-hand side (rhs) of a sequential composition expressions can be simplified to an expression of the form
`fix X = N in N3(X) ; N4` where all the fixpoint variables now only occur on the left-hand side (lhs).

Given the sequential composition of a lhs and rhs, we discard the lhs and we visit the parse tree of the rhs:

* For any fixpoint variable encountered, it is returned as-is.
* For any atomic operation $a$ (function application, or any of the images) encountered, we return the sequential composition of the lhs with $a$.
* For any sequential composition, the lhs is evaluated according to this algorithm and then this procedure is applied recursively to the rhs given the new lhs.
* Any other expression is returned as-is.

For example, consider the expression `fix X = N in X ; X`. We analyze `X ; X` by discarding the lhs (`X`) and visit the rhs (`X`).
Since it's the fixpoint variable, it is returned as is and we are finished with the result being just `X`, and the fixpoint expression has become
`fix X = N in X`.

Now, for a more involved example, consider `fix X = N in (a ; (X || b) ; c)`. We analyze `a ; (X || b) ; c` by discarding the lhs (`a`) and visit the
rhs (`(X || b) ; c`). The variable `X` is left as-is, the function application `b` becomes `a ; b`. The parallel composition is left as-is and we have
`(X || (a;b))`. Since this is the lhs of another sequential composition, this become the new lhs for the following and we repeat this procedure with the current 
rhs `c`. The rhs `c` is a function application  which becomes `(X || (a;b)) ; c` according to the new lhs, and this is the output of the algorithm. The fixpoint
expression thus becomes `fix X = N in ((X || (a;b)) ; c)`.

!!! warning
    
    A fixpoint variable is always assumed to be a constant GNN that returns the node labeling function obtained from the application of the 
    fixpoint variable initialization GNN (N in the expressions above) to the input node labeling function of the fixpoint expression.