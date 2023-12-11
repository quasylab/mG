# Explanation

In this part of the documentation we focus on the theoretical background
of $\mu\mathcal{G}$.

## [Syntax and Semantics of μG](semantics.md)
The $\mu\mathcal{G}$ language has a precise syntax and semantics, which we presented in a prior publication[^1].
Here you will find the abstract syntax, the typing, and the denotational semantics of the language.

[^1]: Matteo Belenchia, Flavio Corradini, Michela Quadrini, and Michele Loreti. 2023. Implementing a CTL Model Checker with μG, a Language for Programming
Graph Neural Networks. In Formal Techniques for Distributed Objects, Components, and Systems: 43rd IFIP WG 6.1 International Conference, FORTE 2023,
Held as Part of the 18th International Federated Conference on Distributed Computing Techniques, DisCoTec 2023, Lisbon, Portugal, June 19–23, 2023,
Proceedings. Springer-Verlag, Berlin, Heidelberg, 37–54. <https://doi.org/10.1007/978-3-031-35355-0_4>

## [Normalization](reduction.md)
A $\mu\mathcal{G}$ generated according to its syntax may be amenable to simplification,
which makes it easier to analyze by the compiler. Here you can find how the normalization process works and how
$\mu\mathcal{G}$ expressions are simplified when the compiler invokes the normalizer.

## [Explanation of μG models](explanation.md)
We show here what kind of explanations <span style="font-variant:small-caps;">libmg</span> can automatically generate from a
$\mu\mathcal{G}$ model.