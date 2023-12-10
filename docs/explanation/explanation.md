# Explanation of μG models

<span style="font-variant:small-caps;">libmg</span> allows for a rudimentary kind of "explanation" for a model's output. Given a model $M$, the input graph
$\mathtt{G}$, and a query node $q$ it generates the sub-graph of $\mathtt{G}$ that contains the node and edges that $M$ used to determine the output label of $q$.
It then shows this sub-graph using the visualization functions in a hierarchical perspective with the query node on top followed below by the other nodes
ordered by hop count.

The algorithm that computes the sub-graph is a $\mu\mathcal{G}$ program itself, and is produced by transforming the expression that generated the model to be
explained. In the case that the model to be explained contained $\psi$ functions that used non-local information or the `if-then-else` construct,
every node can potentially depend on every node in the graph and the procedure just yields the entire graph, visualized in a hierarchical view. Otherwise,
the transformation function is defined as $\mathtt{explain}(\mathcal{N}, q) = \psi_q ; \mathcal{E}(\mathcal{N})$, where $\mathcal{E}$ is defined inductively as:

$$
\begin{align*}
\mathcal{E}(\psi) &= \psi_{id} \\
\mathcal{E}(\lhd_{\sigma}^{\varphi}) &= \rhd_{\sigma_{min}}^{\varphi_{pr_1}} \\
\mathcal{E}(\rhd_{\sigma}^{\varphi}) &= \lhd_{\sigma_{min}}^{\varphi_{pr_1}} \\
\mathcal{E}(\mathcal{N}_1 || \mathcal{N}_2) &= (\mathcal{E}(\mathcal{N}_1) || \mathcal{E}(\mathcal{N}_2)) ; \psi_{min} \\
\mathcal{E}(\texttt{fix } X = \mathcal{N}_1 \texttt{ in } \mathcal{N}_2) &= \texttt{repeat } X = \mathcal{E}(\mathcal{N}_1) \texttt{ in } \mathcal{E}(\mathcal{N}_2) \texttt{ for } k \\
\mathcal{E}(\mathcal{N}(\mathcal{N}_1, \mathcal{N}_2, \ldots)) &= \mathcal{N}(\mathcal{E}(\mathcal{N}_1), \mathcal{E}(\mathcal{N}_2), \ldots)
\end{align*}
$$

where $\psi_q$ denotes the function that sets the label of node $q$ to 1 and that of all other nodes to 0, $\psi_{id}$ denotes the identity function,
$\psi_{min}$ denotes the minimum function, $\varphi_{pr_1}$ is the function that returns the first argument, and $\sigma_{min}$ denotes a function that
aggregates the messages by taking the minimum between the label of the receiver node and the minimum of the inbound messages plus one. The value $k$ is the
number of iterations performed by the corresponding fixpoint layer. The algorithm works by propagating from the query node and building up the hierarchical
view of the relevant sub-graph.