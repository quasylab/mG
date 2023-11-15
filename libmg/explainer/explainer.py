"""Defines an explainer for mG models.

This module defines the means to generate the sub-graph of all nodes that influenced the final label of some query node.

The module contains the following functions:
- ``explanation_nodes(explanation)``
- ``make_graph(explanation, hierarchy, old_graph, labels)``

The module contains the following classes:
- ``MGExplainer``
"""
from __future__ import annotations
import typing
from typing import Callable

from lark import Tree
from lark.exceptions import VisitError
from lark.visitors import Interpreter
import tensorflow as tf
from scipy.sparse import coo_matrix
from spektral.data import Graph
from multiprocessing.pool import ThreadPool

from libmg.compiler.functions import PsiLocal, Phi, Sigma
from libmg.compiler.compiler import MGCompiler, MGModel
from libmg.compiler.grammar import mg_parser
from libmg.compiler.layers import unpack_inputs
from libmg.visualizer.visualizer import print_graph


def explanation_nodes(explanation: tf.Tensor[bool]) -> list[int]:
    """Returns the nodes that are part of the explanation.

    Args:
        explanation: The boolean tensor of shape ``(n_nodes,)`` that marks with ``True`` the nodes that are part of the explanation
            and ``False`` those that do not.
    """
    return typing.cast(list[int], tf.squeeze(tf.where(explanation), axis=-1).numpy().tolist())


def make_graph(explanation: tf.Tensor[bool], hierarchy: tf.Tensor[float], old_graph: list[tf.Tensor], labels: tf.Tensor) -> Graph:
    """Returns the explanation sub-graph.

    Args:
        explanation: The boolean tensor of shape ``(n_nodes,)`` that marks with ``True`` the nodes that are part of the explanation
            and ``False`` those that do not.
        hierarchy: The hierarchy tensor of shape ``(n_nodes,)`` that assigns a numerical value to every node in the explanation. The query node is assigned 0,
            and each >0 number indicates the number of hops of distance to the query node.
        old_graph: The input graph of the explained model.
        labels: The true labels of the nodes in the graph.
    """
    node_feats, adj, edge_feats, _ = unpack_inputs(old_graph)

    new_node_feats = tf.boolean_mask(node_feats, explanation).numpy()
    new_labels = tf.boolean_mask(labels, explanation).numpy() if labels is not None else None

    nodes = set(explanation_nodes(explanation))
    edges = adj.indices.numpy().tolist()
    with ThreadPool() as pool:
        edge_mask = pool.map(lambda edge: edge[0] in nodes and edge[1] in nodes, edges)
    new_adj = tf.sparse.retain(adj, edge_mask)
    new_adj = coo_matrix((new_adj.values.numpy(), (new_adj.indices.numpy()[:, 0], new_adj.indices.numpy()[:, 1])),
                         shape=adj.shape)

    new_edge_feats = tf.boolean_mask(edge_feats, edge_mask).numpy() if edge_feats is not None else None

    hierarchy = tf.cast(hierarchy[explanation], dtype=tf.int64).numpy().tolist()

    new_graph = Graph(x=new_node_feats, a=new_adj, e=new_edge_feats, y=new_labels, hierarchy=hierarchy)

    return new_graph


class MGExplainer(Interpreter):
    # TODO: explain the logic here
    """Explains a mG model output.

    Generates the sub-graph of nodes that are responsible for the label of a given node.

    Attributes:
        model: The model to explain.
        query_node: The node of the input graph for which the sub-graph of relevant nodes will be generated.
        compiler: The compiler for the explainer.
    """
    INF = 1e38
    localize_node = PsiLocal.make_parametrized('localize_node', lambda y: lambda x: tf.one_hot(indices=[int(y)],
                                                                                               depth=tf.shape(x)[0],
                                                                                               axis=0,
                                                                                               on_value=0,
                                                                                               off_value=MGExplainer.INF, dtype=tf.float32))
    # localize_node = lambda y: PsiLocal(lambda x: tf.one_hot(indices=[int(y)], depth=tf.shape(x)[0],
    # axis=0, on_value=0, off_value=ExplainerMG.INF, dtype=tf.float32))
    id = PsiLocal(lambda x: x)
    proj3 = Phi.make('proj', lambda i, e, j: i)
    or_agg = Sigma(lambda m, i, n, x: tf.minimum(tf.math.unsorted_segment_min(m, i, n) + 1, x))
    or_fun = PsiLocal(lambda x: tf.math.reduce_min(x, axis=1, keepdims=True))
    all_nodes_expr = 'fix X = id in (((X;|p3>or) || (X;<p3|or));or)'

    def __init__(self, model: MGModel):
        """Initializes the instance with the model to explain.

        Args:
            model: The model to explain.
        """
        super().__init__()
        self.model = model
        self.query_node: int
        self.compiler = MGCompiler(psi_functions={'node': MGExplainer.localize_node, 'id': MGExplainer.id, 'or': MGExplainer.or_fun},
                                   sigma_functions={'or': MGExplainer.or_agg},
                                   phi_functions={'p3': MGExplainer.proj3},
                                   config=model.config)

    @staticmethod
    def get_original_ids_func(explanation: tf.Tensor[bool]) -> Callable[[int], int]:
        """Returns a function that given an integer i returns the i-th node ID of the nodes in the explanation.

        The returned function is used so that the node IDs of the explanation sub-graph are the same as the original graph.

        Args:
            explanation: The boolean tensor of shape ``(n_nodes,)`` that marks with ``True`` the nodes that are part of the explanation
                and ``False`` those that do not.
        """
        sorted_node_ids = sorted(explanation_nodes(explanation))
        return lambda x: sorted_node_ids[x]

    def explain(self, query_node: int, inputs: list[tf.Tensor], labels: tf.Tensor, filename: str, open_browser: bool = True) -> Graph:
        """Explain the label of a query node by generating the sub-graph of nodes that affected its value.

        The explanation is saved in a PyVis .html file in the working directory.

        Args:
            query_node: The node for which to generate the explanation.
            inputs: The inputs for the model to explain. This is the graph to which the query node belongs.
            labels: The true labels of the graph.
            filename: The name of the .html file to save in the working directory. The string ``graph_`` will be prepended to it.
            open_browser: If true, opens the default web browser and loads up the generated .html page.

        Returns:
            The generated sub-graph.
        """
        # Build the model
        self.query_node = query_node
        try:
            right_branch = self.visit(mg_parser.parse(self.model.expr))
        except VisitError:
            right_branch = mg_parser.parse(MGExplainer.all_nodes_expr)
        left_branch = mg_parser.parse('node[' + str(self.query_node) + ']')
        explainer_expr_tree = mg_parser.parse('left ; right')
        explainer_expr_tree.children = [left_branch, right_branch]
        explainer_model = self.compiler.compile(explainer_expr_tree)

        # Run the model
        hierarchy = tf.squeeze(explainer_model.call(inputs))
        explanation = tf.math.less(hierarchy, MGExplainer.INF)
        graph = make_graph(explanation, hierarchy, inputs, labels)
        print_graph(graph, id_generator=self.get_original_ids_func(explanation), hierarchical=True, show_labels=True, filename=filename,
                    open_browser=open_browser)
        return graph

    def atom_op(self, tree: Tree) -> Tree:
        """Explains a psi function or a variable.

        A local psi function is explained as an identity function (because only the query_node is affected by it). Variables are left as is. Non-local psi
        functions makes the label of the query node dependent on the entire graph and therefore the explanation is aborted.

        Args:
            tree: The expression tree.

        Returns:
            The expression tree that explains this psi function or variable.

        Raises:
            VisitError: The expression is a non-local psi function.
        """
        name = str(tree.children[0].children[0])
        f = self.model.psi_functions.get(name)
        if f is None:
            new_op = tree
        elif isinstance(f, PsiLocal):
            new_op = mg_parser.parse('id')
        else:  # not local neither a variable
            raise VisitError('atom_op', tree, 'Nonlocal psi function')
        return new_op

    def lhd(self, _: Tree) -> Tree:
        """Explains a pre-image expression.

        The pre-image is explained by the post-image that generates the messages as node labels of the successors and aggregates them by taking the minimum.

        Args:
            _: The expression tree.

        Returns:
            The expression tree that explains this pre-image expression.
        """
        new_op = mg_parser.parse('|p3>or')
        return new_op

    def rhd(self, _: Tree) -> Tree:
        """Explains a post-image expression.

        The post-image is explained by the pre-image that generates the messages as node labels of the predecessors and aggregates them by taking the minimum.

        Args:
            _: The expression tree.

        Returns:
            The expression tree that explains this post-image expression.
        """
        new_op = mg_parser.parse('<p3|or')
        return new_op

    def parallel_composition(self, tree: Tree) -> Tree:
        """Explains a parallel composition expression.

        The parallel composition is explained by taking the minimum of its sub-expressions.

        Args:
            tree: The expression tree.

        Returns:
            The expression tree that explains this parallel composition expression.
        """
        children = self.visit_children(tree)
        new_op = mg_parser.parse('SUBST;or')
        tree_copy = tree.copy()
        tree_copy.children = children
        new_op.children[0] = tree_copy
        return new_op

    def ite(self, tree: Tree) -> Tree:
        """Explains an if-then-else expression.

        The if-then-else test is takes into consideration all the nodes in the graph, therefore the explanation is aborted.

        Args:
            tree: The expression tree.

        Returns:
            The expression tree that explains this if-then-else expression.

        Raises:
            VisitError: Always.
        """
        raise VisitError('ite', tree, 'If-Then-Else expression')

    def __default__(self, tree: Tree) -> Tree:  # local var expr, fun def, fun call, fix, repeat, composition
        """Explains all the other expressions.

        Args:
            tree: The expression tree.

        Returns:
            The expression tree that explains this expression.
        """
        tree.children = self.visit_children(tree)
        return tree
