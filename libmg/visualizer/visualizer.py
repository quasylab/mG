"""Defines a visualizer for graphs.

This module defines the functions to view graphs and mG model outputs on a web browser in an interactive way.

The module contains the following functions:
- ``fetch_layer(model, layer_name=None, layer_idx=None)``
- ``show(node_values, adj, edge_values, labels, hierarchy, title_generator, filename, open_browser)``
- ``print_layer(model, inputs, labels=None, layer_name=None, layer_idx=None, open_browser=True)``
- ``print_graph(graph, node_names_func='id', hierarchical=False, show_labels=False, open_browser=True)``
"""
from typing import Callable, Iterator

import numpy as np
import tensorflow as tf
from lark import Tree
from pyvis.network import Network
from pyvis.options import Layout, Options
from spektral.data import Graph

from libmg.compiler.compiler import MGModel
from libmg.compiler.grammar import mg_parser, mg_reconstructor
from libmg.compiler.layers import unpack_inputs


def fetch_layer(model: MGModel, layer_name: str | Tree | None = None, layer_idx: int | None = None) -> tf.keras.layers.Layer:
    """Finds a layer in a mG model, identified either by its name or index.

    If both a layer name and a layer index are provided, index takes precedence.

    Args:
        model: The mG model where the layer is to be found.
        layer_name: The name of the layer to find.
        layer_idx: The index of the layer to find.

    Returns:
        The layer corresponding to the given name or index.

    Raises:
        ValueError: Neither a name nor an index have been given.
        KeyError: No layer of the given name is present in the model.
    """
    if layer_idx is not None:
        return model.get_layer(index=layer_idx)
    elif layer_name is not None:
        tree = layer_name if isinstance(layer_name, Tree) else mg_parser.parse(layer_name)
        layer_hash = hash(tree)
        if layer_hash in model.mg_layers:
            return model.mg_layers[layer_hash]
        else:
            raise KeyError("No layer found with name: " + layer_name if isinstance(layer_name, str) else mg_reconstructor.reconstruct(layer_name))
    else:
        raise ValueError("Layer name or index must be provided!")


def show(node_values: np.ndarray, adj: np.ndarray | Iterator[tuple[int, int]], edge_values: np.ndarray | None, labels: np.ndarray | None,
         hierarchy: list[int] | None, id_generator: Callable[[int], int | str], filename: str, open_browser: bool) -> None:
    """
    Builds a PyVis network using the node features, labels, adjacency matrix and edge features. The result is an .html
    page named graph_``filename``.html.

    Args:
        node_values: The node features with shape ``(n_nodes, n_node_features)``.
        adj: The adjacency list with shape ``(n_edges, 2)``.
        edge_values: The edge features with shape ``(n_edges, n_edge_features)``, if present.
        labels: The true labels of the nodes with shape ``(n_nodes, n_labels)``, if present.
        hierarchy: Used for visualizing the graph in hierarchical mode. For each node, the corresponding number in this list is the position in the hierarchy.
            If ``None``, the graph is not visualized in hierarchical mode.
        id_generator: Used for determining the ID of a node. Usually this is the identity function so that the first node has ID = 0, the second has ID = 1, and
            so on. In other situations, it might be necessary to have different mappings for integer IDs or to use string IDs.
        filename: The name of the .html file to save in the working directory. The string ``graph_`` will be prepended to it.
        open_browser: If true, opens the default web browser and loads up the generated .html page.

    Returns:
        Nothing.
    """
    nodes = [id_generator(i) for i in range(node_values.shape[0])]
    edges = [(int(e[0]), int(e[1])) for e in adj]
    node_labels = [' | '.join([str(label) for label in label_list]) for label_list in node_values.tolist()]
    if edge_values is not None:
        edge_labels = [' | '.join([str(label) for label in label_list]) for label_list in edge_values.tolist()]
    else:
        edge_labels = None
    if labels is not None:
        true_labels = [' | '.join([str(label) for label in label_list]) for label_list in labels.tolist()]
    else:
        true_labels = None
    titles = [str(i) for i in nodes]

    # Build the pyvis network
    net = Network(directed=True, neighborhood_highlight=True, select_menu=True, filter_menu=True)
    layout = Layout(improvedLayout=True)

    if true_labels is not None:
        node_labels = ['[' + feat + '] → [' + target + ']' for feat, target in zip(node_labels, true_labels)]
    else:
        node_labels = ['[' + feat + ']' for feat in node_labels]

    if hierarchy is not None:
        net.options = Options(layout)
        for i, v in enumerate(nodes):
            net.add_node(v, title=titles[i], label=node_labels[i], shape='circle', level=hierarchy[i])
    else:
        layout.hierarchical = layout.Hierarchical(enabled=False)
        net.options = Options(layout)
        del net.options['layout'].hierarchical
        net.add_nodes(nodes, title=titles, label=node_labels, shape=['circle'] * len(nodes))

    if edge_labels is None:
        net.add_edges(edges)
    else:
        for i, edge in enumerate(edges):
            net.add_edge(*edge, label=edge_labels[i])

    net.force_atlas_2based()
    net.show_buttons()
    if open_browser:
        net.show('graph_' + filename + '.html', notebook=False)
    else:
        net.save_graph('graph_' + filename + '.html')


def print_layer(model: MGModel, inputs: list[tf.Tensor], labels: tf.Tensor | None = None, layer_name: str | Tree | None = None, layer_idx: int | None = None,
                filename: str | None = None, open_browser: bool = True) -> None:
    """Visualizes the outputs of a model's layer using PyVis.

    Layer must be identified either by name or index. If both are given, index takes precedence.

    Args:
        model: The mG model where the layer to visualize is to be found.
        inputs: The inputs of the model that are used to generate the output to visualize.
        labels: If provided, also show the node labels alongside the node features generated by the visualized layer.
        layer_name: The name of the layer to find.
        layer_idx: The index of the layer to find.
        filename: The name of the .html file to save in the working directory. The string ``graph_`` will be prepended to it and the provided index or layer
            name will be appended to it.
        open_browser: If true, opens the default web browser and loads up the generated .html page.

    Returns:
        Nothing.

    Raises:
        ValueError: Neither a name nor an index have been given.
        KeyError: No layer of the given name is present in the model.
    """
    layer = fetch_layer(model, layer_name, layer_idx)
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    idx_or_name = layer_idx if layer_idx is not None else layer.name
    labels = labels.numpy() if labels is not None else None
    _, a, e, _ = unpack_inputs(inputs)
    show(debug_model(inputs).numpy(), a.indices.numpy(), e.numpy() if e is not None else None, labels, None, lambda x: x,
         filename + '_' + str(idx_or_name) if filename is not None else str(idx_or_name), open_browser)


def print_graph(graph: Graph, id_generator: Callable[[int], int | str] = lambda x: x, hierarchical: bool = False, show_labels: bool = False,
                filename: str | None = None, open_browser: bool = True) -> None:
    """Visualizes a graph using PyVis.

    Args:
        graph: The graph to visualize.
        id_generator: Used for determining the ID of a node. Usually this is the identity function so that the first node has ID = 0, the second has ID = 1, and
            so on. In other situations, it might be necessary to have different mappings for integer IDs or to use string IDs.
        hierarchical: If true, visualize the graph in hierarchical mode. The graph must have a ``hierarchy`` attribute containing the hierarchy.
        show_labels: If true, show the node labels alongside the node features.
        filename: The name of the .html file to save in the working directory. The string ``graph_`` will be prepended to it.
        open_browser: If true, opens the default web browser and loads up the generated .html page.

    Returns:
        Nothing.
    """
    show(graph.x, zip(graph.a.row, graph.a.col), graph.e, graph.y if show_labels else None, graph.hierarchy if hierarchical else None, id_generator,
         filename if filename is not None else str(graph), open_browser)
