import tensorflow as tf
from pyvis.network import Network
from pyvis.options import Layout, Options
from spektral.data import Graph
from tensorflow.python.keras import backend as K

from .grammar import mg_parser


def fetch_layer(model, layer_name=None, layer_idx=None):
    """
    Finds a layer, identified either by name or index, in the model. If both are
    given, index takes precedence.

    :param model: A mG ``Model``.
    :param layer_name: Name of the layer to find. At least this or an index must be provided.
    :param layer_idx: Index of the layer to find. At least this or a name must be provided.
    :return: The ``Layer`` with the given name or at the given index.
    :raises ValueError: If no layer with the given name is found or if neither an index nor a name is provided.
    """
    if layer_idx is not None:
        return model.get_layer(index=layer_idx).output
    elif layer_name is not None:
        layer_hash = hash(mg_parser.parse(layer_name))
        if layer_hash in model.mg_layers:
            return model.mg_layers[layer_hash].output
        else:
            raise ValueError("No layer found with name: " + layer_name)
    else:
        raise ValueError("Layer name or index must be provided!")


def find_adj(inputs):
    """
    Finds the adjacency matrix in the list ``inputs``.

    :param inputs: A list of ``Tensor`` objects.
    :return: The adjacency matrix ``SparseTensor``.
    """
    return next((x for x in inputs if K.is_sparse(x)))


def find_edges(inputs):
    """
    Finds the edge features matrix in the list ``inputs``, if present.

    :param inputs: A list of ``Tensor`` objects.
    :return: The edge features ``Tensor`` if found or ``None``.
    """
    if len(inputs) == 3 and K.ndim(inputs[-1]) == 2:
        return inputs[-1]
    elif len(inputs) == 4:
        return inputs[-2]
    else:
        return None


def show(node_values, adj, edge_values, labels, filename, open_browser):
    """
    Builds a PyVis network using the node features, labels, adjacency matrix and edge features. The result is an .html
    page named *graph_filename.html*.

    :param node_values: A ``ndarray`` with shape (n_nodes, n_node_features) containing the node features
    :param adj: A `ndarray`` adjacency list
    :param edge_values: An optional ``ndarray`` with shape (n_edges, n_edge_features) containing the edge labels
    :param labels: An optional ``ndarray`` with shape (n_nodes, n_labels) containing the node labels
    :param filename: Name of the .html file to generate
    :param open_browser: If true, opens the default web browser and opens the generated .html page.
    :return: Nothing
    """
    nodes = list(range(node_values.shape[0]))
    # edges = [(int(i), int(j)) for i, j in zip(adj.row, adj.col)]
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

    ### Build the pyvis network ###
    net = Network(directed=True, neighborhood_highlight=True, select_menu=True, filter_menu=True)
    layout = Layout(improvedLayout=True)
    layout.hierarchical = layout.Hierarchical(enabled=False)
    net.options = Options(layout)
    del net.options['layout'].hierarchical
    if true_labels is not None:
        node_labels = ['[' + feat + '] → [' + target + ']' for feat, target in zip(node_labels, true_labels)]
    else:
        node_labels = ['[' + feat + ']' for feat in node_labels]
    net.add_nodes(nodes, title=titles, label=node_labels, shape=['circle'] * len(nodes))
    if edge_labels is None:
        net.add_edges(edges)
    else:
        for i, edge in enumerate(edges):
            net.add_edge(*edge, label=edge_labels[i])
    # net.barnes_hut(gravity=-2000, spring_length=250, spring_strength=0.04)
    # net.toggle_physics(False)
    # net.toggle_stabilization(True)
    net.force_atlas_2based()
    net.show_buttons()
    if open_browser:
        net.show('graph_' + filename + '.html', notebook=False)
    else:
        net.save_graph('graph_' + filename + '.html')


def print_layer(model, inputs, labels=None, layer_name=None, layer_idx=None, open_browser=True):
    """
    Visualizes the outputs of a model's layer using PyVis. Layer must be identified either by name or index. If both are
    given, index takes precedence.

    :param model: A mG ``Model`` that contains the layer to visualize.
    :param inputs: Inputs to feed the model, as obtained by the data ``Loader`` for the model.
    :param labels: If provided, also show the labels alongside the node features. Only works for node level labels.
    :param layer_name: Name of the layer whose output is to be visualized. At least this or an index must be provided.
    :param layer_idx: Index of the layer whose output is to be visualized. At least this or a name must be provided.
    :param open_browser: If true, opens the default web browser and opens the generated .html page.
    :return: Nothing.
    :raises ValueError: If neither an index nor a name is provided.
    """
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx))
    idx_or_name = layer_idx if layer_idx is not None else layer_name
    if idx_or_name is None:
        raise ValueError("Layer name or index must be provided!")
    labels = labels.numpy() if labels is not None else None
    e = find_edges(inputs).numpy() if find_edges(inputs) is not None else None
    show(debug_model(inputs).numpy(), find_adj(inputs).indices.numpy(), e, labels, str(idx_or_name), open_browser)


def print_graph(graph, show_labels=False, open_browser=True):
    """
    Visualizes a graph using PyVis.

    :param graph: A ``Graph``.
    :param show_labels: If true, also show the labels alongside the node features. Only works for node level labels.
    :param open_browser: If true, opens the default web browser and opens the generated .html page.
    :return: Nothing.
    """
    y = graph.y if show_labels else None
    show(graph.x, zip(graph.a.row, graph.a.col), graph.e, y, str(graph), open_browser)
