import tensorflow as tf
from pyvis.network import Network
from pyvis.options import Layout, Options

from libmg.utils import fetch_layer, unpack_inputs


def show(node_values, adj, edge_values, labels, hierarchy, title_generator, filename, open_browser):
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
    nodes = [title_generator(i) for i in range(node_values.shape[0])]
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
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx).output)
    idx_or_name = layer_idx if layer_idx is not None else layer_name
    if idx_or_name is None:
        raise ValueError("Layer name or index must be provided!")
    labels = labels.numpy() if labels is not None else None
    _, a, e, _ = unpack_inputs(inputs)
    show(debug_model(inputs).numpy(), a.indices.numpy(), e.numpy() if e is not None else None, labels, None, lambda x: x, str(idx_or_name), open_browser)


def print_graph(graph, node_names_func='id', hierarchical=False, show_labels=False, open_browser=True):
    """
    Visualizes a graph using PyVis.

    :param graph: A ``Graph``.
    :param show_labels: If true, also show the labels alongside the node features. Only works for node level labels.
    :param open_browser: If true, opens the default web browser and opens the generated .html page.
    :return: Nothing.
    """
    show(graph.x, zip(graph.a.row, graph.a.col), graph.e, graph.y if show_labels else None, graph.hierarchy if hierarchical else None, (lambda x: x) if node_names_func == 'id' else node_names_func, str(graph), open_browser)
