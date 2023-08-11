import tensorflow as tf
from lark import Lark
from lark.reconstruct import Reconstructor
from pyvis.network import Network
from tensorflow.python.keras import backend as K
from libmg.grammar import mg_grammar

parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')
reconstructor = Reconstructor(parser)


def get_name(idx_or_subformula):
    if type(idx_or_subformula) is int:
        return str(idx_or_subformula)
    else:
        return reconstructor.reconstruct(parser.parse(idx_or_subformula))


def fetch_layer(model, layer_name=None, layer_idx=None):
    if layer_idx is not None:
        return model.get_layer(index=layer_idx).output
    else:
        layer_hash = hash(parser.parse(layer_name))
        if layer_hash in model.saved_layers:
            return model.saved_layers[layer_hash].x
        else:
            raise ValueError("No layer found with name: " + layer_name)


def find_adj(inputs):
    if len(inputs) == 1:
        inputs, = inputs
    return next((x for x in inputs if K.is_sparse(x)))


def find_edges(inputs):
    if len(inputs) == 1:
        inputs, = inputs

    if len(inputs) == 3 and K.ndim(inputs[-1]) == 2:
        return inputs[-1]
    elif len(inputs) == 4:
        return inputs[-2]
    else:
        return None


def show(nodes, node_labels, edges, edge_labels, titles, filename, open_browser):
    net = Network(directed=True, neighborhood_highlight=True, select_menu=True, filter_menu=True)
    net.add_nodes(nodes, title=titles, label=node_labels, shape=['circle']*len(nodes))
    if edge_labels is None:
        net.add_edges(edges)
    else:
        for i, edge in enumerate(edges):
            net.add_edge(*edge, label=edge_labels[i])
    net.barnes_hut(gravity=-2000, spring_length=250, spring_strength=0.04)
    net.show_buttons()
    if open_browser:
        net.show('graph_' + filename + '.html', notebook=False)
    else:
        net.save_graph('graph_' + filename + '.html')


def visualize_tf(node_values, adj, edge_values, filename, open_browser):
    if K.is_sparse(node_values) or not K.ndim(node_values) == 2:
        raise ValueError("Not a valid node labeling function!")
    nodes = list(range(node_values.shape[0]))
    edges = [(int(e[0]), int(e[1])) for e in adj.indices.numpy()]
    node_labels = [' | '.join([str(label) for label in label_list]) for label_list in node_values.numpy().tolist()]
    if edge_values is not None:
        edge_labels = [' | '.join([str(label) for label in label_list]) for label_list in edge_values.numpy().tolist()]
    else:
        edge_labels = None
    titles = [str(i) for i in nodes]
    show(nodes, node_labels, edges, edge_labels, titles, filename, open_browser)


def visualize_np(node_values, adj, edge_values, filename, open_browser):
    if len(node_values.shape) != 2:
        raise ValueError("Not a valid node labeling function!")
    nodes = list(range(node_values.shape[0]))
    edges = [(int(i), int(j)) for i, j in zip(adj.row, adj.col)]
    node_labels = [' | '.join([str(label) for label in label_list]) for label_list in node_values.tolist()]
    if edge_values is not None:
        edge_labels = [' | '.join([str(label) for label in label_list]) for label_list in edge_values.tolist()]
    else:
        edge_labels = None
    titles = [str(i) for i in nodes]
    show(nodes, node_labels, edges, edge_labels, titles, filename, open_browser)


def print_layer(model, inputs, layer_name=None, layer_idx=None, open_browser=True):
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx))
    idx_or_name = layer_idx if layer_idx is not None else layer_name
    assert idx_or_name is not None
    visualize_tf(debug_model(inputs), find_adj(inputs), find_edges(inputs), get_name(idx_or_name), open_browser)


def print_graph(graph, open_browser=True):
    visualize_np(graph.x, graph.a, graph.e, str(graph), open_browser)
