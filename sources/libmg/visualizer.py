import tensorflow as tf
from lark import Lark
from pyvis.network import Network
from pyvis.options import Layout, Options
from spektral.data import Graph
from tensorflow.python.keras import backend as K
from libmg.grammar import mg_grammar

parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')


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
    return next((x for x in inputs if K.is_sparse(x)))


def find_edges(inputs):
    if len(inputs) == 3 and K.ndim(inputs[-1]) == 2:
        return inputs[-1]
    elif len(inputs) == 4:
        return inputs[-2]
    else:
        return None


def show(node_values, adj, edge_values, labels, filename, open_browser):
    if len(node_values.shape) != 2:
        raise ValueError("Not a valid node labeling function!")
    nodes = list(range(node_values.shape[0]))
    edges = [(int(i), int(j)) for i, j in zip(adj.row, adj.col)]
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
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx))
    inputs = inputs[0]
    idx_or_name = layer_idx if layer_idx is not None else layer_name
    assert idx_or_name is not None
    labels = labels.numpy() if labels is not None else None
    graph = Graph(x=debug_model(inputs).numpy(), a=find_adj(inputs).indices.numpy(), e=find_edges(inputs).numpy(), y=labels)
    print_graph(graph, False if labels is None else True, open_browser)


def print_graph(graph, show_labels=False, open_browser=True):
    y = graph.y if show_labels else None
    show(graph.x, graph.a, graph.e, y, str(graph), open_browser)
