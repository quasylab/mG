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

def visualize(node_values, adj, edge_values, file_name, open_browser):
    if K.is_sparse(node_values) or not K.ndim(node_values) == 2:
        raise ValueError("Not a valid node labeling function!")
    nodes = list(range(adj.dense_shape[0].numpy()))
    edges = [(int(e[0]), int(e[1])) for e in adj.indices.numpy()]
    node_labels = [' | '.join([str(label) for label in l]) for l in node_values.numpy().tolist()]
    titles = [str(i) for i in nodes]
    net = Network(directed=True, neighborhood_highlight=True, select_menu=True, filter_menu=True)
    net.add_nodes(nodes, title=titles, label=node_labels, shape=['circle']*len(nodes))
    if edge_values is None:
        net.add_edges(edges)
    else:
        edge_labels = [' | '.join([str(label) for label in l]) for l in edge_values.numpy().tolist()]
        for i, edge in enumerate(edges):
            net.add_edge(*edge, label=edge_labels[i])
    net.barnes_hut(gravity=-2000, spring_length=250, spring_strength=0.04)
    net.show_buttons()
    if open_browser:
        net.show('graph_' + file_name + '.html', notebook=False)
    else:
        net.save_graph('graph_' + file_name + '.html')

def print_layer(model, inputs, layer_name=None, layer_idx=None, open_browser=True):
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx))
    visualize(debug_model(inputs), find_adj(inputs), find_edges(inputs), get_name(layer_idx or layer_name), open_browser)
