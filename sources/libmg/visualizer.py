import tensorflow as tf
from pyvis.network import Network
from tensorflow.python.keras import backend as K

def fetch_layer(model, layer_name=None, layer_idx=None):
    if layer_idx is not None:
        return model.get_layer(index=layer_idx).output
    else:
        return model.saved_layers[hash(layer_name)].x

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

def visualize(node_values, adj, edge_values=None):
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
    net.show('mygraph.html', notebook=False)

def print_layer(model, inputs, layer_name=None, layer_idx=None):
    debug_model = tf.keras.Model(inputs=model.inputs, outputs=fetch_layer(model, layer_name, layer_idx))
    visualize(debug_model(inputs), find_adj(inputs), find_edges(inputs))