from tensorflow.python.keras import backend as K

from libmg.grammar import mg_parser




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
        return model.get_layer(index=layer_idx)
    elif layer_name is not None:
        layer_hash = hash(mg_parser.parse(layer_name))
        if layer_hash in model.mg_layers:
            return model.mg_layers[layer_hash]
        else:
            raise ValueError("No layer found with name: " + layer_name)
    else:
        raise ValueError("Layer name or index must be provided!")


def unpack_inputs(inputs):
    if K.is_sparse(inputs[-1]):
        node_feats, adj = inputs
        edge_feats = None
        indexes = None
    elif K.ndim(inputs[-1]) == 2:
        node_feats, adj, edge_feats = inputs
        indexes = None
    elif K.ndim(inputs[-1]) == 1 and K.is_sparse(inputs[-2]):
        node_feats, adj, indexes = inputs
        edge_feats = None
    else:
        node_feats, adj, edge_feats, indexes = inputs
    return node_feats, adj, edge_feats, indexes