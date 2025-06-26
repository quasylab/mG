"""Loads datasets to be processed by TensorFlow models.

This module defines two loaders for datasets, based on the loaders defined in the Spektral library.

The module contains the following classes:

- ``SingleGraphLoader(dataset, epochs=None, sample_weights=None)``
- ``MultipleGraphLoader(dataset, node_level=False, batch_size=1, epochs=None, shuffle=True)``
"""

import tensorflow as tf
import numpy as np
from spektral.data import SingleLoader, DisjointLoader
from spektral.data.utils import sp_matrices_to_sp_tensors, to_disjoint, prepend_none, to_tf_signature


class SingleGraphLoader(SingleLoader):
    """Loads a dataset made up by a single graph to be used by a TensorFlow model.

    Once instantiated, call the `load` method to obtain the generator that can be used with TensorFlow APIs.
    See Spektral's [documentation](https://graphneural.network/data-modes/#single-mode) for additional information,
    as this class is directly derived from Spektral's ``SingleLoader`` class.
    """

    def __init__(self, dataset, epochs=None, sample_weights=None):
        super().__init__(dataset, epochs, sample_weights)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=True)

        output = to_disjoint(**packed)
        output = output[:-1]  # Discard batch index
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        output = (output,)
        if y is not None:
            output += (y,)
        if self.sample_weights is not None:
            output += (self.sample_weights,)

        return output


class MultipleGraphLoader(DisjointLoader):
    """Loads a dataset made up by more than one graph to be used by a TensorFlow model.

    Once instantiated, call the `load` method to obtain the generator that can be used with TensorFlow APIs.
    See Spektral's [documentation](https://graphneural.network/data-modes/#single-mode/) for additional information,
    as this class is directly derived from Spektral's ``DisjointLoader`` class.
    """

    def __init__(self, dataset, node_level=True, batch_size=1, epochs=None, shuffle=True):
        super().__init__(dataset, node_level=node_level, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        n_nodes_list = [len(x) for x in packed['x_list']]

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level, n_nodes_list=n_nodes_list)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output,
        else:
            return output, y

    def tf_signature(self):
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        tf_signature = to_tf_signature(signature)
        if "y" not in signature:
            return tf_signature,
        else:
            return tf_signature


def collate_labels_disjoint(y_list, node_level=False, n_nodes_list=None):
    """
    Collates the given list of labels for disjoint mode.

    If `node_level=False`, the labels can be scalars or must have shape `[n_labels, ]`.
    If `node_level=True`, the labels must have shape `[n_nodes, ]` (i.e., a scalar label
    for each node) or `[n_nodes, n_labels]`.

    :param y_list: a list of np.arrays or scalars.
    :param node_level: bool, whether to pack the labels as node-level or graph-level.
    :return:
        - If `node_level=False`: a np.array of shape `[len(y_list), n_labels]`.
        - If `node_level=True`: a np.array of shape `[n_nodes_total, n_labels]`, where
        `n_nodes_total = sum(y.shape[0] for y in y_list)`.
    """
    if node_level:
        if len(np.shape(y_list[0])) == 1:
            y_list = [y_[:, None] for y_ in y_list]
        return np.vstack(y_list)
    else:
        assert n_nodes_list is not None
        if len(np.shape(y_list[0])) == 0:
            y_list = [np.array([y_]) for y_ in y_list]
        # return np.array(y_list)
        return np.vstack([np.tile(y, (n_nodes, 1)) for y, n_nodes in zip(y_list, n_nodes_list)])
