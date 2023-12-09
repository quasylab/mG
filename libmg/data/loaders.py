"""Loads datasets to be processed by TensorFlow models.

This module defines two loaders for datasets, based on the loaders defined in the Spektral library.

The module contains the following classes:

- ``SingleGraphLoader(dataset, epochs=None, sample_weights=None)``
- ``MultipleGraphLoader(dataset, node_level=False, batch_size=1, epochs=None, shuffle=True)``
"""

import tensorflow as tf
from spektral.data import SingleLoader, DisjointLoader
from spektral.data.utils import collate_labels_disjoint, sp_matrices_to_sp_tensors, to_disjoint, prepend_none, to_tf_signature


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

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        super().__init__(dataset, node_level=True, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

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
