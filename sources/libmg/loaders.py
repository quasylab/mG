import tensorflow as tf
from spektral.data import SingleLoader, DisjointLoader
from spektral.data.utils import collate_labels_disjoint, sp_matrices_to_sp_tensors, to_disjoint, prepend_none, \
    to_tf_signature


class SingleGraphLoader(SingleLoader):
    """
    A ``Loader`` for a dataset consisting of a single graph instance.
    See Spektral's `documentation <https://graphneural.network/data-modes/#single-mode/>`_ for additional information,
    as this class is directly derived from Spektral's ``SingleLoader``.
    """

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
    """
    A ``Loader`` for a dataset consisting of multiple graphs, representing a batch as a disjoint union of graphs.
    See Spektral's `documentation <https://graphneural.network/data-modes/#disjoint-mode/>`_ for additional information,
    as this class is directly derived from Spektral's ``DisjointLoader``.
    """

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
