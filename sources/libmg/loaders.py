import tensorflow as tf
from spektral.data import SingleLoader, DisjointLoader
from spektral.data.utils import collate_labels_disjoint, sp_matrices_to_sp_tensors, to_disjoint, prepend_none, \
    to_tf_signature


class SingleGraphLoader(SingleLoader):
    """A Loader that operates in [single mode](https://graphneural.network/data-modes/#single-mode).

    This loader produces Tensors representing a single graph. As such, it can
    only be used with Datasets of length 1 and the `batch_size` cannot be set.

    The loader supports sample weights through the `sample_weights` argument.
    If given, then each batch will be a tuple `(inputs, labels, sample_weights)`.

    **Arguments**

    - `dataset`: a `spektral.data.Dataset` object with only one graph;
    - `epochs`: int, number of epochs to iterate over the dataset. By default, (`None`) iterates indefinitely;
    - `shuffle`: bool, whether to shuffle the data at the start of each epoch;
    - `sample_weights`: Numpy array, will be appended to the output automatically.

    **Output**

    Returns a tuple `(inputs, labels)` or `(inputs, labels, sample_weights)`.

    `inputs` is a tuple containing the data matrices of the graph, only if they
    are not `None`:

    - `x`: same as `dataset[0].x`;
    - `a`: same as `dataset[0].a` (scipy sparse matrices are converted to SparseTensors);
    - `e`: same as `dataset[0].e`;

    `labels` is the same as `dataset[0].y`.

    `sample_weights` is the same array passed when creating the loader.
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
    """A Loader that operates in [disjoint mode](https://graphneural.network/data-modes/#disjoint-mode).

        This loader represents a batch of graphs via their disjoint union.

        The loader automatically computes a batch index tensor, containing integer
        indices that map each node to its corresponding graph in the batch.

        The adjacency matrix os returned as a SparseTensor, regardless of the input.

        If `node_level=False`, the labels are interpreted as graph-level labels and
        are stacked along an additional dimension.
        If `node_level=True`, then the labels are stacked vertically.

        **Note:** TensorFlow 2.4 or above is required to use this Loader's `load()`
        method in a Keras training loop.

        **Arguments**

        - `dataset`: a graph Dataset;
        - `node_level`: bool, if `True` stack the labels vertically for node-level prediction;
        - `batch_size`: size of the mini-batches;
        - `epochs`: number of epochs to iterate over the dataset. By default, (`None`) iterates indefinitely;
        - `shuffle`: whether to shuffle the data at the start of each epoch.

        **Output**

        For each batch, returns a tuple `(inputs, labels)`.

        `inputs` is a tuple containing:

        - `x`: node attributes of shape `[n_nodes, n_node_features]`;
        - `a`: adjacency matrices of shape `[n_nodes, n_nodes]`;
        - `e`: edge attributes of shape `[n_edges, n_edge_features]`;
        - `i`: batch index of shape `[n_nodes]`.

        `labels` have shape `[batch, n_labels]` if `node_level=False` or
        `[n_nodes, n_labels]` otherwise.

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
