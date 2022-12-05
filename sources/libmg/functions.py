import tensorflow as tf
from collections import UserDict


# Custom dictionary class
class FunctionDict(UserDict):
    @staticmethod
    def parse_key(key):
        tokens = key.split('[')
        true_key = tokens[0]
        arg = '' if len(tokens) == 1 else tokens[1][: tokens[1].find(']')]
        return true_key, arg

    def __getitem__(self, key):
        true_key, arg = self.parse_key(key)
        return self.data[true_key](arg)

    def __setitem__(self, key, value):
        if isinstance(value, tf.keras.layers.Layer):
            self.data[key] = lambda x: value
        elif callable(value):
            self.data[key] = value
        else:
            raise ValueError("Invalid item:", str(value))


class Psi(tf.keras.layers.Layer):
    def __init__(self, single_op=None, multiple_op=None, **kwargs):
        """
        A general function applied on node labels f: (T*, T) -> U. For single graph datasets, which use the
        SingleGraphLoader, only the single_op parameter is necessary. For multiple graph datasets, using the
        MultipleGraphLoader, only the multiple_op parameter is necessary. The multiple_op argument is a function which
        takes an additional parameter to distinguish which values in the first argument refer to which graph. For
        more information, refer to the disjoint data mode in the Spektral library documentation.

        :param single_op: A function that transforms a Tensor of node labels of type T into a node label of type U.
         The function must be compatible with Tensorflow's broadcasting rules. The function takes only one argument of
         type Tensor[T] and uses broadcasting to emulate the tuple (T*, T) in the definition of f.
        :type single_op: (tf.Tensor[T]) -> tf.Tensor[U]
        :param multiple_op: A function that transforms a Tensor of node labels of type T and a Tensor of their
         respective graph indices of type int64 to a node label of type U. The function must be compatible with
         Tensorflow's broadcasting rules. The function must use broadcasting to emulate the tuple (T*, T) in the
         definition of f.
        :type multiple_op: (tf.Tensor[T], tf.Tensor[int]) -> tf.Tensor[U]
        """
        super().__init__(**kwargs)
        if single_op is None and multiple_op is None:
            raise ValueError("At least one operation must be defined!")
        self.single_op = single_op
        self.multiple_op = multiple_op

    def __call__(self, x, i=None):
        if i is not None:
            if self.multiple_op is None:
                raise ValueError("Undefined operation")
            return self.multiple_op(x, i)
        else:
            if self.single_op is None:
                raise ValueError("Undefined operation")
            return self.single_op(x)


class PsiLocal(Psi):
    def __init__(self, f, **kwargs):
        """
        A local transformation of node labels f: T -> U

        :param f: A function that transforms a Tensor of node labels of type T to a Tensor of node labels of type U.
         The function must be compatible with Tensorflow's broadcasting rules.
        :type f: (tf.Tensor[T]) -> tf.Tensor[U]
        """
        super().__init__(single_op=f)

    def __call__(self, x, i=None):
        return self.single_op(x)


class PsiGlobal(Psi):
    def __init__(self, single_op=None, multiple_op=None, **kwargs):
        """
        A global pooling operation on node labels f: T* -> U. For single graph datasets, which use the
        SingleGraphLoader, only the single_op parameter is necessary. For multiple graph datasets, using the
        MultipleGraphLoader, only the multiple_op parameter is necessary. The multiple_op argument is a function which
        takes an additional parameter to distinguish which values in the first argument refer to which graph. For
        more information, refer to the disjoint data mode in the Spektral library documentation.

        :param single_op: A function that transforms a Tensor of node labels of type T to a node label of type U.
        :type single_op: (tf.Tensor[T]) -> U
        :param multiple_op: A function that transforms a Tensor of node labels of type T and a Tensor of their
         respective graph indices of type int64 to a node label of type U.
        :type multiple_op: (tf.Tensor[T], tf.Tensor[int]) -> U
        """
        super().__init__(single_op=single_op, multiple_op=multiple_op)

    def __call__(self, x, i=None):
        if i is not None:
            if self.multiple_op is None:
                raise ValueError("Undefined operation")
            output = self.multiple_op(x, i)
            _, _, count = tf.unique_with_counts(i)
            return tf.repeat(output, count, axis=0)
        else:
            if self.single_op is None:
                raise ValueError("Undefined operation")
            output = self.single_op(x)
            return tf.repeat(output, tf.shape(x)[0], axis=0)


class Phi(tf.keras.layers.Layer):
    def __init__(self, f, **kwargs):
        """
         A function  f: (T, U, T) -> V to compute the message sent by a node i to a node j through edge e.

        :param f: A function applied on a triple composed of a Tensor of source node labels of type T, a Tensor of edge
         labels of type U, and a Tensor of target node labels of type T that returns a Tensor of node labels of type V.
        :type f: (tf.Tensor[T], tf.Tensor[U], tf.Tensor[T]) -> tf.Tensor[V]
        """
        super().__init__(**kwargs)
        self.f = f

    def __call__(self, src, e, tgt):
        return self.f(src, e, tgt)


class Sigma(tf.keras.layers.Layer):
    def __init__(self, f, **kwargs):
        """
        A function f: (T*, U) -> V to aggregate the messages sent to a node, including the current label of the node.

        :param f: A function of four arguments: a Tensor of messages of type T, a Tensor of integer indices that specify
         the id of the node each message is being sent to, a integer that specify the total number of nodes in the graph
         and finally a Tensor of node labels of type U. The function must return a Tensor of node labels of type V.
        :type f: (tf.Tensor[T], tf.Tensor[int], int, tf.Tensor[U]) -> tf.Tensor[V]
        """
        super().__init__(**kwargs)
        self.f = f

    def __call__(self, m, i, n, x):
        return self.f(m, i, n, x)
