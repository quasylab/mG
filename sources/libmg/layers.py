import tensorflow as tf
from tensorflow.python.keras import backend as K
from spektral.layers import MessagePassing


class FunctionApplication(MessagePassing):

    def __init__(self, psi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psi = psi

    def call(self, inputs, **kwargs):
        x, i = self.get_inputs(inputs)
        return self.propagate(x, i)

    @staticmethod
    def get_inputs(inputs):
        if len(inputs) == 1:
            x, = inputs
            i = None
        else:
            x, i = inputs
        return x, i

    def propagate(self, x, i=None, **kwargs):
        return self.psi(x, i)


class PreImage(MessagePassing):
    def __init__(self, sigma, phi=lambda i, e, j: i, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.phi = phi

    def message(self, x, e=None, **kwargs):
        return self.phi(self.get_targets(x), e, self.get_sources(x))

    def aggregate(self, messages, x=None, **kwargs):
        return self.sigma(messages, self.index_targets, self.n_nodes, x)


class PostImage(MessagePassing):
    def __init__(self, sigma, phi=lambda i, e, j: j, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.phi = phi

    def message(self, x, e=None, **kwargs):
        return self.phi(self.get_sources(x), e, self.get_targets(x))

    def aggregate(self, messages, x=None, **kwargs):
        return self.sigma(messages, self.index_sources, self.n_nodes, x)


class Ite(MessagePassing):
    def __init__(self, iftrue, iffalse, **kwargs):
        super().__init__(**kwargs)
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.iftrue_input_len = len(self.iftrue.inputs)
        self.iffalse_input_len = len(self.iffalse.inputs)

    @staticmethod
    def get_inputs(inputs):
        if K.is_sparse(inputs[-1]):
            *x, a = inputs
            e = None
            i = None
        elif K.ndim(inputs[-1]) == 2:
            *x, a, e = inputs
            i = None
        elif K.ndim(inputs[-1]) == 1 and K.is_sparse(inputs[-2]):
            *x, a, i = inputs
            e = None
        else:
            *x, a, e, i = inputs
        return x, a, e, i

    def call(self, inputs, **kwargs):
        x, a, e, i = self.get_inputs(inputs)
        return self.propagate(x, a, e, i)

    def propagate(self, x, a, e=None, i=None, **kwargs):
        test = x[0]
        values = x[1:]  # might also have a fix var in here
        inputs = [a]
        if e is not None:
            inputs.append(e)
        if i is not None:
            iftrue_input_len = self.iftrue_input_len - 2
            iffalse_input_len = self.iffalse_input_len - 2
            _, _, count = tf.unique_with_counts(i)
            partitioned_test = tf.ragged.stack_dynamic_partitions(test, tf.cast(i, dtype=tf.int32), tf.size(count))
            partitioned_values = tf.ragged.stack_dynamic_partitions(values[0], tf.cast(i, dtype=tf.int32), tf.size(count))
            partitioned_idx = tf.ragged.stack_dynamic_partitions(i, tf.cast(i, dtype=tf.int32), tf.size(count))
            if len(values) == 1:
                return tf.reshape(tf.map_fn(lambda args: tf.cond(tf.reduce_all(args[0]), lambda: self.iftrue([args[1]] + inputs + [args[2]]),
                                                                 lambda: self.iffalse([args[1]] + inputs + [args[2]])),
                                            (partitioned_test, partitioned_values, partitioned_idx), swap_memory=True,
                                            infer_shape=False, fn_output_signature=self.iftrue.outputs[0].type_spec),
                                  [-1, self.iftrue.outputs[0].shape[-1]])
            else:
                partitioned_fix_var = tf.ragged.stack_dynamic_partitions(values[1], tf.cast(i, dtype=tf.int32), tf.size(count))
                return tf.reshape(tf.map_fn(lambda args: tf.cond(tf.reduce_all(args[0]),
                                                                 lambda: self.iftrue([args[1], args[2]][:iftrue_input_len] + inputs + [args[3]]),
                                                                 lambda: self.iffalse([args[1], args[2]][:iffalse_input_len] + inputs + [args[3]])),
                                            (partitioned_test, partitioned_values, partitioned_fix_var, partitioned_idx), swap_memory=True,
                                            infer_shape=False, fn_output_signature=self.iftrue.outputs[0].type_spec),
                                  [-1, self.iftrue.outputs[0].shape[-1]])
        else:
            inputs_iftrue = values[:self.iftrue_input_len - 1] + inputs
            inputs_iffalse = values[:self.iffalse_input_len - 1] + inputs
            return tf.cond(tf.reduce_all(test), lambda: self.iftrue(inputs_iftrue), lambda: self.iffalse(inputs_iffalse))


class FixPoint(MessagePassing):

    def __init__(self, gnn_x, top, precision=None, **kwargs):
        super().__init__(**kwargs)
        self.gnn_x = gnn_x
        self.top = top
        # in a GFP, current value should always be <= than the previous value therefore we can avoid using abs
        if precision is not None:
            self.comparator = lambda curr, prev: prev - curr <= precision
        else:
            self.comparator = lambda curr, prev: curr == prev

    def call(self, inputs, **kwargs):
        x, a, e, i = self.get_inputs(inputs)
        return self.propagate(x, a, e, i)

    @staticmethod
    def get_inputs(inputs):
        if K.is_sparse(inputs[-1]):
            *x, a = inputs
            e = None
            i = None
        elif K.ndim(inputs[-1]) == 2:
            *x, a, e = inputs
            i = None
        elif K.ndim(inputs[-1]) == 1 and K.is_sparse(inputs[-2]):
            *x, a, i = inputs
            e = None
        else:
            *x, a, e, i = inputs
        return x, a, e, i

    def propagate(self, x, a, e=None, i=None, **kwargs):
        if len(x) > 0:
            X_o = [self.top(x[0])]
        else:
            X_o = [self.top(a)]
        additional_inputs = [a]
        if e is not None:
            additional_inputs.append(e)
        if i is not None:
            additional_inputs.append(i)
        # x is supposed to be a list of lists e.g. [[x1, a1, e1, i1], [], []]
        # X_o.extend(additional_inputs)
        X = [self.gnn_x(x + X_o + additional_inputs)]
        return tf.while_loop(
            cond=lambda curr, prev: tf.math.logical_not(tf.math.reduce_all(self.comparator(curr[0], prev[0]))),
            body=lambda curr, prev: [[self.gnn_x(x + [curr] + additional_inputs)], curr],
            loop_vars=[X, X_o],
        )[0][0]
