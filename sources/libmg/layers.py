import tensorflow as tf
from tensorflow.python.keras import backend as K
from spektral.layers import MessagePassing

def loop_body(X, F, H, k, m, beta, lam, x0, bsz, f, y):
    k = k + 1
    n = min(k, m)
    G = tf.Variable(F[:, :n] - X[:, :n])

    H = H[:, 1:n + 1, 1:n + 1].assign(
        tf.matmul(G, tf.transpose(G, [0, 2, 1])) + tf.expand_dims(lam * tf.eye(n, dtype=x0.dtype), 0))
    alpha = tf.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]

    X = X[:, k % m].assign(
        beta * tf.matmul(alpha[:, None], F[:, :n])[:, 0] + (1 - beta) * tf.matmul(alpha[:, None], X[:, :n])[:, 0])
    F = F[:, k % m].assign(tf.reshape(f(tf.reshape(X[:, k % m], tf.shape(x0))), (bsz, -1)))

    return [X, F, H, k]


def anderson_mixing(f, x0, m=5, lam=1e-4, tol=1e-10, beta=1.0):
    n_nodes, n_node_features = tf.shape(x0)
    bsz = 1

    X = tf.Variable(tf.zeros((bsz, m, n_nodes * n_node_features), dtype=x0.dtype))
    F = tf.Variable(tf.zeros((bsz, m, n_nodes * n_node_features), dtype=x0.dtype))

    X = X[:, 0].assign(tf.reshape(x0, (bsz, -1)))
    F = F[:, 0].assign(tf.reshape(f(x0), (bsz, -1)))
    X = X[:, 1].assign(F[:, 0])
    F = F[:, 1].assign(tf.reshape(f(tf.reshape(F[:, 0],(tf.shape(x0)))),(bsz, -1)))

    H = tf.Variable(tf.zeros((bsz, m + 1, m + 1), dtype=x0.dtype))
    H = H[:, 0, 1:].assign(1)
    H = H[:, 1:, 0].assign(1)


    y = tf.Variable(tf.zeros((bsz, m + 1, 1), dtype=x0.dtype))
    y = y[:, 0].assign(1)

    output = tf.while_loop(
        cond=lambda XX, FF, HH, k: tf.math.logical_not(tf.math.reduce_all(tf.math.less_equal(tf.math.abs(tf.math.subtract(FF[:, k % m], XX[:, k % m])), tol))),
        body=lambda XX, FF, HH, k: loop_body(XX, FF, HH, k, m, beta, lam, x0, bsz, f, y),
        loop_vars=[X, F, H, 1]
    )

    X_f = output[0]
    k = output[-1]
    return tf.reshape(X_f[:, k % m], tf.shape(x0)), k

def standard_fixpoint(f, x0, tol=1e-10):
    return tf.while_loop(
        cond=lambda curr, prev, k: tf.math.logical_not(tf.math.reduce_all(tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)), tol))),
        body=lambda curr, prev, k: [f(curr), curr, k+1],
        loop_vars=[f(x0), x0, 1],
    )

# print(anderson_mixing(lambda x: x/2, tf.constant([[1.], [1.], [1.]])))
# print(standard_fixpoint(lambda x: x/2, tf.constant([[1.], [1.], [1.]])))

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
    def __init__(self, sigma, phi=lambda i, e, j: j, **kwargs):
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

    def __init__(self, gnn_x, precision=None, **kwargs):
        super().__init__(**kwargs)
        self.gnn_x = gnn_x
        if precision is not None:
            self.comparator = lambda curr, prev: tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)), precision)
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
        saved_args, X_o = x[:-1], x[-1:]
        additional_inputs = [a]
        if e is not None:
            additional_inputs.append(e)
        if i is not None:
            additional_inputs.append(i)
        X = [self.gnn_x(saved_args + X_o + additional_inputs)]
        # alternative: use set_shape in body of while
        return tf.while_loop(
            cond=lambda curr, prev: tf.math.logical_not(tf.math.reduce_all(self.comparator(curr[0], prev[0]))),
            body=lambda curr, prev: [[self.gnn_x(saved_args + curr + additional_inputs)], curr],
            loop_vars=[X, X_o],
            shape_invariants=[[X[0].get_shape()], [X[0].get_shape()]]
        )[0][0]


class Repeat(MessagePassing):

    def __init__(self, gnn_x, repeat, **kwargs):
        super().__init__(**kwargs)
        self.gnn_x = gnn_x
        self.repeat = repeat

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
        saved_args, X_o = x[:-1], x[-1:]
        additional_inputs = [a]
        if e is not None:
            additional_inputs.append(e)
        if i is not None:
            additional_inputs.append(i)
        # X = [self.gnn_x(saved_args + X_o + additional_inputs)]
        # alternative: use set_shape in body of while
        return tf.while_loop(
            cond=lambda curr, iter: tf.less(iter, self.repeat),
            body=lambda curr, iter: [[self.gnn_x(saved_args + curr + additional_inputs)], iter + 1],
            loop_vars=[X_o, 0],
            shape_invariants=[[X_o[0].get_shape()], None]
        )[0][0]