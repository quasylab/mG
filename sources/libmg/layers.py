import tensorflow as tf
from tensorflow.python.keras import backend as K
from spektral.layers import MessagePassing


def loop_body(X, F, H, k, m, beta, lam, x0, f, y, fixpoint_printer):
    k = k + 1
    n = tf.math.minimum(k, m)
    G = tf.reshape(F.stack()[:n] - X.stack()[:n], shape=[n, -1])

    updates_H = tf.matmul(G, tf.transpose(G)) + lam * tf.eye(n, dtype=x0.dtype)
    updates_H = tf.concat([tf.ones((n, 1), dtype=x0.dtype), updates_H], axis=1)
    updates_H = tf.concat([updates_H, tf.zeros((n, m-n), dtype=x0.dtype)], axis=1)
    H = tf.tensor_scatter_nd_update(H, tf.expand_dims(tf.range(start=1, limit=n+1), axis=1), updates_H)

    alpha = tf.linalg.solve(H[:n + 1, :n + 1], y[:n + 1])[1:n + 1, 0]

    updates_X = beta * tf.matmul(alpha[None], tf.reshape(F.stack()[:n], shape=[n, -1]))[0] + (1 - beta) * tf.matmul(alpha[None], tf.reshape(X.stack()[:n], shape=[n, -1]))[0]
    X = X.write(k % m, fixpoint_printer(tf.reshape(updates_X, shape=tf.shape(x0))))
    updates_F = f(X.read(k % m))
    F = F.write(k % m, updates_F)

    return [X, F, H, k]


def anderson_mixing(f, x0, comparator, fixpoint_printer, m=5, lam=1e-4, beta=1.0):
    # nodes * node feats = size
    X = tf.TensorArray(dtype=x0.dtype, size=m, clear_after_read=False)
    F = tf.TensorArray(dtype=x0.dtype, size=m, clear_after_read=False)

    X = X.write(0, x0)
    F = F.write(0, f(x0))
    X = X.write(1, F.read(0))
    F = F.write(1, f(F.read(0)))

    H = tf.sequence_mask(tf.ones((m + 1,)), m + 1, dtype=x0.dtype)
    H = tf.tensor_scatter_nd_update(H, [[0]], tf.expand_dims(
        tf.cast(tf.math.logical_not(tf.sequence_mask(1, m + 1)), dtype=x0.dtype), axis=0))

    y = tf.ones((m + 1, 1), dtype=x0.dtype)

    output = tf.while_loop(
        cond=lambda XX, FF, HH, k: tf.math.logical_not(tf.math.reduce_all(comparator(FF.read(k % m), XX.read(k % m)))),
        body=lambda XX, FF, HH, k: loop_body(XX, FF, HH, k, m, beta, lam, x0, f, y, fixpoint_printer),
        loop_vars=[X, F, H, 1]
    )

    X_f = output[0]
    k = output[-1]
    return [[X_f.read(k % m)], [X_f.read((k - 1) % m)], k]


def anderson_fixpoint(f, x0, comparator, fixpoint_printer):
    return anderson_mixing(lambda x: f([x])[0], x0[0], comparator, fixpoint_printer)


def standard_fixpoint(f, x0, comparator, fixpoint_printer):
    iter = tf.constant(1)
    x = f(x0)
    return tf.while_loop(
        cond=lambda curr, prev, k: tf.math.logical_not(tf.math.reduce_all(comparator(curr[0], prev[0]))),
        body=lambda curr, prev, k: [f(curr), fixpoint_printer(curr), k+1],
        loop_vars=[x, x0, iter],
        shape_invariants=[[x[0].get_shape()], [x[0].get_shape()], iter.get_shape()]
    )


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

    @staticmethod
    def fixpoint_print_and_return(x):
        tf.print(x, output_stream=tf.compat.v1.logging.info)
        return x

    def __init__(self, gnn_x, precision, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.fixpoint_print = self.fixpoint_print_and_return if debug else lambda x: x
        self.gnn_x = gnn_x
        tolerance, solver = precision
        if tolerance is not None:
            self.comparator = lambda curr, prev: tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)), tolerance)
            self.solver = standard_fixpoint if solver == 'iter' else anderson_fixpoint
        else:
            self.comparator = lambda curr, prev: curr == prev
            self.solver = standard_fixpoint

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

        # compute forward pass without tracking the gradient
        output = tf.nest.map_structure(tf.stop_gradient, self.solver(lambda x: [self.gnn_x(saved_args + x + additional_inputs)], X_o, lambda curr, prev: self.comparator(curr, prev), self.fixpoint_print))
        tf.print('fixpoint (solver = {0}) iters: '.format(self.solver.__name__), output[-1], output_stream=tf.compat.v1.logging.info)

        # compute forward pass with the gradient being tracked
        otp = output[0]
        output = self.gnn_x(saved_args + otp + additional_inputs)

        @tf.custom_gradient
        def fixgrad(x):
            def custom_grad(dy):
                @tf.function
                def f(y):
                    return tf.gradients(output, otp, y)[0] + dy
                x0 = dy
                x = f(x0)
                grad = tf.while_loop(
                    cond=lambda curr, prev: tf.math.logical_not(tf.math.reduce_all(tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)), 0.001))),
                    body=lambda curr, prev: [f(curr), curr],
                    loop_vars=[x, x0],
                )
                return grad[0]
            return tf.identity(x), custom_grad

        return fixgrad(output)


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