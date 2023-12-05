"""Defines the layers to be used in mG models.

This module defines the layers necessary to build mG models,as subclasses of Spetral's ``MessagePassing`` class. The algorithm to compute the gradients
of fixpoint computation is defined here in the ``FixPoint`` class. The implementation of Anderson's mixing has been momentarily removed.

The module contains the following functions:
- ``unpack_inputs(inputs, accumulate_x)``

The module contains the following classes:
- ``MGLayer``
- ``FunctionApplication``
- ``PreImage``
- ``PostImage``
- ``Ite``
- ``FixPoint``
- ``Repeat``
"""

from typing import Callable, Any

import tensorflow as tf
import re
from tensorflow.python.eager.backprop_util import IsTrainable
from tensorflow.python.keras import backend
from spektral.layers import MessagePassing
from tensorflow.python.keras.utils import generic_utils

from libmg.compiler.functions import Phi, PsiNonLocal, Sigma


# def loop_body(X, F, H, k, m, beta, lam, x0, f, y, fixpoint_printer):
#     """
#     Body of the loop in the ``anderson_mixing`` function.
#
#     :param X: A ``TensorArray`` that contains the past ``m`` guesses for a fixed point.
#     :param F: A ``TensorArray`` that contains the past ``m`` outputs of ``f``.
#     :param H: A ``Tensor`` that encodes a linear system to solve.
#     :param k: Current iteration number
#     :param m: Number of previous guesses to keep into consideration.
#     :param beta: For beta < 1, makes a damped version of Anderson update. For beta > 1 makes an overprojected version that might help convergence.
#     :param lam: This small value is needed to make the linear system solvable.
#     :param x0: A ``list`` of inputs for the model, acts as initial guess for the fixed point.
#     :param f: A mG model for which we want to compute the fixed point.
#     :param y: A ``Tensor`` that encodes the constant terms of the linear system in ``H``.
#     :param fixpoint_printer: A ``callable`` that handles the printing of its input and then returns it.
#     :return: A ``list`` containing the loop variables X, F, H, and k.
#     """
#     k = k + 1
#     n = tf.math.minimum(k, m)
#     G = tf.reshape(F.stack()[:n] - X.stack()[:n], shape=[n, -1])
#
#     updates_H = tf.matmul(G, tf.transpose(G)) + lam * tf.eye(n, dtype=x0.dtype)
#     updates_H = tf.concat([tf.ones((n, 1), dtype=x0.dtype), updates_H], axis=1)
#     updates_H = tf.concat([updates_H, tf.zeros((n, m - n), dtype=x0.dtype)], axis=1)
#     H = tf.tensor_scatter_nd_update(H, tf.expand_dims(tf.range(start=1, limit=n + 1), axis=1), updates_H)
#
#     alpha = tf.linalg.solve(H[:n + 1, :n + 1], y[:n + 1])[1:n + 1, 0]
#
#     updates_X = beta * tf.matmul(alpha[None], tf.reshape(F.stack()[:n], shape=[n, -1]))[0] + (1 - beta) * \
#                 tf.matmul(alpha[None], tf.reshape(X.stack()[:n], shape=[n, -1]))[0]
#     X = X.write(k % m, fixpoint_printer(tf.reshape(updates_X, shape=tf.shape(x0))))
#     updates_F = f(X.read(k % m))
#     F = F.write(k % m, updates_F)
#
#     return [X, F, H, k]
#
#
# def anderson_mixing(f, x0, comparator, fixpoint_printer, m=5, lam=1e-4, beta=1.0):
#     """
#     This function implements Anderson acceleration (aka Anderson mixing).
#
#     :param f: A mG model for which we want to compute the fixed point.
#     :param x0: A ``list`` of inputs for the model, acts as initial guess for the fixed point.
#     :param comparator: A ``callable`` of two arguments, which should return ``True`` if its two inputs are the *equal*.
#     :param fixpoint_printer: A ``callable`` that handles the printing of its input and then returns it.
#     :param m: Number of previous guesses to keep into consideration.
#     :param lam: This small value is needed to make the linear system solvable.
#     :param beta: For beta < 1, makes a damped version of Anderson update. For beta > 1 makes an overprojected version that might help convergence.
#     :return: The fixed point of ``f`` computed starting from ``x0``, the value of ``f`` in the previous steps, and number of iterations.
#     """
#     # nodes * node feats = size
#     X = tf.TensorArray(dtype=x0.dtype, size=m, clear_after_read=False)
#     F = tf.TensorArray(dtype=x0.dtype, size=m, clear_after_read=False)
#
#     X = X.write(0, x0)
#     F = F.write(0, f(x0))
#     X = X.write(1, F.read(0))
#     F = F.write(1, f(F.read(0)))
#
#     H = tf.sequence_mask(tf.ones((m + 1,)), m + 1, dtype=x0.dtype)
#     H = tf.tensor_scatter_nd_update(H, [[0]], tf.expand_dims(
#         tf.cast(tf.math.logical_not(tf.sequence_mask(1, m + 1)), dtype=x0.dtype), axis=0))
#
#     y = tf.ones((m + 1, 1), dtype=x0.dtype)
#
#     output = tf.while_loop(
#         cond=lambda XX, FF, HH, k: tf.math.logical_not(tf.math.reduce_all(comparator(FF.read(k % m), XX.read(k % m)))),
#         body=lambda XX, FF, HH, k: loop_body(XX, FF, HH, k, m, beta, lam, x0, f, y, fixpoint_printer),
#         loop_vars=[X, F, H, 1]
#     )
#
#     X_f = output[0]
#     k = output[-1]
#     return [[X_f.read(k % m)], [X_f.read((k - 1) % m)], k]
#
#
# def anderson_fixpoint(f, x0, comparator, fixpoint_printer):
#     """
#     Fixed point solver that uses Anderson acceleration (aka Anderson mixing).
#
#     :param f: A mG model for which we want to compute the fixed point.
#     :param x0: A ``list`` of inputs for the model, acts as initial guess for the fixed point.
#     :param comparator: A ``callable`` of two arguments, which should return ``True`` if its two inputs are the *equal*.
#     :param fixpoint_printer: A ``callable`` that handles the printing of its input and then returns it.
#     :return: The fixed point of ``f`` computed starting from ``x0``, the value of ``f`` in the previous steps, and number of iterations.
#     """
#     return anderson_mixing(lambda x: f([x])[0], x0[0], comparator, fixpoint_printer)

def unpack_inputs(inputs: list[tf.Tensor], accumulate_x: bool = False) \
        -> tuple[tf.Tensor | list[tf.Tensor], tf.SparseTensor, tf.Tensor | None, tf.Tensor | None]:
    """Examines a list of tensors and returns it unpacked into its constituents.

    If accumulate_x is true, the first element of the output will be a list of tensors instead of a single tensor.

    Args:
        inputs: The list to be unpacked.
        accumulate_x: If true, accumulate all the extra tensors at the head of the input list in a list.

    Returns:
        A 4-element tuple, which may contain ``None`` values for missing elements, that corresponds to the unpacked list.
    """
    if backend.is_sparse(inputs[-1]):
        if accumulate_x:
            *node_feats, adj = inputs
        else:
            node_feats, adj = inputs
        edge_feats = None
        indexes = None
    elif backend.ndim(inputs[-1]) == 2:
        if accumulate_x:
            *node_feats, adj, edge_feats = inputs
        else:
            node_feats, adj, edge_feats = inputs
        indexes = None
    elif backend.ndim(inputs[-1]) == 1 and backend.is_sparse(inputs[-2]):
        if accumulate_x:
            *node_feats, adj, indexes = inputs
        else:
            node_feats, adj, indexes = inputs
        edge_feats = None
    else:
        if accumulate_x:
            *node_feats, adj, edge_feats, indexes = inputs
        else:
            node_feats, adj, edge_feats, indexes = inputs
    return node_feats, adj, edge_feats, indexes


class MGLayer(MessagePassing):
    """Base class for all mG layers.

    Provides the means to set names to mG layers. Due to TensorFlow naming rules (``tf_name_regex``) all layer names that contain symbols must be mapped
    to the appropriate string name. The name is taken from the corresponding function name for the case of ``FunctionApplication``, ``PreImage``, and
    ``PostImage`` layers, while the class name is used for all other layers.
    """
    tf_name_regex = re.compile(r"^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$")
    translate_dict = {
        '~': 'tilde', '`': 'tick', '!': 'excl', '@': 'at', '#': 'hash', '£': 'pound', '€': 'euro', '$': 'dollar',
        '¢': 'cent', '¥': 'yuan', '§': 'sect', '%': 'perc', '°': 'deg', '^': 'caret', '&': 'and', '*': 'star',
        '(': 'lpar', ')': 'rpar', '+': 'plus', '=': 'eq', '{': 'lcur', '}': 'rcur',
        '[': 'lsqr', ']': 'rsqr', '|': 'pipe', ':': 'colon', ';': 'semicolon',
        '\"': 'quote', '\'': 'apos', '<': 'less', ',': 'comma', '?': 'qst', ' ': '.'
    }
    translate_table = str.maketrans(translate_dict)

    def __init__(self, name: str | None = None):
        """Initializes the instance with a name, if present.

        Args:
            name: Name of the layer. The name will be sanitized, and then show up in ``model.summary()`` calls.
        """
        if name is None:
            super().__init__()
        else:
            super().__init__(name=backend.unique_object_name(generic_utils.to_snake_case(self.__class__.__name__) + '_' + self.validate_name(name),
                                                             zero_based=True))

    @staticmethod
    def validate_name(name: str) -> str:
        """Checks whether the input string satisfies TensorFlow's layer naming constraints, and returns a sanitized version of it.

        Args:
            name: The string to check.

        Returns:
            If the input string satisfies the constraint, the string is returned as is. Otherwise, returns a translated version according to ``translate_dict``.
        """
        if MGLayer.tf_name_regex.match(name):
            return name
        else:
            return name.translate(MGLayer.translate_table)


class FunctionApplication(MGLayer):
    """Layer that applies a mG psi function.

    Attributes:
        psi: The psi function to apply.
    """

    def __init__(self, psi: PsiNonLocal):
        """Initializes the instance with a psi function.

        Args:
            psi: The psi function that this layer will apply.
        """
        super().__init__(psi.name)
        self.psi = psi

    def call(self, inputs: list[tf.Tensor], **kwargs) -> tf.Tensor:
        if len(inputs) == 1:
            x, = inputs
            i = None
        else:
            x, i = inputs
        return self.propagate(x, i)

    def propagate(self, x: tf.Tensor, i: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        return self.psi(x, i)


class PreImage(MGLayer):
    """Layer that applies a mG pre-image expression.

    The layer computes a multiset of messages using a phi function and then aggregates them using a sigma function.
    Each node receives the messages from its predecessors, following the direction of the directed edges.

    Attributes:
        sigma: The sigma function that aggregates messages.
        phi: The phi function that generates messages.
    """

    def __init__(self, sigma: Sigma, phi: Phi = Phi(lambda i, e, j: i, name='Pi1')):
        """Initializes the instance with the given sigma and phi functions.

        Args:
            sigma: The sigma function that aggregates messages.
            phi: The phi function that generates messages, defaults to sending the node label of the sender node.
        """
        super().__init__(phi.name + '_' + sigma.name)
        self.sigma = sigma
        self.phi = phi

    def message(self, x: tf.Tensor, e: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        return self.phi(self.get_sources(x), e, self.get_targets(x))

    def aggregate(self, messages: tf.Tensor, x: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        return self.sigma(messages, self.index_targets, self.n_nodes, x)


class PostImage(MGLayer):
    """Layer that applies a mG post-image expression.

    The layer computes a multiset of messages using a phi function and then aggregates them using a sigma function.
    Each node receives the messages from its successors, following the opposite direction of the directed edges.

    Attributes:
        sigma: The sigma function that aggregates messages.
        phi: The phi function that generates messages.
    """

    def __init__(self, sigma: Sigma, phi: Phi = Phi(lambda i, e, j: i, name='Pi1')):
        """Initializes the instance with the given sigma and phi functions.

        Args:
            sigma: The sigma function that aggregates messages.
            phi: The phi function that generates messages, defaults to sending the node label of the sender node.
        """
        super().__init__(phi.name + '_' + sigma.name)
        self.sigma = sigma
        self.phi = phi

    def message(self, x: tf.Tensor, e: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        return self.phi(self.get_targets(x), e, self.get_sources(x))

    def aggregate(self, messages: tf.Tensor, x: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        return self.sigma(messages, self.index_sources, self.n_nodes, x)


class Ite(MGLayer):
    """Layer that applies a mG choice expression, i.e. an if-then-else (ITE) construct.

    The layer checks whether the test is true across all nodes of the graph, and runs one model or the other accordingly. In the case the input contains
    multiple graphs, the layer runs the right model on each of them separately, then merges the output.

    Attributes:
        iftrue: The model to if the test is true.
        iffalse: The model to if the test is false.
        iftrue_input_len: The number of inputs that the model ``iftrue`` expects to receive.
        iffalse_input_len: The number of inputs that the model ``iffalse`` expects to receive.
    """

    def __init__(self, iftrue: tf.keras.Model, iffalse: tf.keras.Model):
        """Initializes the instance with the models to run according to the outcome of the test.

        Args:
            iftrue: The model to if the test is true.
            iffalse: The model to if the test is false.
        """
        super().__init__()
        self.iftrue = iftrue
        self.iffalse = iffalse
        self.iftrue_input_len = len(self.iftrue.inputs)
        self.iffalse_input_len = len(self.iffalse.inputs)

    def call(self, inputs: list[tf.Tensor], **kwargs) -> tf.Tensor:
        x, a, e, i = unpack_inputs(inputs, True)
        return self.propagate(x, a, e, i)

    def propagate(self, x: tf.Tensor, a: tf.SparseTensor, e: tf.Tensor | None = None, i: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        test = x[0]
        values = x[1:]  # (node labels, fix_var) or (node labels)
        inputs = [a]
        values_offset = 1
        if e is not None:
            inputs.append(e)
            values_offset += 1
        if i is not None:
            # iftrue_input_len = self.iftrue_input_len - 2
            # iffalse_input_len = self.iffalse_input_len - 2
            values_offset += 1
            _, _, count = tf.unique_with_counts(i)
            n_graphs = tf.shape(count)[0]
            partitioned_test_array = tf.TensorArray(dtype=tf.bool, size=n_graphs, infer_shape=False).split(test, count)
            # partitioned_test = tf.ragged.stack_dynamic_partitions(test, tf.cast(i, dtype=tf.int32), tf.size(count))
            # partitioned_values = tf.ragged.stack_dynamic_partitions(values[0], tf.cast(i, dtype=tf.int32), tf.size(count))
            partitioned_values_array = tf.TensorArray(dtype=values[0].dtype, size=n_graphs, infer_shape=False).split(values[0], count)
            n_node_features = values[0].shape[-1]
            # partitioned_idx = tf.ragged.stack_dynamic_partitions(i, tf.cast(i, dtype=tf.int32), tf.size(count))
            partitioned_idx_array = tf.TensorArray(dtype=tf.int32, size=n_graphs, infer_shape=False).split(tf.cast(i, dtype=tf.int32), count)
            # branch = tf.map_fn(lambda args: tf.reduce_all(args), partitioned_test, swap_memory=True, infer_shape=False,
            #                    fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.bool))
            output_array = tf.TensorArray(dtype=self.iftrue.outputs[0].dtype, size=n_graphs, infer_shape=False)

            if len(values) == 1:
                output = tf.while_loop(cond=lambda it, _: it < n_graphs,
                                       body=lambda it, out_arr: [it + 1, out_arr.write(it, tf.cond(tf.reduce_all(partitioned_test_array.read(it)),
                                                                                                   lambda: self.iftrue(
                                                                                                       [tf.ensure_shape(partitioned_values_array.read(it),
                                                                                                                        (None, n_node_features))] +
                                                                                                       inputs + [tf.ensure_shape(partitioned_idx_array.read(it),
                                                                                                                                 (None,))]),
                                                                                                   lambda: self.iffalse(
                                                                                                       [tf.ensure_shape(partitioned_values_array.read(it),
                                                                                                                        (None, n_node_features))] +
                                                                                                       inputs + [tf.ensure_shape(partitioned_idx_array.read(it),
                                                                                                                                 (None,))])))],
                                       loop_vars=[tf.constant(0, dtype=tf.int32), output_array]
                                       )[1]
                return tf.ensure_shape(output.concat(), self.iftrue.outputs[0].shape)
                # output = tf.cond(branch[0], lambda: self.iftrue([partitioned_values[0]] + inputs + [partitioned_idx[0]]),
                #                                          lambda: self.iffalse([partitioned_values[0]] + inputs + [partitioned_idx[0]]))
                # return output
                # '''
                # return tf.reshape(tf.map_fn(lambda args: tf.cond(tf.reduce_all(args[0]),
                #                                                  lambda: self.iftrue([args[1]] + inputs + [args[2]]),
                #                                                  lambda: self.iffalse([args[1]] + inputs + [args[2]])),
                #                             (tf.expand_dims(branch, -1), partitioned_values, partitioned_idx),
                #                             swap_memory=True,
                #                             infer_shape=False,
                #                             fn_output_signature=self.iftrue.outputs[0].type_spec),
                #                   [-1, self.iftrue.outputs[0].shape[-1]])
                # '''
            else:
                # partitioned_fix_var = tf.ragged.stack_dynamic_partitions(values[1], tf.cast(i, dtype=tf.int32), tf.size(count))
                partitioned_fix_var_array = tf.TensorArray(dtype=values[1].dtype, size=n_graphs, infer_shape=False).split(values[1], count)
                n_fix_var_features = values[1].shape[-1]
                output = tf.while_loop(cond=lambda it, _: it < n_graphs,
                                       body=lambda it, out_arr: [it + 1, out_arr.write(it, tf.cond(tf.reduce_all(partitioned_test_array.read(it)),
                                                                                                   lambda: self.iftrue([tf.ensure_shape(
                                                                                                       partitioned_values_array.read(it),
                                                                                                       (None, n_node_features)), tf.ensure_shape(
                                                                                                       partitioned_fix_var_array.read(it),
                                                                                                       (None, n_fix_var_features))][
                                                                                                                       :self.iftrue_input_len - values_offset] +
                                                                                                                       inputs + [tf.ensure_shape(
                                                                                                                               partitioned_idx_array.read(it),
                                                                                                                               (None,))]),
                                                                                                   lambda: self.iffalse([tf.ensure_shape(
                                                                                                       partitioned_values_array.read(it),
                                                                                                       (None, n_node_features)), tf.ensure_shape(
                                                                                                       partitioned_fix_var_array.read(it),
                                                                                                       (None, n_fix_var_features))][
                                                                                                                        :self.iffalse_input_len - values_offset]
                                                                                                                        + inputs + [tf.ensure_shape(
                                                                                                                                partitioned_idx_array.read(it),
                                                                                                                                (None,))])))],
                                       loop_vars=[tf.constant(0, dtype=tf.int32), output_array]
                                       )[1]
                return tf.ensure_shape(output.concat(), self.iftrue.outputs[0].shape)

                # return tf.reshape(tf.map_fn(lambda args: tf.cond(tf.reduce_all(args[0]),
                #                                                  lambda: self.iftrue([args[1], args[2]][:iftrue_input_len] + inputs + [args[3]]),
                #                                                  lambda: self.iffalse([args[1], args[2]][:iffalse_input_len] + inputs + [args[3]])),
                #                             (tf.expand_dims(branch, -1), partitioned_values, partitioned_fix_var, partitioned_idx),
                #                             swap_memory=True,
                #                             infer_shape=False,
                #                             fn_output_signature=self.iftrue.outputs[0].type_spec),
                #                   [-1, self.iftrue.outputs[0].shape[-1]])
        else:
            inputs_iftrue = values[:self.iftrue_input_len - values_offset] + inputs
            inputs_iffalse = values[:self.iffalse_input_len - values_offset] + inputs
            output = tf.cond(tf.reduce_all(test), lambda: self.iftrue(inputs_iftrue), lambda: self.iffalse(inputs_iffalse))
            return tf.ensure_shape(output, self.iftrue.outputs[0].shape)


class FixPoint(MGLayer):
    """Layer that applies a mG fixpoint expression.

    The layer runs a model M on an initial input X, then if the output Y = M(X) is equal to X, it returns Y, otherwise it runs M(Y) and repeats this process
    until M(Y) = Y, at which point it will return the fixed point Y.

    Attributes:
        gnn_x: The model whose fixpoint will be computed by this layer.
        comparator: The function that defines when the fixed point has been reached, i.e. when two guesses are equal.
        fixpoint_printer: The function that will be called during each iteration on the most recent guess. Used for debug prints.
        iters: Number of iterations that this layer executed.
    """

    def __init__(self, gnn_x: tf.keras.Model, tolerance: float | None, debug: bool = False):
        """Initializes the instance with the model for which to compute the fixed point, a tolerance value to test when two solutions are equal and whether to
        print debug information.

        Args:
            gnn_x: The model whose fixpoint will be computed by this layer.
            tolerance: Maximum difference between any two numeric values to consider them to have the same value.
            debug: If true, print debugging information such as intermediate outputs during fixpoint computation and total number of iterations for convergence.
                Prints to the logging console of TensorFlow.
        """
        super().__init__()
        self.gnn_x = gnn_x
        if tolerance is not None:
            self.comparator = lambda curr, prev: tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)), tolerance)
        else:
            self.comparator = lambda curr, prev: curr == prev
        self.fixpoint_printer = self.fixpoint_print_and_return if debug else lambda x: x
        self.iters = None

    @staticmethod
    def fixpoint_print_and_return(x: tf.Tensor) -> tf.Tensor:
        tf.print(x, output_stream=tf.compat.v1.logging.info)
        return x

    @staticmethod
    def standard_fixpoint(f: Callable, x0: list[tf.Tensor],
                          comparator: Callable[[tf.Tensor, tf.Tensor], bool],
                          fixpoint_printer: Callable[[tf.Tensor], tf.Tensor]) -> Any:
        """Fixed point solver that repeats the execution of a model until it converges.

        Args:
            f: The function for which to compute the fixed point.
            x0: The initial inputs for the fixed point computation.
            comparator: The function that defines when the fixed point has been reached, i.e. when two guesses are equal.
            fixpoint_printer: The function that will be called during each iteration on the most recent guess. Used for debug prints.

        Returns:
            The fixed point of the model computed starting from ``x0``, the value of the previous guess, and the number of iterations.
        """
        iter = tf.constant(1)
        x = f(x0)
        return tf.while_loop(
            cond=lambda curr, prev, k: tf.math.logical_not(tf.math.reduce_all(comparator(curr[0], prev[0]))),
            body=lambda curr, prev, k: [f(curr), fixpoint_printer(curr), k + 1],
            loop_vars=[x, x0, iter],
            shape_invariants=[[x[0].get_shape()], [x[0].get_shape()], iter.get_shape()]
        )

    def call(self, inputs: list[tf.Tensor], **kwargs) -> tf.Tensor:
        x, a, e, i = unpack_inputs(inputs, True)
        return self.propagate(x, a, e, i)

    def propagate(self, x: tf.Tensor, a: tf.SparseTensor, e: tf.Tensor | None = None, i: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        saved_args, x_o = x[:-1], x[-1:]
        additional_inputs = [a]
        if e is not None:
            additional_inputs.append(e)
        if i is not None:
            additional_inputs.append(i)

        # compute forward pass without tracking the gradient
        output = tf.nest.map_structure(tf.stop_gradient, self.standard_fixpoint(lambda y: [self.gnn_x(saved_args + y + additional_inputs)], x_o,
                                                                                lambda curr, prev: self.comparator(curr, prev), self.fixpoint_printer))
        self.iters = output[-1]
        tf.print('fixpoint iters: ', self.iters, output_stream=tf.compat.v1.logging.info)

        # compute forward pass with the gradient being tracked
        otp = output[0]
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            if IsTrainable(otp[0]):
                tape.watch(otp)
            output = self.gnn_x(saved_args + otp + additional_inputs)

        # computation of the gradient
        @tf.custom_gradient
        def fixgrad(fixpoint):
            def custom_grad(dy):
                def f(y): return tape.gradient(output, otp, y)[0] + dy

                x0 = dy
                x1 = f(x0)
                grad = tf.while_loop(lambda curr, prev: tf.math.logical_not(tf.math.reduce_all(tf.math.less_equal(tf.math.abs(tf.math.subtract(curr, prev)),
                                                                                                                  0.001))),
                                     lambda curr, prev: [f(curr), curr], [x1, x0])
                return grad[0]

            return tf.identity(fixpoint), custom_grad

        return fixgrad(output)


class Repeat(MGLayer):
    """Layer that applies a mG repeat expression.

    The layer runs a model M on an initial input X, Y = M(X), then runs it again Z = M(Y), and so on, for a fixed number of times.

    Attributes:
        gnn_x: The model to be run for a fixed number of times.
        iters: Number of iterations.
    """

    def __init__(self, gnn_x: tf.keras.Model, iters: int):
        """Initializes the instance with the model and the number of iterations to run.

        Args:
            gnn_x: The model to be run for a fixed number of times.
            iters: Number of iterations.
        """
        super().__init__(name='for_' + str(iters))
        self.gnn_x = gnn_x
        self.iters = iters

    def call(self, inputs: list[tf.Tensor], **kwargs) -> tf.Tensor:
        x, a, e, i = unpack_inputs(inputs, True)
        return self.propagate(x, a, e, i)

    def propagate(self, x: tf.Tensor, a: tf.SparseTensor, e: tf.Tensor | None = None, i: tf.Tensor | None = None, **kwargs) -> tf.Tensor:
        saved_args, x_o = x[:-1], x[-1:]
        additional_inputs = [a]
        if e is not None:
            additional_inputs.append(e)
        if i is not None:
            additional_inputs.append(i)
        return tf.while_loop(
            cond=lambda curr, it: tf.less(it, self.iters),
            body=lambda curr, it: [[self.gnn_x(saved_args + curr + additional_inputs)], it + 1],
            loop_vars=[x_o, 0],
            shape_invariants=[[x_o[0].get_shape()], None]
        )[0][0]
