"""Defines the basic mG functions and a data structure to store them.

This module defines a dictionary for mG functions and the various classes that allows for their definition.

The module contains the following functions:
- ``make_uoperator(op, name)``
- ``make_boperator(op, name)``
- ``make_koperator(op, name)``

The module contains the following classes:
- ``FunctionDict``
- ``Function``
- ``Psi``
- ``PsiNonLocal``
- ``PsiLocal``
- ``PsiGlobal``
- ``Phi``
- ``Sigma``
- ``Constant``
- ``Pi``
- ``Operator``
"""

from __future__ import annotations

from functools import partial
import tensorflow as tf
from collections import UserDict
from typing import Callable, Any, TypeVar, Type


# class MultipleOp(Protocol):
#     def __call__(self, *x: tf.Tensor, i: tf.Tensor) -> tf.Tensor: ...
#
#
# class SingleOp(Protocol):
#     def __call__(self, *x: tf.Tensor) -> tf.Tensor: ...


class FunctionDict(UserDict):
    """Dictionary that maps mG labels to functions.

    This dictionary is instantiated by passing in a dictionary where keys are strings that are valid mG labels and values are either ``Function`` objects,
     ``Function`` subclasses, or functions in zero or one arguments that return ``Function`` objects.

    ``__setitem__``
    --------------------------

    - A ``Function`` is stored in the dictionary wrapped in a zero-argument lambda.
    - Any other zero-argument or one-argument function, or subclass of ``Function``, is stored as is.
    - An attempt to save any other object results in a ``TypeError`` exception.

    Therefore, all elements in the dictionary are either zero-argument or one-argument functions.

    ``__getitem__``
    --------------------------
    The given key is first parsed using the ``parse_key`` method. The key can either be of the form 'xyz[a]' or 'xyz':

    - A key of the form 'xyz' is parsed as is.
    - A key of the form 'xyz[a]' is split into the substring before the square brackets 'xyz' and the substring in the
      square brackets 'a'.
    We call the first substring ``true_key`` and the second substring ``arg``.

    Once the key is parsed, we obtain the item in the dictionary using ``true_key``. The dictionary does not return the
    corresponding value directly, but instead returns
     - the output of the application of ``arg`` to the value, if ``arg`` is present. It is assumed that the value is a one-argument function.
     - the output of calling the value with no argument, if ``arg`` is not present. It is assumed that the value is a zero-argument function.

    """

    @staticmethod
    def parse_key(key: str) -> tuple[str, str | None]:
        """Parses a key for the dictionary.

        Keys can have the form 'xyz[a]' or 'xyz'.

        - 'xyz[a]' is parsed to a tuple (xyz, a)
        - 'xyz' is parsed to a tuple (xyz, None)

        Args:
            key: The key to parse.

        Returns:
             A tuple of the form ``(true_key, arg)`` where ``true_key`` is the substring before the square brackets and ``arg`` is the substring inside the
             square brackets or ``None``.
        """
        tokens = key.split('[')
        true_key = tokens[0]
        arg = None if len(tokens) == 1 else tokens[1][: tokens[1].find(']')]
        return true_key, arg

    def __getitem__(self, key: str):
        true_key, arg = self.parse_key(key)
        if arg is None:
            return self.data[true_key]()
        else:
            return self.data[true_key](arg)

    def __setitem__(self, key: str, value: Callable):
        if isinstance(value, Function):
            self.data[key] = lambda: value
        elif (isinstance(value, type) and issubclass(value, Function)) or (hasattr(value, '__code__') and value.__code__.co_argcount < 2):
            self.data[key] = value
        else:
            raise TypeError("This dictionary only allows Function instances, subclasses, or functions with at most one argument.")

    def __contains__(self, key: object):
        true_key, _ = self.parse_key(str(key))
        return true_key in self.data

    def is_operator(self, key: str) -> bool:
        """Checks if the function corresponding to the given key is a mG operator.

        A function is a mG operator if it is a class that inherits from the ``Operator`` class.

        Args:
            key: The key corresponding to the function to check.

        Returns:
            ``True`` if the function is a mG operator, ``False`` otherwise.

        Raises:
            KeyError: The key is not in the dictionary.
        """
        true_key, _ = self.parse_key(str(key))
        if true_key in self.data:
            value = self.data[true_key]
            if isinstance(value, type) and issubclass(value, Operator):
                return True
            else:
                return False
        else:
            raise KeyError("Key not found", key)


T_A = TypeVar('T_A', bound='Function')


class Function(tf.keras.layers.Layer):
    """Base class for all mG functions.

    Provides some utility constructor methods to instantiate mG function classes. These utility methods allow the definition of mG functions that do not share
    data, such as e.g. weights.

    Attributes:
        f: The function applied by this object.
    """

    def __init__(self, f: Callable, name: str | None = None):
        """Initializes the instance with the function that this object will represent, and, if provided, a name.

        If a name is not provided, it will be used the name of the dynamic class of the instantiated object.

        Args:
            f: The function that this object represents.
            name: The name of the function.
        """
        super().__init__()
        self.f = f
        self._function_name = name or self.__class__.__name__

    @classmethod
    def make(cls: Type[T_A], name: str | None, f: Callable) -> Callable[[], T_A]:
        """Returns a zero-argument function that when called returns an instance of ``cls`` initialized with the provided function ``f`` and ``name``.

        The class ``cls`` is supposed to be a ``Function`` subclass. Calling this method on a suitable ``Function`` subclass
        creates a zero-argument lambda that returns an instance of such subclass. This way, whenever that function is used in a mG program, the dictionary
        automatically regenerates the instance. This is useful when the function has trainable weights, which are not supposed to be shared with other instances
        of the function.

        Examples:
            >>> Phi.make('Proj3', lambda i, e, j: j)
            <function Function.make.<locals>.<lambda> at 0x...>

        Args:
            name: The name of the function.
            f: The function that will be used to instantiate ``cls``.
        """
        if isinstance(f, tf.keras.layers.Layer):  # if f is a layer, regenerate from config to get new weights
            return lambda: cls(type(f).from_config(f.get_config()), name)  # type: ignore
        else:
            return lambda: cls(f, name)

    @classmethod
    def make_parametrized(cls: Type[T_A], name: str | None, f: Callable[[str], Callable] | Callable[..., Any]) -> Callable[[str], T_A]:
        """Returns a one-argument function that when called with the argument ``a``
         returns an instance of ``cls`` initialized with the result of the application of ``a`` to the function ``f`` and ``name``.

        The class ``cls`` is supposed to be a ``Function`` subclass. Calling this method on a suitable ``Function`` subclass
        creates a one-argument lambda that returns an instance of such subclass. This way, whenever that function is used in a mG program, the dictionary
        automatically regenerates the instance. This is useful when the function has trainable weights, which are not supposed to be shared with other instances
        of the function. The function ``f`` may have the form ``lambda x: lambda ... : ...`` or ``lambda x, ... : ...``. The one argument of the lambda returned
        by this function corresponds to the ``x`` argument of ``f``.

        Examples:
            >>> Phi.make_parametrized('Add', lambda y: lambda i, e, j: j + int(y) * e)
            <function Function.make_parametrized.<locals>.<lambda> at 0x...>
            >>> Phi.make_parametrized('Add', lambda y, i, e, j: j + int(y) * e)
            <function Function.make_parametrized.<locals>.<lambda> at 0x...>

        Args:
            name: The name of the function returned by ``f``.
            f: The function that when applied to some argument ``a`` returns the function that will be used to instantiate ``cls``.
        """
        # check if f has only a single argument e.g. lambda x: lambda y: foo(x)(y)
        if f.__code__.co_argcount == 1:
            return lambda a: cls(f(a), name + '_' + a if name is not None else None)
        else:  # e.g. lambda x, y: foo(x)(y)
            return lambda a: cls(partial(f, a), name + '_' + a if name is not None else None)

    @property
    def fname(self):
        """ The name of this function. This can be either the name provided during initialization, if it was provided, or the dynamic class name.
        """
        return self._function_name


class Psi(Function):
    """A psi function of the mG language.
    """

    def __call__(self, x: tuple[tf.Tensor, ...], i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError


class PsiNonLocal(Psi):
    """A psi function of the mG language.

    A non-local function applied on node labels $\\psi: T^* \\times T \\rightarrow U$. For single graph datasets, which use the
    ``SingleGraphLoader``, only the ``single_op`` parameter is necessary. For multiple graph datasets, using the
    ``MultipleGraphLoader``, only the ``multiple_op`` parameter is necessary. The ``multiple_op`` argument is a function which
    takes an additional parameter to distinguish which values in the first argument refer to which graph. For
    more information, refer to the disjoint data mode in the Spektral library `documentation <https://graphneural.network/data-modes/#disjoint-mode/>`_.

    Examples:
        >>> PsiNonLocal(single_op=lambda x: x + 1, multiple_op=lambda x, i: x + 1, name='Add1')
        <PsiNonLocal ...>

    Attributes:
        single_op: A function to be used in conjunction with a ``SingleGraphLoader``
        multiple_op: A function to be used in conjunction with a ``MultipleGraphLoader``
    """

    def __init__(self, single_op: Callable | None = None,
                 multiple_op: Callable | None = None,
                 name: str | None = None):
        """Initializes the instance with a function for a single graph and/or a function a multiple graph operation, and a name.

        Args:
            single_op: The function to be used in conjunction with a ``SingleGraphLoader``. The function must be compatible with TensorFlow's broadcasting
                rules. The function is expected to take in input a tensor X with shape ``(n_nodes, n_node_features)``, containing the labels of
                every node in the graph, and return a tensor of shape ``(n_nodes, n_new_node_features)``, containing the transformed labels.
                The function can use broadcasting to emulate the tuple (T*, T) in the definition of psi.
            multiple_op: The function to be used in conjunction with a ``MultipleGraphLoader``. The function must be compatible with TensorFlow's broadcasting
                rules. The function is expected to take in input a tensor X with shape ``(n_nodes, n_node_features)``, containing the labels of
                every node in the graph, and a tensor of graph indices of shape ``(n_nodes, 1)``, that mark to which graph every node belongs. The function is
                expected to return a tensor of shape ``(n_nodes, n_new_node_features)`` containing the transformed labels.
                The function can use broadcasting to emulate the tuple (T*, T) in the definition of psi.
            name: The name of the function.
        """
        if single_op is None:
            self.single_op = self.single_graph_op
        else:
            self.single_op = single_op  # type: ignore
        if multiple_op is None:
            self.multiple_op = self.multiple_graph_op
        else:
            self.multiple_op = multiple_op  # type: ignore
        super().__init__(self.single_op, name)

    def single_graph_op(self, *x: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def multiple_graph_op(self, *x: tf.Tensor, i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    @classmethod
    def make(cls, name: str | None, single_op: Callable | None = None, multiple_op: Callable | None = None) -> Callable[[], PsiNonLocal]:
        """Returns a zero-argument function that when called returns an instance of ``cls`` initialized with the provided functions ``single_op`` and/or
        ``multiple_op`` and ``name``.

        Calling this method on a ``PsiNonLocal`` class creates a zero-argument lambda that returns an instance of such subclass. This way, whenever that
        function is used in a mG program, the dictionary automatically regenerates the instance. This is useful when the function has trainable weights,
        which are not supposed to be shared with other instances of the function.

        Examples:
            >>> PsiNonLocal.make('Successor', lambda x: x + 1)
            <function PsiNonLocal.make.<locals>.<lambda> at 0x...>

        Args:
            name: The name of the function returned by ``single_op`` and ``multiple_op``.
            single_op: The function that will be used to instantiate ``cls`` as the ``single_op`` argument.
            multiple_op: The function that will be used to instantiate ``cls`` as the ``multiple_op`` argument.

        Raises:
            ValueError: Neither ``single_op`` nor ``multiple_op`` have been provided.
        """
        if single_op is None and multiple_op is None:
            raise ValueError("At least one function must be provided.")
        args = {}
        if single_op is not None:
            args['single_op' if cls is not PsiLocal else 'f'] = single_op
        if multiple_op is not None:
            args['multiple_op'] = multiple_op
        return lambda: cls(**{k: type(v).from_config(v.get_config()) if isinstance(v, tf.keras.layers.Layer) else v for k, v in args.items()}, name=name)

    @classmethod
    def make_parametrized(cls, name: str | None, single_op: Callable[[str], Callable] | Callable[..., Any] | None = None,
                          multiple_op: Callable[[str], Callable] | Callable[..., Any] | None = None) -> Callable[[str], PsiNonLocal]:
        """Returns a one-argument function that when called with the argument ``a`` returns an instance of ``cls`` initialized with the result of the
         application of ``a`` to the function ``single_op`` and/or ``multiple_op``, and ``name``.

        Calling this method on a ``PsiNonLocal`` class creates a one-argument lambda that returns an instance of such subclass. This way, whenever that
        function is used in a mG program, the dictionary automatically regenerates the instance. This is useful when the function has trainable weights,
        which are not supposed to be shared with other instances of the function. The functions ``single_op`` and ``multiple_op`` may have the form ``lambda
        x: lambda ... : ...`` or ``lambda x, ... : ...``. The one argument of the lambda returned by this function corresponds to the ``x`` argument of
        ``single_op`` and ``multiple_op``.

        Examples:
            >>> PsiNonLocal.make_parametrized(name='Add', single_op=lambda y: lambda x: x + y, multiple_op=lambda y: lambda x, i: x + y)
            <function PsiNonLocal.make_parametrized.<locals>.<lambda> at 0x...>
            >>> PsiNonLocal.make_parametrized( name='Add', single_op=lambda y, x: x + y, multiple_op=lambda y, x, i: x + y)
            <function PsiNonLocal.make_parametrized.<locals>.<lambda> at 0x...>

        Args:
            name: The name of the function returned by ``single_op`` and ``multiple_op``.
            single_op: The function that when applied to some argument ``a`` returns the function that will be used to instantiate ``cls``as the
                ``single_op`` argument.
            multiple_op: The function that when applied to some argument ``a`` returns the function that will be used to instantiate ``cls`` as the
                ``multiple_op`` argument.

        Raises:
            ValueError: Neither ``single_op`` nor ``multiple_op`` have been provided.
        """
        if single_op is None and multiple_op is None:
            raise ValueError("At least one function must be provided.")
        args = {}
        if single_op is not None:
            args['single_op' if cls is not PsiLocal else 'f'] = single_op
        if multiple_op is not None:
            args['multiple_op'] = multiple_op
        return lambda a: cls(**{k: v(a) if v.__code__.co_argcount == 1 else partial(v, a) for k, v in args.items()},
                             name=name + '_' + a if name is not None else None)

    def __call__(self, x: tuple[tf.Tensor, ...], i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        if i is not None:
            return self.multiple_op(*x, i=i)
        else:
            return self.single_op(*x)


class PsiLocal(Psi):
    """A psi function of the mG language that only applies a local transformation of node labels.

    A local transformation of node labels $\\psi: T \\rightarrow U$.

    Examples:
        >>> PsiLocal(lambda x: x + 1, name='Add1')
        <PsiLocal ...>
    """

    def __init__(self, f: Callable | None = None, name: str | None = None):
        """Initializes the instance with a function and a name.

        Args:
            f: The function that this object will run when called. The function must be compatible with TensorFlow's broadcasting
                rules. The function is expected to take in input a tensor X with shape ``(n_nodes, n_node_features)``, containing the labels of
                every node in the graph, and return a tensor of shape ``(n_nodes, n_new_node_features)``, containing the transformed labels.
                The function is not supposed to use global information, but this is not enforced.
            name: The name of the function.
        """
        if f is None:
            f = self.func
        super().__init__(f, name)

    def func(self, *x: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def __call__(self, x: tuple[tf.Tensor], i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        return self.f(*x)


class PsiGlobal(PsiNonLocal):
    """A psi function of the mG language that only applies a global transformation of node labels.

    A global transformation (i.e. pooling) of node labels $\\psi: T^* \\rightarrow U$. For single graph datasets, which use the
    ``SingleGraphLoader``, only the ``single_op`` parameter is necessary. For multiple graph datasets, using the
    ``MultipleGraphLoader``, only the ``multiple_op`` parameter is necessary. The ``multiple_op`` argument is a function which
    takes an additional parameter to distinguish which values in the first argument refer to which graph. For
    more information, refer to the disjoint data mode in the Spektral library `documentation <https://graphneural.network/data-modes/#disjoint-mode/>`_.

    Examples:
        >>> PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=True), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='SumPooling')
        <PsiGlobal ...>
    """

    def __init__(self, single_op: Callable | None = None,
                 multiple_op: Callable | None = None, name: str | None = None):
        """Initializes the instance with a function for a single graph and/or a function a multiple graph operation, and a name.

        Args:
            single_op: The function to be used in conjunction with a ``SingleGraphLoader``.
                The function is expected to take in input a tensor X with shape ``(n_nodes, n_node_features)``, containing the labels of
                every node in the graph, and return a tensor of shape ``(n_pooled_features, )``, containing the pooled output. This output is then broadcast to
                label every node.
            multiple_op: The function to be used in conjunction with a ``MultipleGraphLoader``.
                The function is expected to take in input a tensor with shape ``(n_nodes, n_node_features)``, containing the labels of
                every node in the graph, and a tensor of graph indices of shape ``(n_nodes, 1)``, that mark to which graph every node belongs. The function is
                expected to return a tensor of shape ``(n_graphs, n_pooled_features)`` containing the pooled outputs for each distinct graph index. This output
                is then broadcast to label every node of each graph with the pooled features for that graph.
            name: The name of the function.
        """
        super().__init__(single_op=single_op, multiple_op=multiple_op, name=name)

    def single_graph_op(self, *x: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def multiple_graph_op(self, *x: tf.Tensor, i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def __call__(self, x: tuple[tf.Tensor, ...], i: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        if i is not None:
            output = self.multiple_op(*x, i=i)
            _, _, count = tf.unique_with_counts(i)
            return tf.repeat(output, count, axis=0)
        else:
            output = tf.expand_dims(self.single_op(*x), axis=0)
            return tf.repeat(output, tf.shape(x[0])[0], axis=0)


class Phi(Function):
    """A phi function of the mG language.

    A function $\\varphi: T \\times U \\times T \\rightarrow V$ to compute the message sent by a node i to a node j through edge e.

    Examples:
        >>> Phi(lambda i, e, j: i * e, name='EdgeProd')
        <Phi ...>
    """

    def __init__(self, f: Callable[[tuple[tf.Tensor, ...], tf.Tensor, tuple[tf.Tensor, ...]], tf.Tensor | tuple[tf.Tensor, ...]] | None = None,
                 name: str | None = None):
        """Initializes the instance with a function and a name.

        Args:
            f: The function that this object will run when called. The function must be compatible with TensorFlow's broadcasting
                rules. The function is expected to take in input a tensor X1 with shape ``(n_edges, n_node_features)``, containing the labels of
                all nodes sending a message, a tensor E with shape ``(n_edges, n_edge_features)``, containing the labels of all edges in the graph, and a tensor
                X2, containing the labels of all nodes receiving a message. The function is expected to return a tensor with shape
                ``(n_edges, n_message_features)``, containing the messages to be sent to the destination nodes.
            name: The name of the function.
        """
        if f is None:
            f = self.func
        super().__init__(f, name)

    def func(self, src: tuple[tf.Tensor, ...], e: tf.Tensor, tgt: tuple[tf.Tensor, ...],) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def __call__(self, src: tuple[tf.Tensor, ...], e: tf.Tensor, tgt: tuple[tf.Tensor, ...]) -> tf.Tensor | tuple[tf.Tensor, ...]:
        return self.f(*src, e, *tgt)


class Sigma(Function):
    """A sigma function of the mG language.

    A function $\\sigma: T^* \\times U \\rightarrow V$ to aggregate the messages sent to a node, including the current label of the node.

    Examples:
        >>> Sigma(lambda m, i, n, x: tf.math.segment_max(m, i), name='Max')
        <Sigma ...>
    """

    def __init__(self, f: Callable[[tuple[tf.Tensor, ...], tf.Tensor, int, tuple[tf.Tensor, ...]], tf.Tensor | tuple[tf.Tensor, ...]] | None = None,
                 name: str | None = None):
        """Initializes the instance with a function and a name.

        Args:
            f: The function that this object will run when called. The function must be compatible with TensorFlow's broadcasting
                rules. The function is expected to take in input a tensor M with shape ``(n_edges, n_message_features)``, containing the generated messages,
                a tensor IDX with shape ``(n_edges,)``, containing the ids of the node each message is being sent to, the total number of the nodes involved,
                and a tensor X, containing the current node labels. The function is expected to return a tensor with shape
                ``(n_nodes, n_new_node_features)``, containing the new node labels.
            name: The name of the function.
        """
        if f is None:
            f = self.func
        super().__init__(f, name)

    def func(self, m: tuple[tf.Tensor, ...], i: tf.Tensor, n: int, x: tuple[tf.Tensor, ...]) -> tf.Tensor | tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def __call__(self, m: tuple[tf.Tensor, ...], i: tf.Tensor, n: int, x: tuple[tf.Tensor, ...] | None) -> tf.Tensor | tuple[tf.Tensor, ...]:
        assert x is not None
        return self.f(*m, i, n, *x)


class Constant(PsiLocal):
    """A constant psi function of the mG language.

    A constant function $\\psi: T \\rightarrow U$ that maps every node label to a constant value.

    Examples:
        >>> Constant(tf.constant(False), name='False')
        <Constant ...>
    """

    def __init__(self, v: tf.Tensor, name: str | None = None):
        """Initializes the instance with a function and a name.

        Args:
            v: A scalar or tensor value that identifies the constant function.
            name: The name of the function.
        """
        if tf.reduce_all(tf.equal(tf.cast(v, dtype=tf.float32), 1.0)):
            def f(x):
                return tf.ones((tf.shape(x)[0], tf.size(v)), dtype=v.dtype)
        elif tf.reduce_all(tf.equal(tf.cast(v, dtype=tf.float32), 0.0)):
            def f(x):
                return tf.zeros((tf.shape(x)[0], tf.size(v)), dtype=v.dtype)
        elif tf.size(v) == 1:
            def f(x):
                return tf.fill((tf.shape(x)[0], tf.size(v)), value=v)
        else:
            def f(x):
                return tf.tile([v], [tf.shape(x)[0], 1])
        super().__init__(f, name)


class Pi(PsiLocal):
    """A projection psi function of the mG language.

    A projection function $\\psi: T^n \\rightarrow T^m$ that maps every node label to a projection of itself.

    Examples:
        >>> Pi(0, 2, name='FirstTwo')
        <Pi ...>
    """

    def __init__(self, i: int, j: int | None = None, name: str | None = None):
        """Initializes the instance with the projection indexes and a name.

        Args:
            i: 0-based index, start position of the projection function, inclusive.
            j: end position of the projection function, exclusive. Defaults to i + 1.
            name: The name of the function.

        Raises:
            ValueError: start and end index are equal.
        """
        j = i + 1 if j is None else j
        if i == j:
            raise ValueError("Start index and end index cannot be equal.")

        def f(x): return x[:, i:j]

        super().__init__(f, name)


class Operator(PsiLocal):
    """An operator psi function of the mG language.

    An operator function can automatically infer its operands from the input node labels without having to slice them, provided some pre-conditions apply.
    See the documentation for ``make_uoperator``, ``make_boperator``, and ``make_koperator`` to learn about these pre-conditions. This class is not meant to
    be instantiated directly, but to be used specifically through the aforementioned methods.

    Attributes:
        k: Number of the operands for the operator.
    """

    def __init__(self, k: str):
        """Initializes the instance with the number of operands.

        Args:
            k: The number of operands that this operator expects. This number is a string because it will be received during compilation of a mG program by
                the ``FunctionDict``.
        """
        self.k = int(k)
        super().__init__()


def make_uoperator(op: Callable[[tf.Tensor], tf.Tensor], name: str | None = None) -> type:
    """Returns a unary operator psi function.

    A unary operator is equivalent to a ``PsiLocal``.

    Args:
        op: The unary operator function. The function must be compatible with TensorFlow's broadcasting
            rules. The function is expected to take in input a tensor X with shape ``(n_nodes, n_node_features)``, containing the labels of
            every node in the graph, and return a tensor of shape ``(n_nodes, n_new_node_features)``, containing the transformed labels.
        name: The name of the unary operator.

    Returns:
        A subclass of Operator that implements the operator.
    """

    def func(self, x: tf.Tensor) -> tf.Tensor:
        return op(x)

    return type(name or "UOperator", (Operator,), {'func': func})


def make_boperator(op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], name: str | None = None) -> type:
    """Returns a binary operator psi function.

    A binary operator is a binary local transformation of node labels, psi: (T, T) -> U.

    Args:
        op: The binary operator function. The function must be compatible with TensorFlow's broadcasting
            rules. The function is expected to take in input two tensors X1, X2 with shape ``(n_nodes, n_node_features/2)``. The first tensor contains for every
            node the first half of the node features, while the second tensor contains the second half. The operator returns a tensor of shape
            ``(n_nodes, n_new_node_features)``, containing the transformed labels.
        name: The name of the binary operator.

    Returns:
        A subclass of Operator that implements the operator.
    """

    def func(self, *x: list[tf.Tensor]) -> tf.Tensor:
        return op(x[0], x[1])

    return type(name or "BOperator", (Operator,), {'func': func})


def make_koperator(op: Callable[..., tf.Tensor], name: str | None = None) -> type:
    """Returns a k-ary operator psi function.

    A k-ary operator is a k-ary local transformation of node labels, psi: (T^k) -> U.

    Args:
        op: The k-ary operator function. The function must be compatible with TensorFlow's broadcasting
            rules. The function is expected to take in input k tensors X1, X2, ..., Xk with shape ``(n_nodes, n_node_features/k)``. The first tensor contains
            for every node the first k of the node features, the second tensor contains the next k features, and so on. The operator returns a tensor of shape
            ``(n_nodes, n_new_node_features)``, containing the transformed labels.
        name: The name of the k-ary operator.

    Returns:
        A subclass of Operator that implements the operator.
    """

    def func(self, *x: tf.Tensor) -> tf.Tensor:
        return op(*x)

    return type(name or "KOperator", (Operator,), {'func': func})
