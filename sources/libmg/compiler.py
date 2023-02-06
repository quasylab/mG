from __future__ import annotations

from typing import Callable, Type, Optional, Tuple
from lark import Lark, v_args
from lark.visitors import Interpreter
import tensorflow as tf
import time

from .functions import FunctionDict, Psi, Sigma, Phi
from .loaders import SingleGraphLoader, MultipleGraphLoader
from .normalizer import Normalizer, is_fixpoint
from .dummy_dataset import DummyDataset
from .grammar import mg_grammar
from .layers import PreImage, PostImage, FunctionApplication, Ite, FixPoint


class NodeConfig:
    def __init__(self, node_type: tf.DType, node_size: int):
        """
        Specifies the initial type of node labels, in the form of a feature vector

        :param node_type: The Tensorflow type of each element in the feature vector, e.g. tf.float32 or tf.uint8
        :param node_size: The length of the feature vector
        """
        self._type = node_type
        self._size = node_size

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return self._size


class EdgeConfig:
    def __init__(self, edge_type: tf.DType, edge_size: int):
        """
        Specifies the initial type of edge labels, in the form of a feature vector

        :param edge_type: The Tensorflow type of each element in the feature vector, e.g. tf.float32 or tf.uint8
        :param edge_size: The length of the feature vector
        """
        self._type = edge_type
        self._size = edge_size

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return self._size


class CompilationConfig:
    def __init__(self, node_config: NodeConfig, edge_config: EdgeConfig | None, matrix_type: tf.DType,
                 precision: dict[str, Optional[float]], disjoint_loader: bool):
        """
        Configures how to compile a mG formula into a Model. The constructor isn't meant to be used very often, it
        is recommended to use the static constructor methods to build this object.

        :param node_config: A NodeConfig object specifying the initial type of the node labels in the graph
        :param edge_config: An EdgeConfig object specifying the initial type of the node edges in the graph, if any
        :param matrix_type: A Tensorflow type specifying the type of the adjacency matrix entries, usually tf.uint8
        :param precision: Tolerance value for numeric fixpoint computations. Use `None` for exact computation.
        :param disjoint_loader: Set this to True if will be using a MultipleGraphLoader, False for a SingleGraphLoader
        """
        self.node_config = node_config
        self.edge_config = edge_config
        self._matrix_type = matrix_type
        self._precision = precision
        self.disjoint_loader = disjoint_loader

    @staticmethod
    def xa_config(node_config: NodeConfig, matrix_type: tf.DType,
                  precision: dict[str, Optional[float]]) -> CompilationConfig:
        """
        Creates a CompilationConfig object that expects no edge labels in the graph and the use of the
        SingleGraphLoader

        :param node_config: A NodeConfig object specifying the initial type of the node labels in the graph
        :param matrix_type: A Tensorflow type specifying the type of the adjacency matrix entries, usually tf.uint8
        :param precision: Tolerance value for numeric fixpoint computations. Use `None` for exact computation.
        :return: A CompilationConfig object
        """
        return CompilationConfig(node_config, None, matrix_type, precision, False)

    @staticmethod
    def xai_config(node_config: NodeConfig, matrix_type: tf.DType,
                   precision: dict[str, Optional[float]]) -> CompilationConfig:
        """
        Creates a CompilationConfig object that expects no edge labels in the graph and the use of the
        MultipleGraphLoader

        :param node_config: A NodeConfig object specifying the initial type of the node labels in the graph
        :param matrix_type: A Tensorflow type specifying the type of the adjacency matrix entries, usually tf.uint8
        :param precision: Tolerance value for numeric fixpoint computations. Use `None` for exact computation.
        :return: A CompilationConfig object
        """
        return CompilationConfig(node_config, None, matrix_type, precision, True)

    @staticmethod
    def xae_config(node_config: NodeConfig, edge_config: EdgeConfig, matrix_type: tf.DType,
                   precision: dict[str, Optional[float]]) -> CompilationConfig:
        """
        Creates a CompilationConfig object that expects edge labels in the graph and the use of the
        SingleGraphLoader

        :param node_config: A NodeConfig object specifying the initial type of the node labels in the graph
        :param edge_config: An EdgeConfig object specifying the initial type of the node edges in the graph, if any
        :param matrix_type: A Tensorflow type specifying the type of the adjacency matrix entries, usually tf.uint8
        :param precision: Tolerance value for numeric fixpoint computations. Use `None` for exact computation.
        :return: A CompilationConfig object
        """
        return CompilationConfig(node_config, edge_config, matrix_type, precision, False)

    @staticmethod
    def xaei_config(node_config: NodeConfig, edge_config: EdgeConfig, matrix_type: tf.DType,
                    precision: dict[str, Optional[float]]) -> CompilationConfig:
        """
        Creates a CompilationConfig object that expects edge labels in the graph and the use of the
        MultipleGraphLoader

        :param node_config: A NodeConfig object specifying the initial type of the node labels in the graph
        :param edge_config: An EdgeConfig object specifying the initial type of the node edges in the graph, if any
        :param matrix_type: A Tensorflow type specifying the type of the adjacency matrix entries, usually tf.uint8
        :param precision: Tolerance value for numeric fixpoint computations. Use `None` for exact computation.
        :return: A CompilationConfig object
        """
        return CompilationConfig(node_config, edge_config, matrix_type, precision, True)

    single_graph_no_edges_config = xa_config
    single_graph_with_edges_config = xae_config
    multiple_graphs_no_edges_config = xai_config
    multiple_graphs_with_edges_config = xaei_config

    @property
    def node_feature_type(self):
        return self.node_config.type

    @property
    def node_feature_size(self):
        return self.node_config.size

    @property
    def edge_feature_type(self):
        return self.edge_config.type if self.use_edges and self.edge_config is not None else None

    @property
    def edge_feature_size(self):
        return self.edge_config.size if self.use_edges and self.edge_config is not None else None

    @property
    def matrix_type(self):
        return self._matrix_type

    @property
    def use_edges(self):
        return self.edge_config is not None

    @property
    def precision(self):
        return self._precision

    @property
    def use_disjoint(self):
        return self.disjoint_loader

    @property
    def input_spec(self):
        specs = [tf.TensorSpec(shape=(None, self.node_feature_size), dtype=self.node_feature_type),
                 tf.SparseTensorSpec(shape=(None, None), dtype=self.matrix_type)]
        if self.use_edges:
            specs.append(tf.TensorSpec(shape=(None, self.edge_feature_size), dtype=self.edge_feature_type))
        if self.use_disjoint:
            specs.append(tf.TensorSpec(shape=(None,), dtype=tf.int64))
        return tuple(specs)

    @property
    def loader(self):
        if self.disjoint_loader:
            return MultipleGraphLoader
        else:
            return SingleGraphLoader


class IntermediateOutput:
    def __init__(self, name, x, a, e, i):
        self._name = name
        self._x = x
        self._a = a
        self._e = e
        self._i = i

    @property
    def x(self):
        return self._x

    @property
    def a(self):
        return self._a

    @property
    def e(self):
        return self._e

    @property
    def i(self):
        return self._i

    @property
    def name(self):
        return self._name

    @property
    def full_inputs(self):
        output = [self.x, self.a]
        if self.e is not None:
            output.append(self.e)
        if self.i is not None:
            output.append(self.i)
        return output

    @property
    def func_inputs(self):
        output = [self.x]
        if self.i is not None:
            output.append(self.i)
        return output

    @property
    def img_inputs(self):
        output = [self.x, self.a]
        if self.e is not None:
            output.append(self.e)
        return output

    @property
    def fixpoint_inputs(self):
        output = [self.a]
        if self.e is not None:
            output.append(self.e)
        if self.i is not None:
            output.append(self.i)
        return output

    def step(self, name, x):
        return IntermediateOutput(name, x, self.a, self.e, self.i)


class FixPointExpression:
    def __init__(self, name, inputs, outputs):
        # self._expr = lambda args: args[-1][0]  # the last element of args is a tuple (X, A, E, I), and we return X
        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self._args = []  # initially no arguments
        self._name = name  # initially it is just a variable name
        self._input_signature = inputs
        self._signature = outputs

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def signature(self):
        return self._signature

    @property
    def input_signature(self):
        return self._input_signature


class VarConfig:
    def __init__(self, dimension: int, dtype: tf.DType):
        self._dimension = dimension
        self._dtype = dtype
        self._signature = lambda: tf.keras.Input(shape=self.dimension, dtype=self.dtype)

    @property
    def signature(self):
        return self._signature()

    @property
    def dimension(self):
        return self._dimension

    @property
    def dtype(self):
        return self._dtype


# TODO: we made the signature fixed here, check if it breaks something!
class FixPointConfig(VarConfig):
    def __init__(self, dimension, dtype, value, precision=None):
        super().__init__(dimension, dtype)
        self._value = tf.constant(value, dtype=self.dtype)
        self._precision = precision
        self._constructor = lambda x: tf.fill(dims=(tf.shape(x)[0], self.dimension), value=self.value)
        self._signature = self._signature()

    @property
    def value(self):
        return self._value

    @property
    def constructor(self):
        return self._constructor

    @property
    def precision(self):
        return self._precision

    @property
    def signature(self):
        return self._signature

    @property
    def name(self):
        return str(self.dtype.name) + '[' + str(self.dimension) + ']=' + str(self.value.numpy())


# TODO: might have to be removed
class ModelWrapper:
    def __init__(self, model, name):
        self._model = model
        self._name = name

    @property
    def model(self):
        return self._model

    @property
    def name(self):
        return self._name


class MGFunction:
    def __init__(self, name, var_dictionary, body_tree):
        self.name = name
        self.var_dictionary = var_dictionary  # dictionary name: type
        self.body_tree = body_tree

    # type checking could be done here
    def get_args(self, arguments):  # arguments is a list of arguments for the function
        ordered_keys = list(self.var_dictionary)  # insertion order should have been retained
        matched_args = {}
        for i in range(len(arguments)):
            matched_args[ordered_keys[i]] = arguments[i]
        return matched_args


# make function for parallel op, only interested in the x and ignores a, e, i
def make_function_ops2(op_layer, layers, name):
    new_model_inputs = []
    to_concatenate = []
    new_args = []
    old_expr = None
    for layer in layers:
        if type(layer) is FixPointExpression:
            to_concatenate.append(layer.signature)
            old_expr = layer
        else:
            symbolic_layer = tf.keras.Input(type_spec=layer.x.type_spec)
            new_model_inputs.append(symbolic_layer)
            to_concatenate.append(symbolic_layer)
            new_args.append(layer.x)
        # here inputs must be symbolic inputs!!
    assert old_expr is not None
    # here go symbolic inputs
    new_expr = FixPointExpression('(' + name + ')', inputs=new_model_inputs + old_expr.input_signature,
                                  outputs=op_layer(to_concatenate))
    new_expr.args = new_args + old_expr.args  # here go actual saved inputs
    return new_expr


# asterisk in output: step whatever it returns. Used on base gnns, poolers, sequential
# no asterisk in output: only step x. Used on kop, parallel, lhd, rhd, mu, nu

class TreeToTF(Interpreter):
    supported_types = {'bool': tf.bool, 'int': tf.int32, 'float': tf.float32, 'uint8': tf.uint8, 'uint16': tf.uint16,
                       'uint32': tf.uint32, 'uint64': tf.uint64, 'int8': tf.int8, 'int16': tf.int16, 'int32': tf.int32,
                       'int64': tf.int64, 'float16': tf.float16, 'float32': tf.float32, 'float64': tf.float64,
                       'half': tf.float16, 'double': tf.float64}

    def __init__(self, psi_functions, sigma_functions, phi_functions, inputs, precision):
        super().__init__()
        self.psi_functions = psi_functions
        self.sigma_functions = sigma_functions
        self.phi_functions = phi_functions
        self.initial_inputs = inputs
        self.precision = precision
        # Initialization
        self.var = []
        self.var_type = []
        self.context = []
        self.layers = {}
        self.defined_functions = {}
        self.defined_variables = {}
        self.defined_local_variables = {}
        self.var_input = {}
        self.disable_saving_layers = False
        self.eval_if_clause = False
        self.inputs = self.initial_inputs

    def clone(self, inputs):
        new_interpreter = TreeToTF(self.psi_functions, self.sigma_functions, self.phi_functions, inputs, self.precision)
        new_interpreter.var = self.var
        new_interpreter.var_type = self.var_type
        new_interpreter.defined_functions = self.defined_functions
        new_interpreter.defined_variables = self.defined_variables
        new_interpreter.defined_local_variables = self.defined_local_variables
        new_interpreter.var_input = self.var_input
        return new_interpreter

    def add_layer(self, layer, ctx_name):
        if not self.disable_saving_layers:
            self.layers[ctx_name] = layer

    def get_layer(self, ctx_name):
        if ctx_name in self.layers:
            return self.layers[ctx_name]
        else:
            raise ValueError("No layer with name:", ctx_name)

    def undef_layer(self, ctx_name):
        return not (ctx_name in self.layers)

    def get_contextualized_name(self, name):
        if len(self.context) == 0:
            return '(' + name + ')'
        else:
            return '(' + self.head(self.context) + ';' + name + ')'

    def get_precision(self, typ):
        match typ:
            case 'float':
                typ = 'float32'
            case 'int':
                typ = 'int32'
            case 'half':
                typ = 'float16'
            case 'double':
                typ = 'float64'
        return self.precision.get(typ, None)

    @staticmethod
    def pop(stack):
        return stack.pop()

    @staticmethod
    def push(value, stack):
        stack.append(value)

    @staticmethod
    def head(stack):
        return stack[-1]

    @staticmethod
    def get_value(value_token, expected_type):
        if value_token == 'false' and expected_type.is_bool:
            return False
        elif value_token == 'true' and expected_type.is_bool:
            return True
        elif expected_type.is_integer:
            return int(value_token)
        elif expected_type.is_floating:
            return float(value_token)
        else:
            raise SyntaxError("Invalid value: " + value_token)

    def initialize(self):
        self.var = []
        self.var_type = []
        self.context = []
        self.layers = {}
        self.var_input = {}
        self.defined_functions = {}
        self.defined_variables = {}
        self.defined_local_variables = {}
        self.disable_saving_layers = False
        self.eval_if_clause = False
        self.inputs = self.initial_inputs

    @v_args(inline=True)
    def label(self, f):
        return str(f)

    @v_args(inline=True)
    def label_decl(self, var):
        return str(var)

    @v_args(inline=True)
    def type_decl(self, type_decl, dimension):
        return VarConfig(int(dimension), self.supported_types[type_decl])

    @v_args(inline=True)
    def atom_op(self, label):
        label = self.visit(label)
        ctx_name = self.get_contextualized_name(label)
        if len(self.var) > 0 and label == self.head(
                self.var) and not self.eval_if_clause:  # we are inside a fixpoint op and the label matches the fixpoint var
            var_signature = self.head(self.var_type).signature
            return FixPointExpression(str(label), inputs=[var_signature] + self.inputs.full_inputs[1:],
                                      outputs=var_signature)
        elif len(self.var) > 0 and label == self.head(self.var) and self.eval_if_clause:
            return self.inputs.step(self.head(self.var_type).name, self.head(self.var_type).signature)
        elif label in self.var_input:  # we are defining a function
            if isinstance(self.var_input[label], FixPointExpression):
                return self.var_input[label]
            else:
                return self.inputs.step(self.var_input[label].name, self.var_input[label].x)
        elif label in self.defined_local_variables:  # the label matches a local variable
            deferred_function = self.defined_local_variables[label]
            op_layer = self.visit(deferred_function.body_tree)
            ctx_name = op_layer.name
        elif label in self.defined_variables:  # the label matches a global variable
            deferred_function = self.defined_variables[label]
            op_layer = self.visit(deferred_function.body_tree)
            ctx_name = op_layer.name
        elif label in self.psi_functions:  # the label matches a psi function
            op_layer = FunctionApplication(self.psi_functions[label])
        else:
            raise SyntaxError('Undeclared variable: ' + label)
        # execution continues here only in the third, fourth or fifth cases of the if-elif-else
        if self.undef_layer(ctx_name):
            if label in self.defined_variables or label in self.defined_local_variables:
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, op_layer.x)
            else:
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, op_layer(self.inputs.func_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        else:
            return self.get_layer(ctx_name)

    def lhd(self, args):
        args = self.visit_children(args)
        if len(args) == 2:
            edge_function, agg_function = args
            name = '<' + edge_function + '| ' + agg_function
            lhd_layer = PreImage(self.sigma_functions[agg_function], self.phi_functions[edge_function])
        else:
            agg_function, = args
            name = '<| ' + agg_function
            lhd_layer = PreImage(self.sigma_functions[agg_function])
        ctx_name = self.get_contextualized_name(name)
        if self.undef_layer(ctx_name):
            # noinspection PyCallingNonCallable
            layer = self.inputs.step(ctx_name, lhd_layer(self.inputs.img_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.get_layer(ctx_name)

    def rhd(self, args):
        args = self.visit_children(args)
        if len(args) == 2:
            edge_function, agg_function = args
            name = '|' + edge_function + '> ' + agg_function
            rhd_layer = PostImage(self.sigma_functions[agg_function], self.phi_functions[edge_function])
        else:
            agg_function, = args
            name = '|> ' + agg_function
            rhd_layer = PostImage(self.sigma_functions[agg_function])
        ctx_name = self.get_contextualized_name(name)
        if self.undef_layer(ctx_name):
            # noinspection PyCallingNonCallable
            layer = self.inputs.step(ctx_name, rhd_layer(self.inputs.img_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.get_layer(ctx_name)

    @v_args(inline=True)
    def composition(self, phi, psi):
        current_inputs = self.inputs
        phi = self.visit(phi)
        self.push(phi.name, self.context)
        if type(phi) is FixPointExpression:  # phi is a fixpoint expression
            if is_fixpoint(psi, self.head(self.var)):  # both are fixpoint expressions
                # deal as in ite
                self.eval_if_clause = True
                test_input = tf.keras.Input(type_spec=phi.signature.type_spec)
                self.inputs = current_inputs.step(phi.name, test_input)
                psi = self.visit(psi)
                self.eval_if_clause = False
                psi_model = ModelWrapper(tf.keras.Model(
                    inputs=[test_input] + [self.head(self.var_type).signature] + self.initial_inputs.full_inputs[1:],
                    outputs=psi.x),
                                         name=psi.name)

                new_expr = FixPointExpression('(' + psi.name + ')', phi.input_signature, psi_model.model(
                    [phi.signature] + [self.head(self.var_type).signature] + self.initial_inputs.full_inputs[1:]))
                new_expr.args = phi.args
                self.inputs = current_inputs
                return new_expr
            else:  # only phi is a fixpoint expression
                self.disable_saving_layers = True
                self.inputs = current_inputs.step(phi.name, phi.signature)
                psi = self.visit(psi)
                self.pop(self.context)
                self.disable_saving_layers = False
                new_expr = FixPointExpression('(' + psi.name + ')', phi.input_signature, psi.x)
                new_expr.args = phi.args
                self.inputs = current_inputs
                return new_expr
        else:  # phi is not a fixpoint expression
            self.inputs = phi
            psi = self.visit(psi)
            self.pop(self.context)
            self.inputs = current_inputs
            return psi

    def parallel(self, args):
        args = self.visit_children(args)
        name = ' || '.join([arg.name for arg in args])
        has_var = False
        for layer in args:
            if type(layer) is FixPointExpression:
                has_var = True
                break
        if has_var:
            return make_function_ops2(tf.keras.layers.Concatenate(), args, name)
        else:
            ctx_name = self.get_contextualized_name(name)
            if self.undef_layer(ctx_name):  # TODO: maybe self.inputs is the same here?
                layer = args[0].step(ctx_name, tf.keras.layers.Concatenate()([arg.x for arg in args]))
                self.add_layer(layer, ctx_name)
                return layer
            return self.get_layer(ctx_name)

    def fun_def(self, tree):
        args = tree.children
        function_name = self.visit(args[0])
        var_input = {}
        for i in range(len(args[1:-2]) // 2):
            var_name = self.visit(args[1 + i * 2])
            var_type = self.visit(args[1 + i * 2 + 1])
            var_input[var_name] = var_type.signature
        function_tree = args[-2]  # we are not parsing the function right now
        deferred_function = MGFunction(function_name, var_input, function_tree)
        self.defined_functions[function_name] = deferred_function
        return self.visit(args[-1])

    def var_def(self, tree):
        args = tree.children
        for i in range(len(args[0:-1]) // 2):
            var_name = self.visit(args[i * 2])
            function_tree = args[i * 2 + 1]
            deferred_function = MGFunction(var_name, {}, function_tree)
            self.defined_variables[var_name] = deferred_function
        return self.visit(args[-1])

    def fun_call(self, tree):
        args = tree.children
        function_name = self.visit(args[0])
        arguments = [self.visit(arg) for arg in args[1:]]
        ctx_name = self.get_contextualized_name(
            function_name + '(' + ','.join([argument.name for argument in arguments]) + ')')

        deferred_function = self.defined_functions[function_name]
        matched_args = deferred_function.get_args(arguments)  # match args
        self.var_input |= matched_args  # add the deferred function vars to var input
        f_layer = self.visit(deferred_function.body_tree)  # now visit the function body
        for k in matched_args:  # eliminate the variables of this function from var_input
            self.var_input.pop(k)

        if isinstance(f_layer, FixPointExpression):
            return f_layer
        else:
            if self.undef_layer(ctx_name):
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, f_layer.x)
                self.add_layer(layer, ctx_name)
                return layer
            else:
                return self.get_layer(ctx_name)

    def local_var_expr(self, tree):
        args = tree.children
        local_vars = []
        for i in range(len(args[0:-1]) // 2):
            var_name = self.visit(args[i * 2])
            function_tree = args[i * 2 + 1]
            deferred_function = MGFunction(var_name, {}, function_tree)
            local_vars.append(var_name)
            self.defined_local_variables[var_name] = deferred_function
        expr = self.visit(tree.children[-1])
        for k in local_vars:  # eliminate the variables defined by this expression
            self.defined_local_variables.pop(k)
        return expr

    @v_args(inline=True)
    def ite(self, test, iftrue, iffalse):
        test = self.visit(test)
        # where do we have fixpoint variables?
        if len(self.var) > 0:
            fixpoint_idx = [isinstance(test, FixPointExpression), is_fixpoint(iftrue, self.head(self.var)),
                            is_fixpoint(iffalse, self.head(self.var))]
        else:
            fixpoint_idx = [False, False, False]
        # iftrue and iffalse are evaluated in the current context, so we make them from the initial inputs
        if fixpoint_idx[1] is True:
            self.eval_if_clause = True
            iftrue = self.visit(iftrue)
            self.eval_if_clause = False
            iftrue_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs[:1] + [
                self.head(self.var_type).signature] + self.initial_inputs.full_inputs[1:], outputs=iftrue.x),
                                        name=iftrue.name)
        else:
            iftrue = self.visit(iftrue)
            iftrue_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs, outputs=iftrue.x),
                                        name=iftrue.name)
        if fixpoint_idx[2] is True:
            self.eval_if_clause = True
            iffalse = self.visit(iffalse)
            self.eval_if_clause = False
            iffalse_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs[:1] + [
                self.head(self.var_type).signature] + self.initial_inputs.full_inputs[1:], outputs=iffalse.x),
                                         name=iffalse.name)
        else:
            iffalse = self.visit(iffalse)
            iffalse_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs, outputs=iffalse.x),
                                         name=iffalse.name)

        ctx_name = self.get_contextualized_name('if(' + test.name + ',' + iftrue.name + ',' + iffalse.name + ')')
        if (fixpoint_idx[0] and fixpoint_idx[1]) or (fixpoint_idx[0] and fixpoint_idx[2]) or all(fixpoint_idx):
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('(' + 'if(' + test.name + ',' + iftrue.name + ',' + iffalse.name + ')' + ')',
                                          inputs=self.initial_inputs.full_inputs[:1] + test.args + test.input_signature,
                                          outputs=ite_layer([test.signature] + self.initial_inputs.full_inputs[:1] + [
                                              self.head(self.var_type).signature] + test.input_signature[1:]))
            new_expr.args = self.initial_inputs.full_inputs[:1] + test.args
            return new_expr
        elif fixpoint_idx[1] or fixpoint_idx[2]:
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('(' + 'if(' + test.name + ',' + iftrue.name + ',' + iffalse.name + ')' + ')',
                                          inputs=[test.x] + self.initial_inputs.full_inputs[:1] + [
                                              self.head(self.var_type).signature] + self.initial_inputs.full_inputs[1:],
                                          outputs=ite_layer([test.x] + self.initial_inputs.full_inputs[:1] + [
                                              self.head(self.var_type).signature] + self.initial_inputs.full_inputs[
                                                                                    1:]))
            new_expr.args = [test.x] + self.initial_inputs.full_inputs[:1]  # here go actual saved inputs
            return new_expr
        elif fixpoint_idx[0]:
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('(' + 'if(' + test.name + ',' + iftrue.name + ',' + iffalse.name + ')' + ')',
                                          inputs=self.initial_inputs.full_inputs[:1] + test.args + test.input_signature,
                                          outputs=ite_layer([test.signature] + self.initial_inputs.full_inputs[
                                                                               :1] + test.input_signature[1:]))
            new_expr.args = self.initial_inputs.full_inputs[:1] + test.args  # here go actual saved inputs
            return new_expr
        else:
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            if self.undef_layer(ctx_name):
                # we pass the initial inputs
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, ite_layer([test.x] + self.initial_inputs.full_inputs))
                self.add_layer(layer, ctx_name)
                return layer
            else:
                return self.get_layer(ctx_name)

    @v_args(inline=True)
    def fix(self, variable_decl, type_decl, value, body):
        var_name = self.visit(variable_decl)
        self.push(var_name, self.var)
        type_decl = self.visit(type_decl)  # VarConfig object
        value = self.get_value(value, type_decl.dtype)
        precision = self.get_precision(type_decl.dtype.name)
        fixpoint_config = FixPointConfig(type_decl.dimension, type_decl.dtype, value, precision)
        self.push(fixpoint_config, self.var_type)
        nx = self.visit(body)
        if type(nx) is not FixPointExpression:
            raise SyntaxError('Invalid fixpoint expression')
        name = 'mu ' + self.pop(self.var) + ':' + fixpoint_config.name + ' . ' + nx.name
        type_config = self.pop(self.var_type)
        ctx_name = self.get_contextualized_name(name)
        lfp_layer = FixPoint(nx.model, type_config.constructor, type_config.precision)
        if self.undef_layer(ctx_name):
            # noinspection PyCallingNonCallable
            layer = self.inputs.step(ctx_name, lfp_layer(nx.args + self.inputs.fixpoint_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.get_layer(ctx_name)


class GNNCompiler:
    def __init__(self, psi_functions: FunctionDict[str, Psi | Callable[[str], Psi] | Type[Psi]],
                 sigma_functions: FunctionDict[str, Sigma | Callable[[str], Sigma] | Type[Sigma]],
                 phi_functions: FunctionDict[str, Phi | Callable[[str], Phi] | Type[Phi]], config: CompilationConfig):
        """
        A compiler for mG formulas. A formula is transformed into a Tensorflow model using the compile method.

        :param psi_functions: A dictionary of Psi functions, from any among the Psi, PsiLocal and PsiGlobal classes
        :param sigma_functions: A dictionary of Sigma functions
        :param phi_functions: A dictionary of Phi functions
        :param config: A CompilationConfig object to configure this GNNCompiler object
        """
        self.parser = Lark(mg_grammar, maybe_placeholders=False)
        self.macros = Normalizer(self.parser)
        self.model_inputs = [
            tf.keras.Input(shape=config.node_feature_size, name="INPUT_X", dtype=config.node_feature_type),
            tf.keras.Input(shape=(None,), sparse=True, name="INPUT_A", dtype=config.matrix_type)]
        intermediate_output_args = self.model_inputs + [None, None]
        if config.use_edges:
            self.model_inputs.append(
                tf.keras.Input(shape=config.edge_feature_size, name="INPUT_E", dtype=config.edge_feature_type))
            intermediate_output_args[2] = self.model_inputs[-1]
        if config.use_disjoint:
            self.model_inputs.append(tf.keras.Input(shape=(), name="INPUT_I", dtype=tf.int64))
            intermediate_output_args[3] = self.model_inputs[-1]
        self.model_input_spec = config.input_spec
        dummy_dataset = DummyDataset(config.node_feature_size, config.node_feature_type, config.matrix_type,
                                     config.edge_feature_size, config.edge_feature_type)
        self.dummy_loader = MultipleGraphLoader(dummy_dataset, node_level=True, batch_size=1, shuffle=False,
                                                epochs=1) if \
            config.use_disjoint else SingleGraphLoader(dummy_dataset, epochs=1)
        self.interpreter = TreeToTF(psi_functions, sigma_functions, phi_functions,
                                    IntermediateOutput("INPUT", *intermediate_output_args), config.precision)

    @staticmethod
    def graph_mode_constructor(model, input_spec, method):
        if method == 'call':
            @tf.function(input_signature=[input_spec])
            def serve(x):
                print('Tracing')
                return model(x, training=False)

            return serve
        else:
            model.run_eagerly = True
            predict_func = model.make_predict_function()
            model.predict_function = tf.function(predict_func, input_signature=[
                tf.data.IteratorSpec((input_spec,))])
            model.run_eagerly = False
            return model

    @staticmethod
    def dummy_run(model, dummy_loader, method):
        elapsed = 0.0
        if method == 'call':
            for x, y in dummy_loader.load():
                start = time.perf_counter()
                model(x)
                end = time.perf_counter()
                elapsed = end - start
                print("Dummy run completed in", elapsed, "s", sep=' ')
                break
        else:
            start = time.perf_counter()
            model.predict(dummy_loader.load(), steps=dummy_loader.steps_per_epoch)
            end = time.perf_counter()
            elapsed = end - start
            print("Dummy run completed in", elapsed, "s", sep=' ')
        return elapsed

    def compile(self, expr: str, loss: tf.keras.losses.Loss = None, verbose: bool = False,
                optimize: Optional[str] = None, return_compilation_time: bool = False) \
            -> tf.keras.Model | Callable | Tuple[tf.keras.Model, float] | Tuple[Callable, float]:
        """
        Compiles a mG formula `expr` into a Tensorflow Model.

        :param expr: A mG formula to evaluate
        :param loss: A Tensorflow loss function to be used if the model has to be trained. [Not yet implemented]
        :param verbose: Set this to True to print some debug information.
        :param optimize: Set this to "call" to optimize the model for being used with the "call" API, set this to
         "predict" to optimize the model for being used with the "predict" API, set this to None to leave the model as
         is. When set to "call" the model is transformed into a function.
        :param return_compilation_time: Set this to True to also return the time spent to compile the model. If optimize
        was set to None, compilation time is always 0.
        :return: A Tensorflow Model that is the mG evaluation of 'expr'.
        """
        self.interpreter.initialize()
        outputs = self.interpreter.visit(self.macros.visit(self.parser.parse(expr)))
        model = tf.keras.Model(inputs=self.model_inputs, outputs=outputs.x)
        model.compile(loss=loss)  # this is for training
        if verbose is True:
            model.summary()
            print('Optimized: ' + str(optimize))
        self.interpreter.initialize()
        compile_time = 0
        if optimize:
            model = GNNCompiler.graph_mode_constructor(model, self.model_input_spec, optimize)
            compile_time = GNNCompiler.dummy_run(model, self.dummy_loader, optimize)
        if return_compilation_time:
            return model, compile_time
        else:
            return model
