from __future__ import annotations

from typing import Callable, Type, Optional, Tuple

from bidict import bidict
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

    def step(self, name, x, free_vars):  # add a check here for free variables
        if self.x.ref() in free_vars:
            free_vars[x.ref()] = free_vars.pop(self.x.ref())
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


class FixPointConfig:
    def __init__(self, dimension, dtype, name):
        self._signature = tf.keras.Input(shape=dimension, dtype=dtype)
        self._initial_var_name = name

    @property
    def signature(self):
        return self._signature

    @property
    def name(self):
        return self._initial_var_name


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
    def __init__(self, name, var_list, body_tree):
        self.name = name
        self.var_list = var_list  # dictionary name: type
        self.body_tree = body_tree

    # type checking could be done here
    def get_args(self, arguments):  # arguments is a list of arguments for the function
        ordered_keys = self.var_list  # insertion order should have been retained
        matched_args = {}
        for i in range(len(arguments)):
            matched_args[ordered_keys[i]] = arguments[i]
        return matched_args


def analyze_fix_var(var):
    if type(var) is FixPointExpression:
        return var.signature.shape[1], var.signature.dtype
    else:
        return var.x.shape[1], var.x.dtype


# make function for parallel op, only interested in the x and ignores a, e, i
def make_function_ops2(op_layer, layers, name):
    new_model_inputs = []
    to_concatenate = []
    new_args = []
    fixpoints_input_signatures = {}  # type: dict[tf.python.util.object_identity.Reference, None]
    fixpoints_saved_args = {}  # type: dict[tf.python.util.object_identity.Reference, None]
    for layer in layers:
        if type(layer) is FixPointExpression:
            to_concatenate.append(layer.signature)
            for input_signature in layer.input_signature:
                fixpoints_input_signatures.pop(input_signature.ref(), None)
                fixpoints_input_signatures[input_signature.ref()] = None
            for saved_arg in layer.args:
                fixpoints_saved_args.pop(saved_arg.ref(), None)
                fixpoints_saved_args[saved_arg.ref()] = None
        else:
            symbolic_layer = tf.keras.Input(type_spec=layer.x.type_spec)
            new_model_inputs.append(symbolic_layer)
            to_concatenate.append(symbolic_layer)
            new_args.append(layer.x)
        # here inputs must be symbolic inputs!!
    # assert old_expr is not None
    # here go symbolic inputs
    old_expr_input_signatures = [input_sig.deref() for input_sig in fixpoints_input_signatures.keys()]
    old_expr_args = [saved_arg.deref() for saved_arg in fixpoints_saved_args.keys()]
    new_expr = FixPointExpression('(' + name + ')', inputs=new_model_inputs + old_expr_input_signatures,
                                  outputs=op_layer(to_concatenate))
    new_expr.args = new_args + old_expr_args  # here go actual saved inputs
    return new_expr


# asterisk in output: step whatever it returns. Used on base gnns, poolers, sequential
# no asterisk in output: only step x. Used on kop, parallel, lhd, rhd, mu, nu

class TreeToTF(Interpreter):
    supported_types = {'bool': tf.bool, 'int': tf.int32, 'float': tf.float32, 'uint8': tf.uint8, 'uint16': tf.uint16,
                       'uint32': tf.uint32, 'uint64': tf.uint64, 'int8': tf.int8, 'int16': tf.int16, 'int32': tf.int32,
                       'int64': tf.int64, 'float16': tf.float16, 'float32': tf.float32, 'float64': tf.float64,
                       'half': tf.float16, 'double': tf.float64}

    def __init__(self, psi_functions, sigma_functions, phi_functions, inputs, precision, parser):
        super().__init__()
        self.psi_functions = psi_functions
        self.sigma_functions = sigma_functions
        self.phi_functions = phi_functions
        self.initial_inputs = inputs
        self.precision = precision
        self.parser = parser
        # Initialization
        self.fix_var = {}
        self.free_fix_var = bidict({})
        self.context = []
        self.layers = {}
        self.defined_functions = {}
        self.defined_variables = {}
        self.defined_local_variables = {}
        self.var_input = {}
        self.disable_saving_layers = False
        self.eval_if_clause = False
        self.inputs = self.initial_inputs

    def add_layer(self, layer, ctx_name):
        if not self.disable_saving_layers:
            self.layers[hash(self.parser.parse(ctx_name))] = layer

    def get_layer(self, ctx_name):
        if hash(self.parser.parse(ctx_name)) in self.layers:
            return self.layers[hash(self.parser.parse(ctx_name))]
        else:
            raise ValueError("No layer with name:", ctx_name)

    def undef_layer(self, ctx_name):
        return not (hash(self.parser.parse(ctx_name)) in self.layers)

    def get_contextualized_name(self, name):
        if len(self.context) == 0:
            return '(' + name + ')'
        else:
            return '(' + self.context[-1] + ';' + name + ')'

    def get_precision(self, typ):
        match typ:
            case 'float32':
                return self.precision.get(typ, self.precision.get('float', None))
            case 'int32':
                return self.precision.get(typ, self.precision.get('int', None))
            case 'float16':
                return self.precision.get(typ, self.precision.get('half', None))
            case 'float64':
                return self.precision.get(typ, self.precision.get('double', None))
            case _:
                return self.precision.get(typ, None)

    def current_fix_var(self):
        return next(reversed(self.fix_var))

    def current_fix_var_config(self):
        return self.fix_var[next(reversed(self.fix_var))]

    def initialize(self):
        self.fix_var = {}
        self.free_fix_var = bidict({})
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
    def atom_op(self, label):
        label = self.visit(label)
        ctx_name = self.get_contextualized_name(label)
        if len(self.fix_var) > 0 and label == self.current_fix_var() and not self.eval_if_clause:
            # we are inside a fixpoint op and the label matches the fixpoint var
            var_signature = self.current_fix_var_config().signature
            return FixPointExpression(str(label), inputs=[var_signature] + self.inputs.full_inputs[1:],
                                      outputs=var_signature)
        elif len(self.fix_var) > 0 and label == self.current_fix_var() and self.eval_if_clause:
            return self.inputs.step(self.current_fix_var_config().name, self.current_fix_var_config().signature, self.free_fix_var)
        elif label in self.var_input:  # we are defining a function
            if isinstance(self.var_input[label], FixPointExpression):
                return self.var_input[label]
            else:
                return self.inputs.step(self.var_input[label].name, self.var_input[label].x, self.free_fix_var)
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
        elif label in self.fix_var:  # the label is a fix_var, but not the current one
            io = self.inputs.step(label, self.fix_var[label].signature, self.free_fix_var)
            self.free_fix_var[io.x.ref()] = label
            return io
            # return self.inputs.step(label, self.fix_var[label].signature)  # must be treated as if it was a psi function
        else:
            raise SyntaxError('Undeclared variable: ' + label)
        # execution continues here only in the third, fourth or fifth cases of the if-elif-else
        if self.undef_layer(ctx_name):
            if label in self.defined_variables or label in self.defined_local_variables:
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, op_layer.x, self.free_fix_var)
            else:
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, op_layer(self.inputs.func_inputs), self.free_fix_var)
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
            layer = self.inputs.step(ctx_name, lhd_layer(self.inputs.img_inputs), self.free_fix_var)
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
            layer = self.inputs.step(ctx_name, rhd_layer(self.inputs.img_inputs), self.free_fix_var)
            self.add_layer(layer, ctx_name)
            return layer
        return self.get_layer(ctx_name)

    @v_args(inline=True)
    def composition(self, phi, psi):
        current_inputs = self.inputs
        phi = self.visit(phi)
        self.context.append(phi.name)
        if type(phi) is FixPointExpression:  # phi is a fixpoint expression
            if is_fixpoint(psi, self.current_fix_var()):  # both are fixpoint expressions
                # deal as in ite
                self.eval_if_clause = True
                test_input = tf.keras.Input(type_spec=phi.signature.type_spec)
                self.inputs = current_inputs.step(phi.name, test_input, self.free_fix_var)
                psi = self.visit(psi)
                self.eval_if_clause = False
                psi_model = ModelWrapper(tf.keras.Model(
                    inputs=[test_input] + [self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[1:],
                    outputs=psi.x),
                                         name=psi.name)

                new_expr = FixPointExpression('(' + psi.name + ')', phi.input_signature, psi_model.model(
                    [phi.signature] + [self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[1:]))
                new_expr.args = phi.args
                self.inputs = current_inputs
                return new_expr
            else:  # only phi is a fixpoint expression
                self.disable_saving_layers = True
                self.inputs = current_inputs.step(phi.name, phi.signature, self.free_fix_var)
                psi = self.visit(psi)
                self.context.pop()
                self.disable_saving_layers = False
                new_expr = FixPointExpression('(' + psi.name + ')', phi.input_signature, psi.x)
                new_expr.args = phi.args
                self.inputs = current_inputs
                return new_expr
        else:  # phi is not a fixpoint expression
            self.inputs = phi
            psi = self.visit(psi)
            self.context.pop()
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
            if self.undef_layer(ctx_name):
                layer = self.inputs.step(ctx_name, tf.keras.layers.Concatenate()([arg.x for arg in args]), self.free_fix_var)
                self.add_layer(layer, ctx_name)
                return layer
            return self.get_layer(ctx_name)

    def fun_def(self, tree):
        args = tree.children
        function_name = self.visit(args[0])
        var_input = []
        for i in range(len(args[1:-2])):
            var_name = self.visit(args[1 + i])
            var_input.append(var_name)
        function_tree = args[-2]  # we are not parsing the function right now
        deferred_function = MGFunction(function_name, var_input, function_tree)
        self.defined_functions[function_name] = deferred_function
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
                layer = self.inputs.step(ctx_name, f_layer.x, self.free_fix_var)
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
        if len(self.fix_var) > 0:
            fixpoint_idx = [isinstance(test, FixPointExpression), is_fixpoint(iftrue, self.current_fix_var()),
                            is_fixpoint(iffalse, self.current_fix_var())]
        else:
            fixpoint_idx = [False, False, False]
        # iftrue and iffalse are evaluated in the current context, so we make them from the initial inputs
        if fixpoint_idx[1] is True:
            self.eval_if_clause = True
            iftrue = self.visit(iftrue)
            self.eval_if_clause = False
            iftrue_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs[:1] + [
                self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[1:], outputs=iftrue.x),
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
                self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[1:], outputs=iffalse.x),
                                         name=iffalse.name)
        else:
            iffalse = self.visit(iffalse)
            iffalse_model = ModelWrapper(tf.keras.Model(inputs=self.initial_inputs.full_inputs, outputs=iffalse.x),
                                         name=iffalse.name)

        ctx_name = self.get_contextualized_name('if ' + test.name + ' then ' + iftrue.name + ' else ' + iffalse.name)
        if (fixpoint_idx[0] and fixpoint_idx[1]) or (fixpoint_idx[0] and fixpoint_idx[2]) or all(fixpoint_idx):
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('if ' + test.name + ' then ' + iftrue.name + ' else ' + iffalse.name,
                                          inputs=self.initial_inputs.full_inputs[:1] + test.args + test.input_signature,
                                          outputs=ite_layer([test.signature] + self.initial_inputs.full_inputs[:1] + [
                                              self.current_fix_var_config().signature] + test.input_signature[1:]))
            new_expr.args = self.initial_inputs.full_inputs[:1] + test.args
            return new_expr
        elif fixpoint_idx[1] or fixpoint_idx[2]:
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('if ' + test.name + ' then ' + iftrue.name + ' else ' + iffalse.name,
                                          inputs=[test.x] + self.initial_inputs.full_inputs[:1] + [
                                              self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[1:],
                                          outputs=ite_layer([test.x] + self.initial_inputs.full_inputs[:1] + [
                                              self.current_fix_var_config().signature] + self.initial_inputs.full_inputs[
                                                                                    1:]))
            new_expr.args = [test.x] + self.initial_inputs.full_inputs[:1]  # here go actual saved inputs
            return new_expr
        elif fixpoint_idx[0]:
            ite_layer = Ite(iftrue_model.model, iffalse_model.model)
            # noinspection PyCallingNonCallable
            new_expr = FixPointExpression('if ' + test.name + ' then ' + iftrue.name + ' else ' + iffalse.name,
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
                layer = self.inputs.step(ctx_name, ite_layer([test.x] + self.initial_inputs.full_inputs), self.free_fix_var)
                self.add_layer(layer, ctx_name)
                return layer
            else:
                return self.get_layer(ctx_name)

    @v_args(inline=True)
    def fix(self, variable_decl, initial_var_gnn, body):
        var_name = self.visit(variable_decl)
        initial_gnn_var = self.visit(initial_var_gnn)
        initial_gnn_var_dimension, initial_gnn_var_type = analyze_fix_var(initial_gnn_var)
        precision = self.get_precision(initial_gnn_var_type.name)
        fixpoint_config = FixPointConfig(initial_gnn_var_dimension, initial_gnn_var_type, initial_gnn_var.name)
        self.fix_var[var_name] = fixpoint_config
        nx = self.visit(body)
        if type(nx) is not FixPointExpression:
            raise ValueError('Invalid fixpoint expression')
        name = 'fix ' + var_name + ' = ' + fixpoint_config.name + ' in ' + nx.name
        self.fix_var.pop(var_name)
        self.free_fix_var.inverse.pop(var_name, None)
        lfp_layer = FixPoint(nx.model, precision)
        if len(self.free_fix_var) == 0 and type(initial_gnn_var) is not FixPointExpression:
            ctx_name = self.get_contextualized_name(name)
            if self.undef_layer(ctx_name):
                # noinspection PyCallingNonCallable
                layer = self.inputs.step(ctx_name, lfp_layer(nx.args + [initial_gnn_var.x] + self.inputs.fixpoint_inputs), self.free_fix_var)
                self.add_layer(layer, ctx_name)
                return layer
            return self.get_layer(ctx_name)
        else:  # we have free fixpoint variables
            # take all the free vars and remove them from nx.args
            freevars = []
            outputs = []
            for i, t in enumerate(nx.args):
                if self.free_fix_var[t.ref()] == self.current_fix_var():
                    freevars.append(self.current_fix_var_config())
                    outputs.append(t)
                    nx.args.pop(i)
                    break

            model = tf.keras.Model(inputs=[freevar.signature for freevar in freevars] + self.inputs.fixpoint_inputs, outputs=outputs)

            if type(initial_gnn_var) is FixPointExpression:
                new_expr = FixPointExpression('(' + name + ')',
                                              nx.args + initial_gnn_var.args + [freevar.signature for freevar in
                                                                                freevars] + initial_gnn_var.input_signature,
                                              lfp_layer(nx.args + initial_gnn_var.args + [model(
                                                  [freevar.signature for freevar in
                                                   freevars] + self.inputs.fixpoint_inputs)] + [
                                                            initial_gnn_var.signature] + self.inputs.fixpoint_inputs))
                new_expr.args = nx.args + initial_gnn_var.args
                return new_expr
            else:
                new_expr = FixPointExpression('(' + name + ')',
                                              nx.args + [initial_gnn_var.x] + [freevar.signature for freevar in
                                                                               freevars] + self.inputs.fixpoint_inputs,
                                              lfp_layer(nx.args + [model([freevar.signature for freevar in
                                                                          freevars] + self.inputs.fixpoint_inputs)] + [
                                                            initial_gnn_var.x] + self.inputs.fixpoint_inputs))
                new_expr.args = nx.args + [initial_gnn_var.x]
                return new_expr


class GNNCompiler:
    def __init__(self, psi_functions: FunctionDict[str, Psi | Callable[[str], Psi] | Type[Psi]],
                 sigma_functions: FunctionDict[str, Sigma | Callable[[str], Sigma] | Type[Sigma]],
                 phi_functions: FunctionDict[str, Phi | Callable[[str], Phi] | Type[Phi]],
                 config: CompilationConfig):
        """
        A compiler for mG formulas. A formula is transformed into a Tensorflow model using the compile method.

        :param psi_functions: A dictionary of Psi functions, from any among the Psi, PsiLocal and PsiGlobal classes
        :param sigma_functions: A dictionary of Sigma functions
        :param phi_functions: A dictionary of Phi functions
        :param config: A CompilationConfig object to configure this GNNCompiler object
        """
        self.parser = Lark(mg_grammar, maybe_placeholders=False, parser='lalr')
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
                                    IntermediateOutput("INPUT", *intermediate_output_args), config.precision, self.parser)

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
                print("Tracing completed in", elapsed, "s", sep='')
                break
        elif method == 'predict':
            start = time.perf_counter()
            model.predict(dummy_loader.load(), steps=dummy_loader.steps_per_epoch)
            end = time.perf_counter()
            elapsed = end - start
            print("Tracing completed in ", elapsed, "s", sep='')
        else:
            for x, in dummy_loader.load():
                start = time.perf_counter()
                model.predict_on_batch(x)
                end = time.perf_counter()
                elapsed = end - start
                print("Tracing completed in ", elapsed, "s", sep='')
                break
        return elapsed

    def compile(self, expr: str, verbose: bool = False) -> tf.keras.Model:
        """
        Compiles a mG formula `expr` into a Tensorflow Model.

        :param expr: A mG formula to evaluate
        :param verbose: Set this to True to print some debug information.
        :return: A Tensorflow Model that is the mG evaluation of 'expr'.
        """
        self.interpreter.initialize()
        outputs = self.interpreter.visit(self.macros.visit(self.parser.parse(expr)))
        model = tf.keras.Model(inputs=self.model_inputs, outputs=outputs.x)
        model.saved_layers = self.interpreter.layers
        if verbose is True:
            model.summary()
        self.interpreter.initialize()
        return model

    def optimize(self, model: tf.keras.Model, optimize: str) -> Tuple[tf.keras.Model, float] | Tuple[Callable, float]:
        """
        Performs tracing on the input model, and returns it, together with the time took for tracing.

        :param model: A TensorFlow model
        :param optimize: Set this to "call" to optimize the model for being used with the "call" API, set this to
        "predict" to optimize the model for being used with the "predict" API. When set to "call" the model is transformed into a function.
        """
        model = GNNCompiler.graph_mode_constructor(model, self.model_input_spec, optimize)
        compile_time = GNNCompiler.dummy_run(model, self.dummy_loader, optimize)
        return model, compile_time
