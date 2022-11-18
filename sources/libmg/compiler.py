from typing import List, Union

from lark import Lark, v_args
from lark.visitors import Interpreter
import tensorflow as tf
import time

# a, e, i cannot change
from .loaders import SingleGraphLoader, MultipleGraphLoader
from .macros import ApplyMacros
from .dummy_dataset import DummyDataset
from .grammar import mg_grammar
from .layers import PreImage, PostImage, LeastFixPoint, GreatestFixPoint, FunctionApplication


class NodeConfig:
    def __init__(self, node_type, node_size):
        self._type = node_type
        self._size = node_size

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return self._size


class EdgeConfig:
    def __init__(self, edge_type, edge_size):
        self._type = edge_type
        self._size = edge_size

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return self._size


class CompilationConfig:
    def __init__(self, node_config, edge_config, matrix_type, disjoint_loader):
        self.node_config = node_config
        self.edge_config = edge_config
        self._matrix_type = matrix_type
        self.disjoint_loader = disjoint_loader

    @staticmethod
    def xa_config(node_feature_type, node_feature_size, matrix_type):
        return CompilationConfig(NodeConfig(node_feature_type, node_feature_size), None, matrix_type, False)

    @staticmethod
    def xai_config(node_feature_type, node_feature_size, matrix_type):
        return CompilationConfig(NodeConfig(node_feature_type, node_feature_size), None, matrix_type, True)

    @staticmethod
    def xae_config(node_feature_type, node_feature_size, edge_feature_type, edge_feature_size, matrix_type):
        return CompilationConfig(NodeConfig(node_feature_type, node_feature_size),
                                 EdgeConfig(edge_feature_type, edge_feature_size), matrix_type, False)

    @staticmethod
    def xaei_config(node_feature_type, node_feature_size, edge_feature_type, edge_feature_size, matrix_type):
        return CompilationConfig(NodeConfig(node_feature_type, node_feature_size),
                                 EdgeConfig(edge_feature_type, edge_feature_size), matrix_type, True)

    @property
    def node_feature_type(self):
        return self.node_config.type

    @property
    def node_feature_size(self):
        return self.node_config.size

    @property
    def edge_feature_type(self):
        return self.edge_config.type if self.use_edges else None

    @property
    def edge_feature_size(self):
        return self.edge_config.size if self.use_edges else None

    @property
    def matrix_type(self):
        return self._matrix_type

    @property
    def use_edges(self):
        return self.edge_config is not None

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
    def __init__(self, var_name, signature):
        self._expr = lambda args: args[-1][0]  # the last element of args is a tuple (X, A, E, I), and we return X
        self._args = []  # initially no arguments
        self._name = var_name  # initially it is just a variable name
        self._signature = signature

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, value):
        self._expr = value

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


class Bottom:
    def __init__(self, dimension, value):
        self._dimension = dimension
        self._value = value if tf.is_tensor(value) else tf.constant(value)
        self.constructor = lambda x: tf.fill(dims=(tf.shape(x)[0], dimension), value=value)

    def __call__(self, x):
        return self.constructor(x)

    @property
    def dimension(self):
        return self._dimension

    @property
    def value(self):
        return self._value


class Top:
    def __init__(self, dimension, value):
        self._dimension = dimension
        self._value = value if tf.is_tensor(value) else tf.constant(value)
        self.constructor = lambda x: tf.fill(dims=(tf.shape(x)[0], dimension), value=value)

    def __call__(self, x):
        return self.constructor(x)

    @property
    def dimension(self):
        return self._dimension

    @property
    def value(self):
        return self._value


class FixPointConfig:
    def __init__(self, config, precision=None):
        self._constructor = config
        self._signature = tf.keras.Input(shape=config.dimension, dtype=config.value.dtype)
        self._precision = precision

    @property
    def signature(self):
        return self._signature

    @property
    def precision(self):
        return self._precision

    @property
    def constructor(self):
        return self._constructor


# make function for parallel op, only interested in the x and ignores a, e, i
def make_function_ops(op_layer, layers, name):
    args = []
    mappings: List[Union[int, List[int]]] = []
    signature = None
    for layer in layers:
        if type(layer) is FixPointExpression:
            idx_args = []
            signature = layer.signature
            for layer_arg in layer.args:
                args.append(layer_arg)
                idx_args.append(len(args) - 1)
            mappings.append(idx_args)
        else:
            args.append(layer.x)
            mappings.append(len(args) - 1)

    def f(_args):
        terms = []
        for i in range(len(layers)):
            if type(layers[i]) is FixPointExpression:
                mapped_args = mappings[i]
                assert isinstance(mapped_args, list)
                inputs = [_args[mapped_arg] for mapped_arg in mapped_args]
                inputs.append(_args[-1])
                arg = layers[i].expr(inputs)
                terms.append(arg)
            else:
                mapped_arg = mappings[i]
                terms.append(_args[mapped_arg])
        return op_layer(terms)

    output_dimension = 0
    for layer in layers:
        if type(layer) is FixPointExpression:
            output_dimension += layer.signature.shape[1]
        else:
            output_dimension += layer.x.shape[1]

    assert signature is not None
    new_expr = FixPointExpression('(' + name + ')', tf.keras.Input(shape=output_dimension, dtype=signature.dtype))
    new_expr.expr = f
    new_expr.args = args

    return new_expr


# asterisk in output: step whatever it returns. Used on base gnns, poolers, sequential
# no asterisk in output: only step x. Used on kop, parallel, lhd, rhd, mu, nu


class TreeToTF(Interpreter):
    def __init__(self, psi_functions, sigma_functions, phi_functions, bottoms, tops, inputs, parser):
        super().__init__()
        self.psi_functions = psi_functions
        self.sigma_functions = sigma_functions
        self.phi_functions = phi_functions
        self.bottoms = bottoms
        self.tops = tops
        self.initial_inputs = inputs
        self.parser = parser
        # Initialization
        self.var = []
        self.var_type = []
        self.context = []
        self.layers = {}
        self.disable_saving_layers = False
        self.inputs = self.initial_inputs

    def add_layer(self, layer, ctx_name):
        if not self.disable_saving_layers:
            self.layers[ctx_name] = layer

    def get_contextualized_name(self, name):
        if len(self.context) == 0:
            return '(' + name + ')'
        else:
            return '(' + self.head(self.context) + ';' + name + ')'

    @staticmethod
    def pop(stack):
        return stack.pop()

    @staticmethod
    def push(value, stack):
        stack.append(value)

    @staticmethod
    def head(stack):
        return stack[-1]

    def initialize(self):
        self.var = []
        self.var_type = []
        self.context = []
        self.layers = {}
        self.disable_saving_layers = False
        self.inputs = self.initial_inputs

    @v_args(inline=True)
    def function_name(self, f):
        return str(f)

    @v_args(inline=True)
    def variable_decl(self, var):
        return str(var)

    @v_args(inline=True)
    def type_decl(self, type_decl):
        return str(type_decl)

    @v_args(inline=True)
    def fun_app(self, function):
        f = self.visit(function)
        ctx_name = self.get_contextualized_name(f)
        f_layer = FunctionApplication(self.psi_functions[f])
        if ctx_name not in self.layers:
            layer = self.inputs.step(ctx_name, f_layer(self.inputs.func_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        else:
            return self.layers[ctx_name]

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
        if ctx_name not in self.layers:
            layer = self.inputs.step(ctx_name, lhd_layer(self.inputs.img_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.layers[ctx_name]

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
        if ctx_name not in self.layers:
            layer = self.inputs.step(ctx_name, rhd_layer(self.inputs.img_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.layers[ctx_name]

    @v_args(inline=True)
    def variable(self, var):
        if str(var) == self.head(self.var):
            return FixPointExpression(str(var), self.head(self.var_type).signature)
        else:
            raise SyntaxError('Undeclared variable: ', str(var))

    @v_args(inline=True)
    def composition(self, phi, psi):
        current_inputs = self.inputs
        phi = self.visit(phi)
        self.push(phi.name, self.context)
        if type(phi) is FixPointExpression:  # phi is a fixpoint expression
            self.disable_saving_layers = True
            self.inputs = current_inputs.step(phi.name, phi.signature)
            psi = self.visit(psi)
            self.pop(self.context)
            self.disable_saving_layers = False
            if type(psi) is FixPointExpression:  # both are fixpoint expressions
                raise ValueError("A macro should have been applied instead!")
            # only phi is a fixpoint expression
            m2 = tf.keras.Model(inputs=self.inputs.full_inputs, outputs=psi.x)
            new_expr = FixPointExpression('(' + psi.name + ')',
                                          tf.keras.Input(shape=psi.x.shape[1], dtype=psi.x.dtype))
            new_expr.expr = lambda args: m2([phi.expr(args), *args[-1][1:]])
            new_expr.args = phi.args
            self.inputs = current_inputs
            return new_expr
        else:  # phi is not a fixpoint expression
            self.inputs = phi
            psi = self.visit(psi)
            if type(psi) is FixPointExpression:  # phi is not a fixpoint expression but psi is
                raise ValueError("A macro should have been applied instead!")
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
            return make_function_ops(tf.keras.layers.Concatenate(), args, name)
        else:
            ctx_name = self.get_contextualized_name(name)
            if ctx_name not in self.layers:
                layer = args[0].step(ctx_name, tf.keras.layers.Concatenate()([arg.x for arg in args]))
                self.add_layer(layer, ctx_name)
                return layer
            return self.layers[ctx_name]

    @v_args(inline=True)
    def mu_formula(self, variable_decl, type_decl, nx):
        var_name = self.visit(variable_decl)
        self.push(var_name, self.var)
        type_decl = self.visit(type_decl)
        self.push(self.bottoms[type_decl], self.var_type)
        nx = self.visit(nx)
        if type(nx) is not FixPointExpression:
            raise SyntaxError('Invalid fixpoint expression')
        name = 'mu ' + self.pop(self.var) + ',' + type_decl + ' . ' + nx.name
        type_config = self.pop(self.var_type)
        ctx_name = self.get_contextualized_name(name)
        lfp_layer = LeastFixPoint(nx.expr, type_config.constructor, type_config.precision)
        if ctx_name not in self.layers:
            layer = self.inputs.step(ctx_name, lfp_layer(nx.args + self.inputs.fixpoint_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.layers[ctx_name]

    @v_args(inline=True)
    def nu_formula(self, variable_decl, type_decl, nx):
        var_name = self.visit(variable_decl)
        self.push(var_name, self.var)
        type_decl = self.visit(type_decl)
        self.push(self.tops[type_decl], self.var_type)
        nx = self.visit(nx)
        if type(nx) is not FixPointExpression:
            raise SyntaxError('Invalid fixpoint expression')
        name = 'nu ' + self.pop(self.var) + ',' + type_decl + ' . ' + nx.name
        type_config = self.pop(self.var_type)
        ctx_name = self.get_contextualized_name(name)
        gfp_layer = GreatestFixPoint(nx.expr, type_config.constructor, type_config.precision)
        if ctx_name not in self.layers:
            layer = self.inputs.step(ctx_name, gfp_layer(nx.args + self.inputs.fixpoint_inputs))
            self.add_layer(layer, ctx_name)
            return layer
        return self.layers[ctx_name]


class GNNCompiler:
    def __init__(self, psi_functions, sigma_functions, phi_functions, bottoms, tops, config: CompilationConfig):
        self.parser = Lark(mg_grammar, maybe_placeholders=False)
        self.macros = ApplyMacros(self.parser)
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
        self.dummy_loader = MultipleGraphLoader(dummy_dataset, node_level=True, batch_size=1, shuffle=False, epochs=1) if\
            config.use_disjoint else SingleGraphLoader(dummy_dataset, epochs=1)
        self.interpreter = TreeToTF(psi_functions, sigma_functions, phi_functions, bottoms, tops,
                                    IntermediateOutput("INPUT", *intermediate_output_args), self.parser)

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

    def compile(self, expr, loss=None, verbose=False, optimize=None, return_compilation_time=False):
        self.interpreter.initialize()
        outputs = self.interpreter.visit(self.macros.transform(self.parser.parse(expr)))
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
