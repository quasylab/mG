import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, PsiGlobal, Sigma, Phi
from libmg import SingleGraphLoader, MultipleGraphLoader
from libmg import GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig
from libmg import Dataset
from libmg.functions import Constant, make_koperator, make_boperator, make_uoperator


class A(PsiLocal):
    def func(self, x):
        return tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)


class B(PsiLocal):
    def func(self, x):
        return tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)


class C(PsiLocal):
    def func(self, x):
        return tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)


class And(PsiLocal):
    def func(self, x):
        return tf.math.reduce_all(x, axis=1, keepdims=True)


class Or(PsiLocal):
    def func(self, x):
        return tf.math.reduce_any(x, axis=1, keepdims=True)


class Not(PsiLocal):
    def func(self, x):
        return tf.math.logical_not(x)


class Id(PsiLocal):
    def func(self, x):
        return x


class Add1(PsiLocal):
    def func(self, x):
        return x + 1


class Sub1(PsiLocal):
    def func(self, x):
        return x - 1


class Le2(PsiLocal):
    def func(self, x):
        return x < 2


class Max(Sigma):
    def func(self, m, i, n, x):
        return tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)


class UMax(Sigma):
    def func(self, m, i, n, x):
        return tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n), tf.bool)


class IsZero(Phi):
    def func(self, src, e, tgt):
        return tf.math.logical_and(tgt, tf.equal(e, 0))


class One(PsiLocal):
    def func(self, x):
        return tf.ones_like(x)


class Min(PsiLocal):
    def func(self, x):
        return x[:, :1] - x[:, 1:]


class Subtract(PsiLocal):
    def func(self, x):
        return tf.math.subtract(x[:, :1], x[:, 1:])


class Sum(PsiGlobal):
    def multiple_graph_op(self, x, i):
        return tf.math.segment_sum(x, i)

    def single_graph_op(self, x):
        return tf.reduce_sum(x, axis=0, keepdims=True)


def base_tester(dataset, compilers, expressions):
    loaders = [SingleGraphLoader(dataset, epochs=1),
               MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)] + \
              [SingleGraphLoader(dataset, epochs=1),
               MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
    for loader, compiler in zip(loaders, compilers):
        for e in expressions:
            model = compiler.compile(e)
            for inputs in loader.load():
                print(model.call([inputs], training=False))


class TestDataset(Dataset):
    def __init__(self, n=1, edges=False, **kwargs):
        self.n = n
        self.edges = edges
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        for i in range(self.n):
            x = np.array([[1], [2], [4], [1], [1]])
            a = coo_matrix(([1, 1, 1, 1, 1, 1, 1], ([0, 0, 1, 2, 2, 3, 4], [1, 2, 2, 1, 3, 4, 1])), shape=(5, 5))
            e = np.array([[1], [0], [0], [0], [1], [1], [1]])
            if self.edges:
                graphs.append(Graph(x, a, e))
            else:
                graphs.append(Graph(x, a))
        return graphs


class CudaTest(tf.test.TestCase):
    def setUp(self):
        super(CudaTest, self).setUp()

    def test_cuda(self):
        self.assertEqual(tf.test.is_built_with_cuda(), True)


class BaseTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = TestDataset(n=1, edges=False)
        psi_dict_lambdas = {'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            '100': Constant(tf.constant([100.0, 100.0], dtype=tf.float32)),
            'true': Constant(tf.constant(True), name='True'),
            'false': Constant(tf.constant(False), name='False'),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x),
            'eq': PsiLocal(lambda x: tf.equal(x[:, :1], x[:, 1:])),
            'pr1': PsiLocal(lambda x: x[:, 1:]),
            'le': PsiLocal(lambda x: x < 2),
            'div': PsiLocal(lambda x: x / 2),
            'add1': PsiLocal(lambda x: x + 1),
            'sub1': PsiLocal(lambda x: x - 1),
            '-': PsiLocal(lambda x: tf.math.subtract(x[:, :1], x[:, 1:])),
            '+': make_koperator(tf.math.add_n),
            '++': make_boperator(tf.math.add),
            '~': make_uoperator(tf.math.logical_not)}
        self.psi_dict_lambdas = psi_dict_lambdas
        sigma_dict_lambdas = {
            '*': Sigma(lambda m, i, n, x: tf.math.segment_prod(m, i)),
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))}
        self.sigma_dict_lambdas = sigma_dict_lambdas
        psi_dict_subclassing = {'a': A, 'b': B, 'c': C, 'true': Constant(tf.constant(True)),
                                'false': Constant(tf.constant(False)), 'and': And,
                                'or': Or, 'not': Not, 'id': Id, 'le': Le2, 'add1': Add1, 'sub1': Sub1, '-': Subtract,
                                '+': make_koperator(tf.math.add_n),
                                '++': make_boperator(tf.math.add),
                                '~': make_uoperator(tf.math.logical_not)}
        sigma_dict_subclassing = {'or': Max, 'uor': UMax}
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions={},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions={},
                config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))
        ]

    def test_simple_expr(self):
        expr = ['a', 'true', 'false', '<| uor', '|> or', 'a;not', '(a || b);and', '(a || b);or', 'a; false', 'a || b']
        base_tester(self.dataset, self.compilers, expr)

    def test_fixpoint_expr(self):
        expr = ['fix X = false in (X ; |> or)', 'fix X = false in ((X || a);and)',
                'fix X = true in  (X ; |> or)', 'fix X = true in ((X || a);and)',
                'fix Y = false in (Y ; |> or)',
                '(fix X = false in (X ; |> or)) || (fix X = true in ((X || a);and))']
        base_tester(self.dataset, self.compilers, expr)

    def test_numeric_fixpoint_expr(self):
        expr = ['fix X = 100 in (X ; div)']
        compiler = GNNCompiler(
            psi_functions=self.psi_dict_lambdas,
            sigma_functions=self.sigma_dict_lambdas,
            phi_functions={},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {'float32': (0.001, 'anderson')}))
        model = compiler.compile(expr[0])
        loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs in loader.load():
            model.call([inputs], training=False)

    def test_seq_expr(self):
        expr = ['a;not', 'a;not;not', 'a;not;not;not',
                'fix X = false in (X ; not; not)',
                'fix X = false in (X ; id)',
                'fix X = true in (a ; ((X || not);and))',
                'fix X = false in (((X || a);or) ; ((X || not);or))']
        base_tester(self.dataset, self.compilers, expr)

    def test_par_expr(self):
        expr = ['a || b || c', '(a || (b || c));or', 'fix X = false in ((X || a);or)',
                'fix X = false in ((a || X);and)']
        base_tester(self.dataset, self.compilers, expr)

    def test_univariate_functions(self):
        expr = ["""
        def test(X){
        (true || X);and;not
        } in test(a)
        """, """
        def test(X){
        (true || X);and;not
        } in test(test(a))
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_multivariate_functions(self):
        expr = ["""
        def test(X, Y){
        (Y || X);and;not
        } in test(a, b)
        """, """
        def test(X, Y){
        (Y || X);and;not
        } in test(test(a, b), b)
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_opk_macro(self):
        expr = ['and(a, b, c)', 'and(a, b, or(c, true))', """
        def test(X, Y){
        (Y || X);and;not
        } in test(a, and(b, c))
        """,
        """
        def test(X, Y){
        (Y || and(X, c));and;not
        } in test(a, b)
        """, """
        and(a, def test(X, Y){
        (Y || X);and;not
        } in test(b, c))
        """, 'and(a, b) || true(-(add1, sub1))',
                '+(add1, sub1, id)',
                '++(add1, sub1)',
                '~(true)']

        base_tester(self.dataset, self.compilers, expr)

    def test_variables(self):
        expr = ["""
        let test = b;not, test2 = a;not in
        test || test2
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse_var_layers(self):
        expr = ["""
        let test = b;not, test2 = a;not in
        a
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_local_variable_expr(self):
        expr = ["""
        let x = a;not, y = b in
        (x || y);and
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_ite(self):
        expr = ['(if a then b else false) || b', 'add1; if le then add1 else add1']
        base_tester(self.dataset, self.compilers, expr[1:])

    def test_ite_multiple(self):
        dataset = TestDataset(n=2, edges=False)
        dataset[0].x = np.array([[1], [1], [1], [1], [1]])
        dataset[1].x = np.array([[2], [2], [2], [2], [2]])
        expr = 'if le then add1 else sub1'
        loaders = [MultipleGraphLoader(dataset, epochs=1, batch_size=2, shuffle=False),
                   MultipleGraphLoader(dataset, epochs=1, batch_size=2, shuffle=False)]
        compilers = [self.compilers[1], self.compilers[3]]
        for loader, compiler in zip(loaders, compilers):
            model = compiler.compile(expr)
            for inputs in loader.load():
                self.assertAllEqual([[2], [2], [2], [2], [2], [1], [1], [1], [1], [1]],
                                    model.call(inputs, training=False))

    def test_new_fixpoint(self):
        expr = ['fix X = true in X', 'fix X = true in (X;false;|>or)',
                'fix X = true in (X || a);and',
                'fix X = true in (((a;not) || (X;|>or));and)',
                'fix X = true in (let y = a;not in ((y || (X;|>or));and))',
                'def test(y){ ((a;not) || (y;|>or));and } in fix X = true in (test(X))',
                'fix X = true in (if X then true else false)',
                'fix X = true in (if true then X else false)',
                'fix X = true in (if false then false else X)',
                'fix X = false in (if true then X else X)', 'fix X = false in (if X then X else true)',
                'fix X = true in (if X then false else X)', 'fix X = false in (if X then X else X)',
                ]
        base_tester(self.dataset, self.compilers, expr)

        dataset = TestDataset(n=2, edges=False)
        dataset[0].x = np.array([[1], [1], [1], [1], [1]])
        dataset[1].x = np.array([[2], [2], [2], [2], [2]])
        compilers = [self.compilers[1], self.compilers[3]]

        def get_loader():
            return MultipleGraphLoader(dataset, epochs=1, batch_size=2)

        expr = ['fix X = true in (if X then true else false)', 'fix X = true in (if true then X else false)',
                'fix X = false in (if X then X else true)', 'fix X = false in (if X then X else X)']
        for compiler in compilers:
            for e in expr:
                model = compiler.compile(e)
                for inputs in get_loader().load():
                    model.call(inputs, training=False)

    def test_nested_fixpoints(self):
        expr = ['fix X = false in (fix Y = true in (X || Y);and)',
                'fix X = false in (fix Y = false in (X || Y || true);or)',
                'fix X = false in (fix Y = X in Y)', 'fix X = false in (fix Y = X in Y;not;not)',
                'fix X = true in (fix Y = true in (X || Y || a);and)',
                'fix Z = true in (fix X = false in (fix Y = false in (X || Y || Z);or))',
                'fix X = true in ((fix Y = false in (X || Y);and) || (fix Z = true in (X || Z);and));or',
                'fix X = false in (fix Y = true in ((X;not) || Y);or)',
                'fix X = false in (fix Y = true in X)']
        base_tester(self.dataset, self.compilers, expr)

    def test_constants(self):
        expr = ['fix X = (false || true)  in X']
        base_tester(self.dataset, self.compilers, expr)

    def test_repeat(self):
        expr = ['repeat X = false in X;not for 3']
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; |> or) || (fix X = false in ((a || X) ; or)) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)] + \
                  [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]

        for i, (loader, compiler) in enumerate(zip(loaders, self.compilers)):
            if i % 2 == 0:
                expected_n_layers = 11  # account for the I layer
            else:
                expected_n_layers = 12
            model = compiler.compile(expr, memoize=True)
            for inputs in loader().load():
                model.call([inputs], training=False)
            self.assertEqual(expected_n_layers, len(model.layers))

    def test_verbose(self):
        expr = ['a', 'true', 'false', '<| uor', '|> or', 'a;not', '(a || b);and', '(a || b);or', 'a; false', 'a || b',
                'if true then true else false', 'fix X = true in (((X;false) || a);or)', 'repeat X = true in X;not for 3']
        compiler = self.compilers[2]
        for e in expr:
            compiler.compile(e, verbose=True)

    def test_disable_eval_if(self):
        expr = ['fix X = false in (if true then (((if true then X else X) || X);or) else true)',
                'fix X = false in (if true then (((if true then X else false) || X);or) else true)',
                'fix X = false in (if true then (((if X then false else false) || X);or) else true)',
                'fix X = false in (if true then ((true || X);or) else true)',
                'fix X = false in (if true then (X;not;not) else true)',
                'def f(Y){Y} in fix X = false in (if true then f(X) else true)',
                'fix X = false in (if true then (fix Y = true in ((X || Y);and)) else true)',
                'repeat X = false in (if true then (fix Y = true in ((X || Y);and)) else true) for 2',
                'fix X = false in (if true then (fix Y = X in Y) else true)']
        compiler = self.compilers[0]
        for e in expr:
            model = compiler.compile(e)
            model.summary()
            print(model.mg_layers)
            for k,v in model.mg_layers.items():
                print(v.name)
            for inputs in SingleGraphLoader(self.dataset, epochs=1).load():
                print(model.call([inputs], training=False))

    def test_names(self):
        expr = ['true;false;true;false;false']
        expr = ['true;false;true;(true || false);false']
        expr = ['true;false;true;(if true then false else true);false']
        expr = ['true;false;true;(if true then false else true);false']
        expr = ['true;false;true;(fix X = (true;false) in X);false']
        expr = ['true;false;true;(repeat X = (true;false) in X for 3);false']
        expr = ['true;false;true;(fix X = true in X);false']
        expr = ['true;false;true;(fix X = (true;false) in (X;false));false']
        expr = ['true;false;true;(def f(X){X;false} in f(true;false));false']
        expr = ['true;false;true;(let X = true;false in (X;false));false']
        expr = ['true ; false ; let X = true in (X || (let Y = false in (Y;true)))']
        expr = ['true ; def f(X){X} in (f(true) || f(false))']
        expr = ['true ; fix X = false in X;true;false']
        expr = ['true ; if not then true else false']
        expr = ['fix X = true in (fix Y = false in ((X;id) || Y);and)']

        expr = ['true;(if true then false else true)']
        expr = ['true ; fix X = false in (if true then X else X)']
        expr = ['true ; fix X = false in (if X then true else false)']
        expr = ['true ; fix X = false in (if X then true else X)']

        expr = ['true ; fix X = false in (if (((X;not) || true);and) then false else true)']
        expr = ['true ; fix X = false in (if (((X;not) || true);and) then (((X;not) || true);and) else true)']
        compiler = self.compilers[0]
        for e in expr:
            model = compiler.compile(e)


class EdgeTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = TestDataset(n=1, edges=True)
        psi_dict_lambdas = {'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            'true': Constant(tf.constant(True)),
            'false': Constant(tf.constant(False)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x)}
        sigma_dict_lambdas = {
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))}
        phi_dict_lambdas = {'z': Phi(lambda i, e, j: tf.math.logical_and(j, tf.equal(e, 0)))}

        psi_dict_subclassing = {'a': A, 'b': B, 'c': C, 'true': Constant(tf.constant(True)),
                                'false': Constant(tf.constant(False)), 'and': And,
                                'or': Or, 'not': Not, 'id': Id}
        sigma_dict_subclassing = {'or': Max, 'uor': UMax}
        phi_dict_subclassing = {'z': IsZero}

        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions=phi_dict_lambdas,
            config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions=phi_dict_lambdas,
                config=CompilationConfig.xaei_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions=phi_dict_subclassing,
                config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions=phi_dict_subclassing,
                config=CompilationConfig.xaei_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {}))]

    def test_edge_expr(self):
        expr = ['a ; |z> or', 'a ; <z| uor', '( (a ; |z> or) || (b ; |z> or) ); or', ' a ; |z> or ; |z> or',
                '(b ; |z> or) || ( c ; |z> or)', 'fix X = true in (X ; |z> or)']
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; <z| uor) || (fix X = false in (((b ; <z| uor) || X);or)) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)] + \
                  [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]

        for i, (loader, compiler) in enumerate(zip(loaders, self.compilers)):
            if i % 2 == 0:
                expected_n_layers = 12  # account for the I layer
            else:
                expected_n_layers = 13
            model = compiler.compile(expr, memoize=True)
            for inputs in loader().load():
                model.call([inputs], training=False)
            self.assertEqual(expected_n_layers, len(model.layers))


class PoolTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.single_dataset = TestDataset(n=1, edges=False)
        self.multiple_dataset = TestDataset(n=10, edges=False)
        psi_dict_lambdas = {'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            'true': Constant(tf.constant(True)),
            'false': Constant(tf.constant(False)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x),
            'one': PsiLocal(tf.ones_like),
            'min': PsiLocal(lambda x: x[:, :1] - x[:, 1:]),
            'gsum': PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=True),
                              multiple_op=lambda x, i: tf.math.segment_sum(x, i))}
        sigma_dict_lambdas = {
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))}
        psi_dict_subclassing = {'a': A, 'b': B, 'c': C, 'true': Constant(tf.constant(True)),
                                'false': Constant(tf.constant(False)), 'and': And,
                                'or': Or, 'not': Not, 'id': Id, 'one': One, 'min': Min, 'gsum': Sum}
        sigma_dict_subclassing = {'or': Max, 'uor': UMax}
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions={},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions={},
                config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_subclassing,
                sigma_functions=sigma_dict_subclassing,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))]

    def test_global_pooling(self):
        expr = ['gsum', '(gsum || one);min']
        loaders = [SingleGraphLoader(self.single_dataset, epochs=1),
                   MultipleGraphLoader(self.multiple_dataset, node_level=True, batch_size=10, shuffle=False,
                                       epochs=1)] + \
                  [SingleGraphLoader(self.single_dataset, epochs=1),
                   MultipleGraphLoader(self.multiple_dataset, node_level=True, batch_size=10, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader.load():
                    model.call([inputs], training=False)


if __name__ == '__main__':
    tf.test.main()
