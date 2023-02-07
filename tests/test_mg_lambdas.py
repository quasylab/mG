import os
import tensorflow as tf
import numpy as np

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, PsiGlobal, Sigma, Phi, FunctionDict
from libmg import SingleGraphLoader, MultipleGraphLoader
from libmg import GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig
from libmg import Dataset
from libmg.functions import Constant

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


def base_tester(dataset, compilers, expressions):
    loaders = [SingleGraphLoader(dataset, epochs=1),
               MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
    for loader, compiler in zip(loaders, compilers):
        for e in expressions:
            model = compiler.compile(e)
            for inputs in loader.load():
                model.call([inputs], training=False)


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
        psi_dict = FunctionDict({'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            'true': Constant(lambda n: tf.ones((n, 1), dtype=tf.bool)),
            'false': Constant(lambda n: tf.zeros((n, 1), dtype=tf.bool)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x),
            'eq': PsiLocal(lambda x: tf.equal(x[:, :1], x[:, 1:])),
            'pr1': PsiLocal(lambda x: x[:, 1:]),
            'le': PsiLocal(lambda x: x < 2),
            'add1': PsiLocal(lambda x: x + 1),
            'sub1': PsiLocal(lambda x: x - 1)})
        sigma_dict = FunctionDict({
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))})
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict,
            sigma_functions=sigma_dict,
            phi_functions=FunctionDict({}),
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict,
                sigma_functions=sigma_dict,
                phi_functions=FunctionDict({}),
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))]

    def base_tester(self, expressions):
        loaders = [SingleGraphLoader(self.dataset, epochs=1),
                   MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expressions:
                model = compiler.compile(e)
                for inputs in loader.load():
                    model.call([inputs], training=False)

    def test_simple_expr(self):
        expr = ['a', 'true', 'false', '<| uor', '|> or', 'a;not', '(a || b);and', '(a || b);or', 'a; false', 'a || b']
        base_tester(self.dataset, self.compilers, expr)

    def test_fixpoint_expr(self):
        expr = ['fix X:bool[1] = false in (X ; |> or)', 'fix X:bool[1] = false in ((X || a);and)',
                'fix X:bool[1] = true in  (X ; |> or)', 'fix X:bool[1] = true in ((X || a);and)',
                'fix Y:bool[1] = false in (Y ; |> or)',
                '(fix X:bool[1] = false in (X ; |> or)) || (fix X:bool[1] = true in ((X || a);and))']
        base_tester(self.dataset, self.compilers, expr)

    def test_seq_expr(self):
        expr = ['a;not', 'a;not;not', 'a;not;not;not',
                'fix X:bool[1] = false in (X ; not; not)',
                'fix X:bool[1] = false in (X ; id)', 'fix X:bool[1] = true in (a ; ((X || not);and))',
                'fix X:bool[1] = false in (((X || a);or) ; ((X || not);or))']
        base_tester(self.dataset, self.compilers, expr)

    def test_par_expr(self):
        expr = ['a || b || c', '(a || (b || c));or', 'fix X:bool[1] = false in ((X || a);or)',
                'fix X:bool[1] = false in ((a || X);and)']
        base_tester(self.dataset, self.compilers, expr)

    def test_univariate_functions(self):
        expr = ["""
        def test(X:bool[1]){
        (true || X);and;not
        }
        test(a)
        """, """
        def test(X:bool[1]){
        (true || X);and;not
        }
        test(test(a))
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_multivariate_functions(self):
        expr = ["""
        def test(X:bool[1], Y:bool[1]){
        (Y || X);and;not
        }
        test(a, b)
        """, """
        def test(X:bool[1], Y:bool[1]){
        (Y || X);and;not
        }
        test(test(a, b), b)
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_variables(self):
        expr = ["""
        let test = b;not, test2 = a;not
        test || test2
        """]
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse_var_layers(self):
        expr = ["""
        let test = b;not, test2 = a;not
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
        base_tester(self.dataset, self.compilers, expr)

    def test_ite_multiple(self):
        dataset = TestDataset(n=2, edges=False)
        dataset[0].x = np.array([[1], [1], [1], [1], [1]])
        dataset[1].x = np.array([[2], [2], [2], [2], [2]])
        expr = 'if le then add1 else sub1'
        loader = MultipleGraphLoader(dataset, epochs=1, batch_size=2)
        compiler = self.compilers[1]
        model = compiler.compile(expr)
        for inputs in loader.load():
            self.assertAllEqual([[2], [2], [2], [2], [2], [1], [1], [1], [1], [1]], model.call(inputs, training=False))

    # def test_loop(self):
    #    expr = ['add1; while le do add1']
    #    self.base_tester(expr)

    '''
    def test_old_fixpoint(self):
        expr = 'nu X:bool[1] = true . (((a;not) || (X;|>or));and)'
        loader = lambda: SingleGraphLoader(self.dataset, epochs=1)
        compiler = self.compilers[0]
        model = compiler.compile(expr)
        for inputs in loader().load():
            print(model.call([inputs], training=False))
    '''

    def test_new_fixpoint(self):
        expr = ['fix X:bool[1] = true in X', 'fix X:bool[1] = true in (X;false;|>or)',
                'fix X:bool[1] = true in (X || a);and',
                'fix X:bool[1] = true in (((a;not) || (X;|>or));and)',
                'fix X:bool[1] = true in (let y = a;not in ((y || (X;|>or));and))',
                'def test(y:bool[1]){ ((a;not) || (y;|>or));and } fix X:bool[1] = true in (test(X))',
                'fix X:bool[1] = true in (if X then true else false)',
                'fix X:bool[1] = true in (if true then X else false)',
                'fix X:bool[1] = true in (if false then false else X)',
                'fix X:bool[1] = false in (if true then X else X)', 'fix X:bool[1] = false in (if X then X else true)',
                'fix X:bool[1] = true in (if X then false else X)', 'fix X:bool[1] = false in (if X then X else X)',
                ]
        base_tester(self.dataset, self.compilers, expr)

        dataset = TestDataset(n=2, edges=False)
        dataset[0].x = np.array([[1], [1], [1], [1], [1]])
        dataset[1].x = np.array([[2], [2], [2], [2], [2]])

        expr = 'fix X:bool[1] = true in (if X then true else false)'
        loader = MultipleGraphLoader(dataset, epochs=1, batch_size=2)
        compiler = self.compilers[1]
        model = compiler.compile(expr)
        for inputs in loader.load():
            model.call(inputs, training=False)

        expr = 'fix X:bool[1] = true in (if true then X else false)'
        loader = MultipleGraphLoader(dataset, epochs=1, batch_size=2)
        compiler = self.compilers[1]
        model = compiler.compile(expr)
        for inputs in loader.load():
            model.call(inputs, training=False)

        expr = 'fix X:bool[1] = false in (if X then X else true)'
        loader = MultipleGraphLoader(dataset, epochs=1, batch_size=2)
        compiler = self.compilers[1]
        model = compiler.compile(expr)
        for inputs in loader.load():
            model.call(inputs, training=False)

        expr = 'fix X:bool[1] = false in (if X then X else X)'
        loader = MultipleGraphLoader(dataset, epochs=1, batch_size=2)
        compiler = self.compilers[1]
        model = compiler.compile(expr)
        for inputs in loader.load():
            model.call(inputs, training=False)

    def test_constants(self):
        expr = ['fix X:bool[1] = (false || true)  in X']
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; |> or) || (fix X:bool[1] = false in ((a || X) ; or)) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        expected_n_layers = 11
        for loader, compiler in zip(loaders, self.compilers):
            model = compiler.compile(expr)
            for inputs in loader().load():
                model.call([inputs], training=False)
            self.assertEqual(len(model.layers), expected_n_layers)
            expected_n_layers += 1  # account for the I layer


class EdgeTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = TestDataset(n=1, edges=True)
        psi_dict = FunctionDict({'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            'true': Constant(lambda n: tf.ones((n, 1), dtype=tf.bool)),
            'false': Constant(lambda n: tf.zeros((n, 1), dtype=tf.bool)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x)})
        sigma_dict = FunctionDict({
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))})
        phi_dict = FunctionDict({'z': Phi(lambda i, e, j: tf.math.logical_and(j, tf.equal(e, 0)))})
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict,
            sigma_functions=sigma_dict,
            phi_functions=phi_dict,
            config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict,
                sigma_functions=sigma_dict,
                phi_functions=phi_dict,
                config=CompilationConfig.xaei_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {}))]

    def test_edge_expr(self):
        expr = ['a ; |z> or', 'a ; <z| uor', '( (a ; |z> or) || (b ; |z> or) ); or', ' a ; |z> or ; |z> or',
                '(b ; |z> or) || ( c ; |z> or)', 'fix X:bool[1] = true in (X ; |z> or)']
        base_tester(self.dataset, self.compilers, expr)

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; <z| uor) || (fix X:bool[1] = false in (((b ; <z| uor) || X);or)) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        expected_n_layers = 12
        for loader, compiler in zip(loaders, self.compilers):
            model = compiler.compile(expr)
            for inputs in loader().load():
                model.call([inputs], training=False)
            self.assertEqual(expected_n_layers, len(model.layers))
            expected_n_layers += 1  # account for the I layer


class PoolTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.single_dataset = TestDataset(n=1, edges=False)
        self.multiple_dataset = TestDataset(n=10, edges=False)
        psi_dict = FunctionDict({'a': PsiLocal(
            lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
            'b': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
            'c': PsiLocal(
                lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
            'true': Constant(lambda n: tf.ones((n, 1), dtype=tf.bool)),
            'false': Constant(lambda n: tf.zeros((n, 1), dtype=tf.bool)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x),
            'one': PsiLocal(tf.ones_like),
            'min': PsiLocal(lambda x: x[:, :1] - x[:, 1:]),
            'gsum': PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=True))})
        sigma_dict = FunctionDict({
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))})
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict,
            sigma_functions=sigma_dict,
            phi_functions=FunctionDict({}),
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=FunctionDict({'a': PsiLocal(
                    lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
                    'b': PsiLocal(
                        lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
                    'c': PsiLocal(
                        lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
                    'true': Constant(lambda n: tf.ones((n, 1), dtype=tf.bool)),
                    'false': Constant(lambda n: tf.zeros((n, 1), dtype=tf.bool)),
                    'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
                    'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
                    'not': PsiLocal(lambda x: tf.math.logical_not(x)),
                    'id': PsiLocal(lambda x: x),
                    'one': PsiLocal(tf.ones_like),
                    'min': PsiLocal(lambda x: x[:, :1] - x[:, 1:]),
                    'gsum': PsiGlobal(multiple_op=tf.math.segment_sum)}),
                sigma_functions=sigma_dict,
                phi_functions=FunctionDict({}),
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))]

    def test_global_pooling(self):
        expr = ['gsum', '(gsum || one);min']
        loaders = [SingleGraphLoader(self.single_dataset, epochs=1),
                   MultipleGraphLoader(self.multiple_dataset, node_level=True, batch_size=10, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader.load():
                    model.call([inputs], training=False)


if __name__ == '__main__':
    tf.test.main()
