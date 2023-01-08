import os
import tensorflow as tf
import numpy as np

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, PsiGlobal, Sigma, Phi, FunctionDict
from libmg import SingleGraphLoader, MultipleGraphLoader
from libmg import GNNCompiler, CompilationConfig, FixPointConfig, NodeConfig, EdgeConfig
from libmg import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


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
                'true': PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool)),
                'false': PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool)),
                'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
                'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
                'not': PsiLocal(lambda x: tf.math.logical_not(x)),
                'id': PsiLocal(lambda x: x)})
        sigma_dict = FunctionDict({
                'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
                'uor': Sigma(
                    lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                               tf.bool))})
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict,
            sigma_functions=sigma_dict,
            phi_functions=FunctionDict({}),
            bottoms={'b': FixPointConfig(1, False)},
            tops={'b': FixPointConfig(1, True)},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8)),
            GNNCompiler(
                psi_functions=psi_dict,
                sigma_functions=sigma_dict,
                phi_functions=FunctionDict({}),
                bottoms={'b': FixPointConfig(1, False)},
                tops={'b': FixPointConfig(1, True)},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8))]

    def test_simple_expr(self):
        expr = ['a', '<| uor', '|> or', 'a;not', '(a || b);and', '(a || b);or', 'a; false', 'a || b']
        loaders = [SingleGraphLoader(self.dataset, epochs=1),
                   MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader.load():
                    model.call([inputs], training=False)

    def test_fixpoint_expr(self):
        expr = ['mu X,b . (X ; |> or)', 'mu X,b . ((X || a);and)', 'nu X,b .  (X ; |> or)', 'nu X,b . ((X || a);and)',
                'mu Y,b . (Y ; |> or)', 'mu X,b . (X ; |> or) || nu X,b . ((X || a);and)']
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader().load():
                    model.call([inputs], training=False)

    def test_seq_expr(self):
        expr = ['a;not', 'a;not;not', 'a;not;not;not', 'mu X,b . (a ; ((X || not);and))', 'mu X,b . (X ; not; not)',
                'mu X,b . (X ; id)', 'mu X,b . ((X ; not) ; ((X || not);or))']
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader().load():
                    model.call([inputs], training=False)

    def test_par_expr(self):
        expr = ['a || b || c', '(a || (b || c));or']
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader().load():
                    model.call([inputs], training=False)

        # These give an error about wrong shapes in a loop, which is to be expected
        expr = ['mu X,b . (X || a)', 'mu X,b . (a || X)']

    def test_functions(self):
        loader = lambda: SingleGraphLoader(self.dataset, epochs=1)
        compiler = self.compilers[0]

        expr = """
        def test(X:b){
        (true || X);and;not
        }
        test(a)
        """
        model = compiler.compile(expr, verbose=True)
        for inputs in loader().load():
            print(model.call([inputs], training=False))

        expr = """
        def test(X:b){
        (true || X);and;not
        }
        test(test(a))
        """
        model = compiler.compile(expr, verbose=True)
        for inputs in loader().load():
            print(model.call([inputs], training=False))

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; |> or) || mu X,b . ((a || X) ; or) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        expected_n_layers = 10
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
                'true': PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool)),
                'false': PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool)),
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
            bottoms={'b': FixPointConfig(1, False)},
            tops={'b': FixPointConfig(1, True)},
            config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8)),
            GNNCompiler(
                psi_functions=psi_dict,
                sigma_functions=sigma_dict,
                phi_functions=phi_dict,
                bottoms={'b': FixPointConfig(1, False)},
                tops={'b': FixPointConfig(1, True)},
                config=CompilationConfig.xaei_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8))]

    def test_edge_expr(self):
        expr = ['a ; |z> or', 'a ; <z| uor', '( (a ; |z> or) || (b ; |z> or) ); or', ' a ; |z> or ; |z> or',
                '(b ; |z> or) || ( c ; |z> or)', 'nu X,b . (X ; |z> or)']
        loaders = [SingleGraphLoader(self.dataset, epochs=1),
                   MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        for loader, compiler in zip(loaders, self.compilers):
            for e in expr:
                model = compiler.compile(e)
                for inputs in loader.load():
                    model.call([inputs], training=False)

    def test_reuse(self):
        expr = 'a || ((a || b);or) || (b ; <z| uor) || mu X,b . (((b ; <z| uor) || X);or) || (a ; not)'
        loaders = [lambda: SingleGraphLoader(self.dataset, epochs=1),
                   lambda: MultipleGraphLoader(self.dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
        expected_n_layers = 11
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
                'true': PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool)),
                'false': PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool)),
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
            bottoms={'b': FixPointConfig(1, False)},
            tops={'b': FixPointConfig(1, True)},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8)),
            GNNCompiler(
                psi_functions=FunctionDict({'a': PsiLocal(
                    lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
                    'b': PsiLocal(
                        lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
                    'c': PsiLocal(
                        lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
                    'true': PsiLocal(lambda x: tf.ones((tf.shape(x)[0], 1), dtype=tf.bool)),
                    'false': PsiLocal(lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.bool)),
                    'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
                    'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
                    'not': PsiLocal(lambda x: tf.math.logical_not(x)),
                    'id': PsiLocal(lambda x: x),
                    'one': PsiLocal(tf.ones_like),
                    'min': PsiLocal(lambda x: x[:, :1] - x[:, 1:]),
                    'gsum': PsiGlobal(multiple_op=tf.math.segment_sum)}),
                sigma_functions=sigma_dict,
                phi_functions=FunctionDict({}),
                bottoms={'b': FixPointConfig(1, False)},
                tops={'b': FixPointConfig(1, True)},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8))]

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
