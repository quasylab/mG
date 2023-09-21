import tensorflow as tf
import numpy as np
import os

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, Sigma, EdgeConfig
from libmg import SingleGraphLoader
from libmg import GNNCompiler, CompilationConfig, NodeConfig
from libmg import Dataset
from libmg.explainer import ExplainerMG
from libmg.functions import Constant


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
            y = np.array([[2], [4], [8], [2], [2]])
            if self.edges:
                graphs.append(Graph(x, a, e, y))
            else:
                graphs.append(Graph(x, a, y=y))
        return graphs


class BaseTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset_only_nodes = TestDataset(n=1, edges=False)
        self.dataset_nodes_and_edges = TestDataset(n=1, edges=True)
        self.dataset_multiple_graphs = TestDataset(n=2, edges=False)
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
            'eq': PsiLocal(lambda x: tf.equal(x[:, :1], x[:, 1:])),
            'pr1': PsiLocal(lambda x: x[:, 1:]),
            'le': PsiLocal(lambda x: x < 2),
            'add1': PsiLocal(lambda x: x + 1),
            'sub1': PsiLocal(lambda x: x - 1)}
        sigma_dict_lambdas = {
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))}

        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions={},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions={},
                config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))]

    def tearDown(self):
        for file in os.listdir("."):
            if file.endswith(".html"):
                os.remove(file)
                pass

    def test_explainer_only_nodes(self):
        expr = ['a', 'a;true', 'a;true;false', '|>or', '<|uor', '|>or;|>or', 'a || b || true',
                'let X = a in X', 'def f(X){X} in f(a)', 'a ; (if false then true else (false;|>or))',
                'a ; (false || true)', 'repeat X = false in X;|>or for 3',
                'fix X = true in ((a || (X;|>or));and)', 'fix X = true in (if X then true else false)',
                'if (a;|>or) then false else true']
        expr = ['fix X = true in ((a || (X;|>or));and)']
        compiler = self.compilers[0]
        loader = SingleGraphLoader(self.dataset_only_nodes, epochs=1)
        for e in expr:
            model = compiler.compile(e)
            explainer = ExplainerMG(model)
            for inputs, y in loader.load():
                for v in range(5):
                    explainer.explain(v, inputs, y, open_browser=False)


if __name__ == '__main__':
    tf.test.main()
