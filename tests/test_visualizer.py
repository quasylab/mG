import tensorflow as tf
import numpy as np
import os

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, Sigma, FunctionDict, EdgeConfig
from libmg import SingleGraphLoader, MultipleGraphLoader
from libmg import GNNCompiler, CompilationConfig, NodeConfig
from libmg import Dataset
from libmg.functions import Constant
from libmg.visualizer import print_layer


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


class BaseTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset_only_nodes = TestDataset(n=1, edges=False)
        self.dataset_nodes_and_edges = TestDataset(n=1, edges=True)
        self.dataset_multiple_graphs = TestDataset(n=2, edges=False)
        psi_dict_lambdas = FunctionDict({'a': PsiLocal(
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
        sigma_dict_lambdas = FunctionDict({
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))})

        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions=FunctionDict({}),
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions=FunctionDict({}),
                config=CompilationConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions=FunctionDict({}),
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))]

    def tearDown(self):
        for file in os.listdir("."):
            if file.endswith(".html"):
                os.remove(file)


    def test_visualizer_only_nodes(self):
        expr = 'a || b'
        compiler = self.compilers[0]
        loader = SingleGraphLoader(self.dataset_only_nodes, epochs=1)
        model = compiler.compile(expr)
        for inputs in loader.load():
            print_layer(model, inputs, layer_idx=-1, open_browser=False)
            print_layer(model, inputs, layer_name='a', open_browser=False)
            print_layer(model, inputs, layer_name='(a)', open_browser=False)

    def test_visualizer_nodes_and_edges(self):
        expr = 'a || b'
        compiler = self.compilers[1]
        loader = SingleGraphLoader(self.dataset_nodes_and_edges, epochs=1)
        model = compiler.compile(expr)
        for inputs in loader.load():
            print_layer(model, inputs, layer_idx=-1, open_browser=False)
            print_layer(model, inputs, layer_name='a', open_browser=False)
            print_layer(model, inputs, layer_name='(a)', open_browser=False)

    def test_visualizer_multiple_graphs(self):
        expr = 'a || b'
        compiler = self.compilers[2]
        loader = MultipleGraphLoader(self.dataset_multiple_graphs, epochs=1, node_level=True, batch_size=2)
        model = compiler.compile(expr)
        for inputs in loader.load():
            print_layer(model, inputs, layer_idx=-1, open_browser=False)
            print_layer(model, inputs, layer_name='a', open_browser=False)
            print_layer(model, inputs, layer_name='(a)', open_browser=False)