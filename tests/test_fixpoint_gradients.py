import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
import numpy as np
from keras import Sequential, Input, Model
from keras.layers import Dense

from scipy.sparse import coo_matrix
from spektral.data import Graph
from libmg import PsiLocal, PsiGlobal, Sigma, Phi
from libmg import SingleGraphLoader, MultipleGraphLoader
from libmg import GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig
from libmg import Dataset
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


def base_tester(dataset, compilers, expressions):
    loaders = [SingleGraphLoader(dataset, epochs=1),
               MultipleGraphLoader(dataset, node_level=True, batch_size=1, shuffle=False, epochs=1)]
    for loader, compiler in zip(loaders, compilers):
        for e in expressions:
            model = compiler.compile(e)
            for inputs in loader.load():
                model.call([inputs], training=False)


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
            '0': Constant(tf.constant(0, dtype=tf.float32)),
            'true': Constant(tf.constant(True)),
            'false': Constant(tf.constant(False)),
            'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
            'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
            'not': PsiLocal(lambda x: tf.math.logical_not(x)),
            'id': PsiLocal(lambda x: x),
            '%': PsiLocal(lambda x: tf.cast(x, dtype=tf.float32)),
            '+': PsiLocal(lambda x: tf.reduce_sum(x, axis=1, keepdims=True)),
            'eq': PsiLocal(lambda x: tf.equal(x[:, :1], x[:, 1:])),
            'pr1': PsiLocal(lambda x: x[:, 1:]),
            'le': PsiLocal(lambda x: x < 2),
            'div': PsiLocal(lambda x: x / 2),
            'add1': PsiLocal(lambda x: x + 1),
            'sub1': PsiLocal(lambda x: x - 1),
            'dense': PsiLocal(tf.keras.layers.Dense(1, activation='linear')),
            'norm': PsiLocal(tf.keras.layers.BatchNormalization())}
        self.psi_dict_lambdas = psi_dict_lambdas
        sigma_dict_lambdas = {
            '*': Sigma(lambda m, i, n, x: tf.math.segment_prod(m, i)),
            'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
            'uor': Sigma(
                lambda m, i, n, x: tf.cast(tf.math.unsorted_segment_max(tf.cast(m, tf.uint8), i, n),
                                           tf.bool))}
        self.sigma_dict_lambdas = sigma_dict_lambdas
        self.compilers = [GNNCompiler(
            psi_functions=psi_dict_lambdas,
            sigma_functions=sigma_dict_lambdas,
            phi_functions={},
            config=CompilationConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {'float': (0.001, 'iter')})),
            GNNCompiler(
                psi_functions=psi_dict_lambdas,
                sigma_functions=sigma_dict_lambdas,
                phi_functions={},
                config=CompilationConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {}))
        ]

    def test_fix(self):
        expr = 'fix X = 0 in ((X;dense) || (id;%));+'
        loader = SingleGraphLoader(self.dataset)
        compiler = self.compilers[0]
        model = compiler.compile(expr, verbose=True)
        # x_in = Input(shape=(1,))
        # a_in = Input(shape=(None,))
        # dense = tf.keras.layers.Dense(1)(x_in)
        # model = Model(inputs=[x_in, a_in], outputs=dense)
        model.compile(optimizer='sgd', loss='mse')
        model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100)
        x, y = loader.load().__iter__().__next__()
        print(model.call(x))


if __name__ == '__main__':
    tf.test.main()