import shutil
import numpy as np
import tensorflow as tf
import os

from libmg.data.loaders import SingleGraphLoader
from libmg.explainer.explainer import MGExplainer
from libmg.tests.test_compiler import setup_test_datasets


class TestExplainer(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        datasets, compilers = setup_test_datasets(use_labels=True)
        xa_dataset, _, xae_dataset, _ = datasets
        xa_compiler, _, xae_compiler, _ = compilers
        cls.compilers = [xa_compiler, xae_compiler]
        cls.datasets = [xa_dataset, xae_dataset]

    @classmethod
    def tearDownClass(cls):
        for file in os.listdir('.'):
            if file.startswith('graph_test_'):
                if file.endswith('.html'):
                    os.remove(file)
                    pass
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    pass

    def test_explainer(self):
        expressions = [
                       'a', 'a;true', 'a;true;false',
                       '|>or', '<|u^', '|>or;|>or',
                       'a || b || true',
                       'a ; (false || true)',
                       'let X = a in X',
                       'def f(X){X} in f(a)',
                       'if (a;|>or) then false else true',
                       'a ; (if false then true else (false;|>or))',
                       'repeat X = false in X;|>or for 2',
                       'fix X = true in ((a || (X;|>or));and)',
                       'fix X = true in (if X then true else false)',
                       ]
        expected_nodess = [np.array([[1]]), np.array([[1]]), np.array([[1]]), np.array([[1], [2], [4]]), np.array([[1]]), np.array([[1], [2], [4], [1]]),
                           np.array([[1]]), np.array([[1]]), np.array([[1]]), np.array([[1]]), np.array([[1], [2], [4], [1], [1]]),
                           np.array([[1], [2], [4], [1], [1]]),
                           np.array([[1], [2], [4], [1]]),
                           np.array([[1], [2], [4], [1], [1]]),
                           np.array([[1], [2], [4], [1], [1]])]
        loaders = [SingleGraphLoader(self.datasets[0], epochs=1), SingleGraphLoader(self.datasets[1], epochs=1)]
        for expr, expected_nodes in zip(expressions, expected_nodess):
            for loader, compiler in zip(loaders, self.compilers):
                model = compiler.compile(expr)
                explainer = MGExplainer(model)
                for inputs, y in loader.load():
                    for engine in ['pyvis', 'cosmo']:
                        graph = explainer.explain(0, inputs, filename='test_explainer', open_browser=False, engine=engine)
                        np.testing.assert_array_equal(graph.x, expected_nodes)
