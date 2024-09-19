import shutil

import tensorflow as tf
import os

from libmg.data.loaders import SingleGraphLoader, MultipleGraphLoader
from libmg.tests.test_compiler import setup_test_datasets
from libmg.visualizer.visualizer import print_graph, print_layer


class TestVisualizer(tf.test.TestCase):

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

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        datasets, compilers = setup_test_datasets(use_labels=True)
        xa_dataset, xai_dataset, xae_dataset, xaei_dataset = datasets
        xa_compiler, xai_compiler, xae_compiler, xaei_compiler = compilers
        cls.compilers = [xa_compiler, xai_compiler, xae_compiler, xaei_compiler]
        cls.datasets = [xa_dataset, xai_dataset, xae_dataset, xaei_dataset]

    def test_print_layer(self):
        expr = 'a || ((a || b);or) || (b ; |> or) || (fix X = false in ((a || X) ; or)) || (a ; not)'
        loaders = [SingleGraphLoader(self.datasets[0], epochs=1),
                   MultipleGraphLoader(self.datasets[1], batch_size=2, shuffle=False, epochs=1)] + \
                  [SingleGraphLoader(self.datasets[2], epochs=1),
                   MultipleGraphLoader(self.datasets[3], batch_size=2, shuffle=False, epochs=1)]
        for loader, compiler, filename in zip(loaders, self.compilers, ['xa', 'xai', 'xae', 'xaei']):
            model = compiler.compile(expr)
            for inputs, y in loader.load():
                for engine in ['pyvis', 'cosmo']:
                    print_layer(model, inputs, labels=y, layer_idx=-1, filename='test_labels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=None, layer_idx=-1, filename='test_nolabels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=y, layer_idx=3, filename='test_labels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=None, layer_idx=3, filename='test_nolabels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=y, layer_name='a || b', filename='test_labels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=None, layer_name='a || b', filename='test_nolabels_' + filename, open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=y, layer_name='(fix X = false in ((a || X) ; or))', filename='test_labels_' + filename,
                                open_browser=False, engine=engine)
                    print_layer(model, inputs, labels=None, layer_name='(fix X = false in ((a || X) ; or))', filename='test_nolabels_' + filename,
                                open_browser=False, engine=engine)

    def test_print_graph(self):
        xaei_dataset = self.datasets[-1]
        for i, graph in enumerate(xaei_dataset):
            for engine in ['pyvis', 'cosmo']:
                print_graph(graph, show_labels=False, open_browser=False, filename='test_nolabels_' + str(i), engine=engine)
                print_graph(graph, show_labels=True, open_browser=False, filename='test_labels_' + str(i), engine=engine)
