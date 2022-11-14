import os
import tensorflow as tf

from compiler import CompilationConfig, GNNCompiler, FixPointConfig, Bottom, Top
from examples.CTL.CTL import build_model, Max, true, false, Not, And, Or, p3
from layers import PsiLocal
from loaders.multiple_graph_loader import MultipleGraphLoader
from loaders.single_graph_loader import SingleGraphLoader
from sources.examples.CTL.datasets.pnml_kripke_dataset import PetriNetDataset, MCCTypes
from sources.examples.CTL.datasets.random_kripke_dataset import RandomKripkeDataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"


class CudaTest(tf.test.TestCase):
    def setUp(self):
        super(CudaTest, self).setUp()

    def test_cuda(self):
        self.assertEqual(tf.test.is_built_with_cuda(), True)


class TrainableTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        trainable_f = PsiLocal(tf.keras.layers.Dense(2))
        float_to_bool = PsiLocal(lambda x: tf.where(tf.reduce_any(tf.less(x, 1), axis=1, keepdims=True),
                                                    tf.constant([False]),
                                                    tf.constant([True])))
        self.formulae = ['a']
        self.atomic_propositions = ['a', 'b', 'c']
        self.dataset = RandomKripkeDataset(1, 1000, 1, self.atomic_propositions,
                                           self.formulae, name='Dataset_Trainable_Test2', skip_model_checking=False,
                                           probabilistic=False)
        self.compiler = GNNCompiler(psi_functions={'true': true, 'false': false,
                                                   'not': Not, 'and': And, 'or': Or, 'tr': trainable_f,
                                                   'tb': float_to_bool},
                                    sigma_functions={'or': Max},
                                    phi_functions={'p3': p3},
                                    bottoms={'b': FixPointConfig(Bottom(1, False))},
                                    tops={'b': FixPointConfig(Top(1, True))},
                                    config=CompilationConfig.xa_config(tf.uint8, 1, tf.uint8))

    def debug(self):
        inputs = tf.keras.Input(shape=1, dtype=tf.uint8)
        conv = tf.keras.layers.Dense(1)
        outputs = conv(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile()
        model.summary()
        print(model.call(tf.constant([1], dtype=tf.uint8), training=False))

    def test_trainable(self):
        expr = 'tr'
        model = self.compiler.compile(expr, verbose=True)
        d_loader = SingleGraphLoader(self.dataset, epochs=1)
        for inputs, y in d_loader.load():
            outputs = model.call(inputs, training=False)
            print(outputs)


if __name__ == '__main__':
    tf.test.main()
