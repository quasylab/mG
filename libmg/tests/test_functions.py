from __future__ import annotations
import pytest
import tensorflow as tf
from libmg.compiler.functions import FunctionDict, make_uoperator, make_boperator, make_koperator, Phi, Sigma, PsiNonLocal, PsiGlobal, Constant, Pi, PsiLocal


class TestFunctionDict:
    @pytest.mark.parametrize('key, output', [('add', ('add', None)), ('sub', ('sub', None)), ('addk[1]', ('addk', '1'))])
    def test_parse_key(self, key, output):
        assert FunctionDict.parse_key(key) == output

    @pytest.mark.parametrize('func, output', [('add', False), ('add_op1', True), ('add_op2', True), ('add_opk', True)])
    def test_is_operator(self, func, output):
        my_dict = FunctionDict({'add': PsiLocal(lambda x: tf.math.add(x, 1)),
                                'add_op1': make_uoperator(lambda x: tf.math.add(x, 1)),
                                'add_op2': make_boperator(tf.math.add),
                                'add_opk': make_koperator(tf.math.add_n)})
        assert my_dict.is_operator(func) == output

    @pytest.mark.parametrize('func', ['add', 'sub', 'addk[1]'])
    def test_lambda_interface(self, func):
        my_dict = FunctionDict({'add': PsiLocal(lambda x: tf.math.add(x, 1)),
                                'sub': PsiLocal(lambda x: tf.math.subtract(x, 1)),
                                'addk': lambda k: PsiLocal(lambda x: tf.math.add(x, int(k)))})
        assert isinstance(my_dict[func], PsiLocal)

    @pytest.mark.parametrize('func', ['add', 'sub', 'addk[1]'])
    def test_make_interface(self, func):
        my_dict = FunctionDict({'add': PsiLocal.make('add', lambda x: tf.math.add(x, 1)), 'sub': PsiLocal.make('sub', lambda x: tf.math.subtract(x, 1)),
                                'addk': PsiLocal.make_parametrized('addk', lambda x, y: tf.math.add(y, int(x)))})
        assert isinstance(my_dict[func], PsiLocal)

    @pytest.mark.parametrize('func', ['add', 'sub', 'addk[1]'])
    def test_subclass_interface(self, func):
        class Add(PsiLocal):
            def func(self, x):
                return tf.math.add(x)

        class Sub(PsiLocal):
            def func(self, x):
                return tf.math.subtract(x)

        class Addk(PsiLocal):
            def __init__(self, y, **kwargs):
                self.y = int(y)
                super().__init__(**kwargs)

            def func(self, x):
                return tf.math.add(x, self.y)

        my_dict = FunctionDict({'add': Add, 'sub': Sub, 'addk': Addk})
        assert isinstance(my_dict[func], PsiLocal)


class TestFunction:
    @pytest.mark.parametrize('cls, f', [(PsiLocal, lambda x: x), (Phi, lambda i, e, j: j), (Sigma, lambda m, i, n, x: tf.math.segment_max(m, i)),
                                        (PsiLocal, tf.keras.layers.Dense(1)), (Phi, tf.keras.layers.Dense(1)), (Sigma, tf.keras.layers.Dense(1))])
    def test_make(self, cls, f):
        assert isinstance(cls.make('TestFunction', f)(), cls)

    @pytest.mark.parametrize('cls, f', [(PsiLocal, lambda y: lambda x: x + y), (Phi, lambda y: lambda i, e, j: j * int(y)),
                                        (Sigma, lambda y: lambda m, i, n, x: tf.math.segment_max(m, i) + int(y)),
                                        (PsiLocal, lambda y, x: x + y), (Phi, lambda y, i, e, j: j * int(y)),
                                        (Sigma, lambda y, m, i, n, x: tf.math.segment_max(m, i) + int(y))])
    def test_make_parametrized(self, cls, f):
        assert isinstance(cls.make_parametrized('TestFunction', f)('1'), cls)


class TestPsiNonLocal(tf.test.TestCase):
    def test_make(self):
        for args in [(PsiNonLocal, lambda x: x, lambda x: x), (PsiNonLocal, lambda x: x, None), (PsiNonLocal, None, lambda x: x),
                     (PsiGlobal, lambda x: tf.reduce_sum(x, axis=0, keepdims=False), lambda x, i: tf.math.segment_sum(x, i)),
                     (PsiGlobal, lambda x: tf.reduce_sum(x, axis=0, keepdims=False), None),
                     (PsiGlobal, None, lambda x, i: tf.math.segment_sum(x, i)),
                     (PsiNonLocal, tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)),
                     (PsiNonLocal, tf.keras.layers.Dense(1), None),
                     (PsiNonLocal, None, tf.keras.layers.Dense(1)),
                     (PsiGlobal, tf.keras.layers.Dense(1), tf.keras.layers.Dense(1)),
                     (PsiGlobal, tf.keras.layers.Dense(1), None),
                     (PsiGlobal, None, tf.keras.layers.Dense(1))
                     ]:
            cls, single_op, multiple_op = args
            assert isinstance(cls.make('TestFunction', single_op, multiple_op)(), cls)

    def test_make_parametrized(self):
        for args in [(PsiNonLocal, lambda y: lambda x: x + y, lambda y: lambda x: x + y),
                     (PsiNonLocal, lambda y: lambda x: x + y, None),
                     (PsiNonLocal, None, lambda y: lambda x: x + y),
                     (PsiGlobal, lambda y: lambda x: tf.reduce_sum(x + y, axis=0, keepdims=False), lambda y: lambda x, i: tf.math.segment_sum(x + y, i)),
                     (PsiGlobal, lambda y: lambda x: tf.reduce_sum(x + y, axis=0, keepdims=False), None),
                     (PsiGlobal, None, lambda y: lambda x, i: tf.math.segment_sum(x + y, i)),
                     (PsiNonLocal, lambda y, x: x + y, lambda y, x: x + y),
                     (PsiNonLocal, lambda y, x: x + y, None),
                     (PsiNonLocal, None, lambda y, x: x + y),
                     (PsiGlobal, lambda y, x: tf.reduce_sum(x + y, axis=0, keepdims=False), lambda y, x, i: tf.math.segment_sum(x + y, i)),
                     (PsiGlobal, lambda y, x: tf.reduce_sum(x + y, axis=0, keepdims=False), None),
                     (PsiGlobal, None, lambda y, x, i: tf.math.segment_sum(x + y, i)),
                     ]:
            cls, single_op, multiple_op = args
            assert isinstance(cls.make_parametrized('TestFunction', single_op, multiple_op)('1'), cls)

    def test_call(self):
        class IncrementPsi(PsiNonLocal):
            def single_graph_op(self, x: tf.Tensor) -> tf.Tensor:
                return x + 1

            def multiple_graph_op(self, x: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
                return x + 1

        for psi in [PsiNonLocal(single_op=lambda x: x + 1, multiple_op=lambda x, i: x + 1),
                    IncrementPsi(),
                    PsiNonLocal.make(single_op=lambda x: x + 1, multiple_op=lambda x, i: x + 1, name='Increment')()]:
            self.assertAllEqual(psi([tf.constant([[1], [2], [3]])]), tf.constant([[2], [3], [4]]))
            self.assertAllEqual(psi([tf.constant([[1], [2], [3]])], tf.constant([0, 0, 1])), tf.constant([[2], [3], [4]]))

    def test_weights(self):
        class MultipleLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.f = tf.keras.layers.Dense(10, activation='linear')

            def __call__(self, x, i):
                return self.f(x)

        class PsiDense(PsiNonLocal):
            def __init__(self):
                super().__init__()
                self.single_dense = tf.keras.layers.Dense(10, activation='linear')
                self.multiple_dense = tf.keras.layers.Dense(10, activation='linear')

            def single_graph_op(self, x: tf.Tensor) -> tf.Tensor:
                return self.single_dense(x)

            def multiple_graph_op(self, x: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
                return self.multiple_dense(x)

        inputs_x = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]
        inputs_i = tf.constant([0, 1])
        for psi in [PsiNonLocal(single_op=tf.keras.layers.Dense(10, activation='linear'), multiple_op=MultipleLayer()),
                    PsiDense(),
                    PsiNonLocal.make(single_op=tf.keras.layers.Dense(10, activation='linear'), multiple_op=MultipleLayer(), name='Dense')()]:
            psi(inputs_x)
            assert psi.weights[0].shape == (5, 10)
            assert psi.weights[1].shape == (10,)
            psi(inputs_x, inputs_i)
            assert psi.weights[2].shape == (5, 10)
            assert psi.weights[3].shape == (10,)


class TestPsiLocal(tf.test.TestCase):
    def test_call(self):
        class IncrementPsi(PsiLocal):
            def func(self, x: tf.Tensor) -> tf.Tensor:
                return x + 1

        for psi in [PsiLocal(lambda x: x + 1),
                    IncrementPsi(),
                    PsiLocal.make('Increment', lambda x: x + 1)()]:
            self.assertAllEqual(psi([tf.constant([[1], [2], [3]])]), tf.constant([[2], [3], [4]]))

    def test_weights(self):
        class PsiLocalDense(PsiLocal):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(10, activation='linear')

            def func(self, x: tf.Tensor) -> tf.Tensor:
                return self.dense(x)

        inputs_x = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]
        for psi in [PsiLocal(tf.keras.layers.Dense(10, activation='linear')),
                    PsiLocalDense(),
                    PsiLocal.make('Dense', tf.keras.layers.Dense(10, activation='linear'))()]:
            psi(inputs_x)
            assert psi.weights[0].shape == (5, 10)
            assert psi.weights[1].shape == (10,)


class TestPsiGlobal(tf.test.TestCase):
    def test_call(self):
        class PoolPsi(PsiGlobal):
            def single_graph_op(self, x: tf.Tensor) -> tf.Tensor:
                return tf.reduce_sum(x, axis=0, keepdims=False)

            def multiple_graph_op(self, x: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
                return tf.math.segment_sum(x, i)

        for psi in [PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i)),
                    PoolPsi(),
                    PsiGlobal.make('Pool', single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i))()]:
            self.assertAllEqual(psi([tf.constant([[1], [2], [3]])]), tf.constant([[6], [6], [6]]))
            self.assertAllEqual(psi([tf.constant([[1], [2], [3], [4]])], tf.constant([0, 0, 1, 1])), tf.constant([[3], [3], [7], [7]]))

    def test_weights(self):

        class MultipleLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.f = tf.keras.layers.Dense(10, activation='linear')

            def __call__(self, x, i):
                return self.f(x)

        class PsiGlobalDense(PsiNonLocal):
            def __init__(self):
                super().__init__()
                self.single_dense = tf.keras.layers.Dense(10, activation='linear')
                self.multiple_dense = tf.keras.layers.Dense(10, activation='linear')

            def single_graph_op(self, x: tf.Tensor) -> tf.Tensor:
                return self.single_dense(x)

            def multiple_graph_op(self, x: tf.Tensor, i: tf.Tensor) -> tf.Tensor:
                return self.multiple_dense(x)

        inputs_x = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]
        inputs_i = tf.constant([0, 1])
        for psi in [PsiGlobal(single_op=tf.keras.layers.Dense(10, activation='linear'), multiple_op=MultipleLayer()),
                    PsiGlobalDense(),
                    PsiGlobal.make(single_op=tf.keras.layers.Dense(10, activation='linear'), multiple_op=MultipleLayer(), name='Dense')()]:
            psi(inputs_x)
            assert psi.weights[0].shape == (5, 10)
            assert psi.weights[1].shape == (10,)
            psi(inputs_x, inputs_i)
            assert psi.weights[2].shape == (5, 10)
            assert psi.weights[3].shape == (10,)


class TestPhi(tf.test.TestCase):
    def test_call(self):
        class EdgeProd(Phi):
            def func(self, src: tf.Tensor, e: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
                return src * e

        for phi in [Phi(lambda i, e, j: i * e),
                    EdgeProd(),
                    Phi.make('Pool', lambda i, e, j: i * e)()]:
            self.assertAllEqual(phi([tf.constant([[1], [2], [2], [3]])], tf.constant([[1], [0], [3], [1]]), [tf.constant([[3], [2], [1], [1]])]),
                                tf.constant([[1], [0], [6], [3]]))

    def test_weights(self):
        class Message(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.f = tf.keras.layers.Dense(10, activation='linear')

            def __call__(self, src, e, tgt):
                return self.f(src)

        class PhiDense(Phi):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(10, activation='linear')

            def func(self, src: tf.Tensor, e: tf.Tensor, tgt: tf.Tensor) -> tf.Tensor:
                return self.dense(src)

        inputs_src = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]
        inputs_e = tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        inputs_tgt = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]
        for phi in [Phi(Message()),
                    PhiDense(),
                    Phi.make('Dense', Message())()]:
            phi(inputs_src, inputs_e, inputs_tgt)
            assert phi.weights[0].shape == (5, 10)
            assert phi.weights[1].shape == (10,)


class TestSigma(tf.test.TestCase):
    def test_call(self):
        class Max(Sigma):
            def func(self, m: tf.Tensor, i: tf.Tensor, n: int, x: tf.Tensor) -> tf.Tensor:
                return tf.math.segment_max(m, i)

        for sigma in [Sigma(lambda m, i, n, x: tf.math.segment_max(m, i)),
                      Max(),
                      Sigma.make('Max', lambda m, i, n, x: tf.math.segment_max(m, i))()]:
            self.assertAllEqual(sigma((tf.constant([[1], [2], [2], [3]]),), tf.constant([0, 0, 1, 2]), 3, [tf.constant([[1], [2], [3]])]),
                                tf.constant([[2], [2], [3]]))

    def test_weights(self):
        class Aggregate(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.f = tf.keras.layers.Dense(10, activation='linear')

            def __call__(self, m, i, n, x):
                return self.f(m)

        class SigmaDense(Sigma):
            def __init__(self):
                super().__init__()
                self.dense = tf.keras.layers.Dense(10, activation='linear')

            def func(self, m: tf.Tensor, i: tf.Tensor, n: int, x: tf.Tensor) -> tf.Tensor:
                return self.dense(m)

        inputs_m = (tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]]),)
        inputs_i = tf.constant([0, 1])
        inputs_n = 5
        inputs_x = [tf.constant([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])]

        for phi in [Sigma(Aggregate()),
                    SigmaDense(),
                    Sigma.make('Dense', Aggregate())()]:
            phi(inputs_m, inputs_i, inputs_n, inputs_x)
            assert phi.weights[0].shape == (5, 10)
            assert phi.weights[1].shape == (10,)


class TestConstant(tf.test.TestCase):
    def test_call(self):
        for c, out in [(Constant(tf.constant([5])), tf.constant([[5], [5], [5]])),
                       (Constant(tf.constant([5, 5])), tf.constant([[5, 5], [5, 5], [5, 5]])),
                       (Constant(tf.constant([True])), tf.constant([[True], [True], [True]])),
                       (Constant(tf.constant([False])), tf.constant([[False], [False], [False]]))]:
            self.assertAllEqual(c([tf.constant([[1], [2], [3]])]), out)


class TestPi(tf.test.TestCase):
    def test_call(self):
        for pi, out in [(Pi(0), tf.constant([[1], [6]])),
                        (Pi(1, 2), tf.constant([[2], [7]])),
                        (Pi(2, -1), tf.constant([[3, 4], [8, 9]])),
                        (Pi(1, 5), tf.constant([[2, 3, 4, 5], [7, 8, 9, 10]]))]:
            self.assertAllEqual(pi([tf.constant([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])]), out)


class TestOperators(tf.test.TestCase):
    def test_make_uoperator(self):
        op = make_uoperator(lambda x: x + 1, 'successor')('1')
        self.assertAllEqual(op([tf.constant([[1], [2], [3]])]), tf.constant([[2], [3], [4]]))

    def test_make_boperator(self):
        op = make_boperator(lambda x, y: x + y, 'add')('2')
        self.assertAllEqual(op([tf.constant([[1], [2], [3]]), tf.constant([[4], [5], [6]])]), tf.constant([[5], [7], [9]]))

    def test_make_koperator(self):
        op = make_koperator(tf.math.add_n, 'add')('5')
        self.assertAllEqual(op([[tf.constant([[1], [6], [11]]),
                                tf.constant([[2], [7], [12]]),
                                tf.constant([[3], [8], [13]]),
                                tf.constant([[4], [9], [14]]),
                                tf.constant([[5], [10], [15]])]]), tf.constant([[15], [40], [65]]))
