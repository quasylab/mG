import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
from spektral.data import Graph


from libmg.compiler.functions import PsiLocal, PsiGlobal, Sigma, Phi
from libmg.data.loaders import SingleGraphLoader, MultipleGraphLoader
from libmg.compiler.compiler import MGCompiler, CompilerConfig, NodeConfig, EdgeConfig
from libmg.data.dataset import Dataset
from libmg.compiler.functions import Constant, make_koperator, make_boperator, make_uoperator


class DatasetTest(Dataset):
    g1 = (np.array([[1], [2], [4], [1], [1]], dtype=np.uint8),
          coo_matrix(([1, 1, 1, 1, 1, 1, 1], ([0, 0, 1, 2, 2, 3, 4], [1, 2, 2, 1, 3, 4, 1])), shape=(5, 5), dtype=np.uint8),
          np.array([[1], [0], [0], [0], [1], [1], [1]], dtype=np.uint8),
          np.array([[2], [4], [8], [2], [2]], dtype=np.uint8))
    g2 = (np.array([[1], [2], [3], [4]], dtype=np.uint8),
          coo_matrix(([1, 1, 1, 1, 1], ([0, 0, 1, 3, 3], [1, 3, 1, 1, 2])), shape=(4, 4), dtype=np.uint8),
          np.array([[0], [1], [0], [1], [0]], dtype=np.uint8),
          np.array([[2], [4], [6], [8]], dtype=np.uint8))

    def __init__(self, multiple=False, edges=False, labels=False, **kwargs):
        self.multiple = multiple
        self.edges = edges
        self.labels = labels
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        x1, a1, e1, y1 = self.g1
        g1 = Graph(x1, a1, e1 if self.edges else None, y1 if self.labels else None)
        graphs.append(g1)
        if self.multiple:
            x2, a2, e2, y2 = self.g2
            g2 = Graph(x2, a2, e2 if self.edges else None, y2 if self.labels else None)
            graphs.append(g2)
        return graphs


def setup_test_datasets(use_labels):
    xa_dataset = DatasetTest(multiple=False, edges=False, labels=use_labels)
    xai_dataset = DatasetTest(multiple=True, edges=False, labels=use_labels)
    xae_dataset = DatasetTest(multiple=False, edges=True, labels=use_labels)
    xaei_dataset = DatasetTest(multiple=True, edges=True, labels=use_labels)
    psi_dict_lambdas = {
        'a': PsiLocal(lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 0, dtype=tf.uint8)), tf.bool)),
        'b': PsiLocal(lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 1, dtype=tf.uint8)), tf.bool)),
        'c': PsiLocal(lambda x: tf.cast(tf.bitwise.bitwise_and(x, tf.constant(2 ** 2, dtype=tf.uint8)), tf.bool)),
        '100': Constant(tf.constant([100.0, 100.0], dtype=tf.float32)),
        'true': Constant(tf.constant(True), name='True'),
        'false': Constant(tf.constant(False), name='False'),
        'and': PsiLocal(lambda x: tf.math.reduce_all(x, axis=1, keepdims=True)),
        'or': PsiLocal(lambda x: tf.math.reduce_any(x, axis=1, keepdims=True)),
        'not': PsiLocal(lambda x: tf.math.logical_not(x)),
        'id': PsiLocal(lambda x: x),
        'eq': PsiLocal(lambda x: tf.equal(x[:, :1], x[:, 1:])),
        'pr1': PsiLocal(lambda x: x[:, 1:]),
        'less10': PsiLocal(lambda x: x < 10),
        'div2': PsiLocal(lambda x: x / 2),
        'add1': PsiLocal(lambda x: x + 1),
        'sub1': PsiLocal(lambda x: x - 1),
        '0': Constant(tf.constant(0, dtype=tf.float32)),
        'dense': PsiLocal(tf.keras.layers.Dense(1, activation='linear')),
        '%': PsiLocal(lambda x: tf.cast(x, dtype=tf.float32)),
        'one': PsiLocal(tf.ones_like),
        'gsum': PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0), multiple_op=lambda x, i: tf.math.segment_sum(x, i)),
        '-': PsiLocal(lambda x: tf.math.subtract(x[:, :1], x[:, 1:])),
        'add': PsiLocal(lambda x: tf.math.add(x[:, :1], x[:, 1:])),
        '+': make_koperator(tf.math.add_n),
        '++': make_boperator(tf.math.add),
        '~': make_uoperator(tf.math.logical_not)}
    sigma_dict_lambdas = {
        'u*': Sigma(lambda m, i, n, x: tf.math.unsorted_segment_prod(m, i, n)),
        '*': Sigma(lambda m, i, n, x: tf.math.segment_prod(m, i)),
        'max': Sigma(lambda m, i, n, x: tf.math.segment_max(m, i)),
        'umin': Sigma(lambda m, i, n, x: tf.math.unsorted_segment_min(m, i, n)),
        'or': Sigma(lambda m, i, n, x: tf.cast(tf.math.segment_max(tf.cast(m, tf.uint8), i), tf.bool)),
    }
    phi_dict_lambdas = {'*': Phi(lambda i, e, j: i * e), 'p1': Phi(lambda i, e, j: i)}

    xa_compiler = MGCompiler(
        psi_functions=psi_dict_lambdas,
        sigma_functions=sigma_dict_lambdas,
        phi_functions=phi_dict_lambdas,
        config=CompilerConfig.xa_config(NodeConfig(tf.uint8, 1), tf.uint8, {'float': 0.000001}))
    xai_compiler = MGCompiler(
        psi_functions=psi_dict_lambdas,
        sigma_functions=sigma_dict_lambdas,
        phi_functions=phi_dict_lambdas,
        config=CompilerConfig.xai_config(NodeConfig(tf.uint8, 1), tf.uint8, {'float': 0.000001}))
    xae_compiler = MGCompiler(
        psi_functions=psi_dict_lambdas,
        sigma_functions=sigma_dict_lambdas,
        phi_functions=phi_dict_lambdas,
        config=CompilerConfig.xae_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {'float': 0.000001}))
    xaei_compiler = MGCompiler(
        psi_functions=psi_dict_lambdas,
        sigma_functions=sigma_dict_lambdas,
        phi_functions=phi_dict_lambdas,
        config=CompilerConfig.xaei_config(NodeConfig(tf.uint8, 1), EdgeConfig(tf.uint8, 1), tf.uint8, {'float': 0.000001}))

    return (xa_dataset, xai_dataset, xae_dataset, xaei_dataset), (xa_compiler, xai_compiler, xae_compiler, xaei_compiler)


class TestCompiler(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        datasets, compilers = setup_test_datasets(use_labels=False)
        xa_dataset, xai_dataset, xae_dataset, xaei_dataset = datasets
        xa_compiler, xai_compiler, xae_compiler, xaei_compiler = compilers
        cls.compilers = [xa_compiler, xai_compiler, xae_compiler, xaei_compiler]
        cls.edge_compilers = [xae_compiler, xaei_compiler]
        cls.datasets = [xa_dataset, xai_dataset, xae_dataset, xaei_dataset]
        cls.edge_datasets = [xae_dataset, xaei_dataset]

    def run_model(self, datasets, compilers, expr, expected):
        if len(datasets) == 2:
            loaders = [SingleGraphLoader(datasets[0], epochs=1),
                       MultipleGraphLoader(datasets[1], batch_size=2, shuffle=False, epochs=1)]
            expected_outputs = [expected[0], tf.concat(expected, 0)]
        else:
            loaders = [SingleGraphLoader(datasets[0], epochs=1),
                       MultipleGraphLoader(datasets[1], batch_size=2, shuffle=False, epochs=1)] + \
                      [SingleGraphLoader(datasets[2], epochs=1),
                       MultipleGraphLoader(datasets[3], batch_size=2, shuffle=False, epochs=1)]
            expected_outputs = [expected[0], tf.concat(expected, 0), expected[0], tf.concat(expected, 0)]
        for loader, compiler, expected_output in zip(loaders, compilers, expected_outputs):
            model = compiler.compile(expr)
            for inputs in loader.load():
                if expected_output.dtype.is_floating:
                    self.assertAllClose(model.call(inputs, training=False), expected_output, atol=0.01, rtol=0.01)
                else:
                    self.assertAllEqual(model.call(inputs, training=False), expected_output)

    def test_psi(self):
        expressions = ['a', 'true', 'add1']
        expected_output = [(tf.constant([[True], [False], [False], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[2], [3], [5], [2], [2]], dtype=tf.uint8),
                            tf.constant([[2], [3], [4], [5]], dtype=tf.uint8)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_image(self):
        expressions = ['<p1|umin', '<|umin', '|p1>max', '|>max']
        expected_output = [(tf.constant([[255], [1], [1], [4], [1]], dtype=tf.uint8),
                            tf.constant([[255], [1], [4], [1]], dtype=tf.uint8)),
                           (tf.constant([[255], [1], [1], [4], [1]], dtype=tf.uint8),
                            tf.constant([[255], [1], [4], [1]], dtype=tf.uint8)),
                           (tf.constant([[4], [4], [2], [1], [2]], dtype=tf.uint8),
                            tf.constant([[4], [2], [0], [3]], dtype=tf.uint8)),
                           (tf.constant([[4], [4], [2], [1], [2]], dtype=tf.uint8),
                            tf.constant([[4], [2], [0], [3]], dtype=tf.uint8)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

        expressions = ['<*|u*', '|*>*']
        expected_output = [(tf.constant([[1], [0], [0], [4], [1]], dtype=tf.uint8),
                            tf.constant([[1], [0], [0], [1]], dtype=tf.uint8)),
                           (tf.constant([[0], [0], [0], [1], [2]], dtype=tf.uint8),
                            tf.constant([[0], [0], [1], [0]], dtype=tf.uint8)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.edge_datasets, self.edge_compilers, expr, expected)

    def test_psi_global(self):
        expressions = ['gsum', '(gsum || one);-']
        expected_output = [(tf.constant([[9], [9], [9], [9], [9]], dtype=tf.uint8),
                            tf.constant([[10], [10], [10], [10]], dtype=tf.uint8)),
                           (tf.constant([[8], [8], [8], [8], [8]], dtype=tf.uint8),
                            tf.constant([[9], [9], [9], [9]], dtype=tf.uint8)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_seq_comp(self):
        expressions = ['one;add1', 'one;add1;add1', 'one;add1;add1;add1', 'gsum ; if less10 then add1 else sub1', 'true ; if not then true else false',
                       'true ; fix X = false in (if true then not else X)', 'true ; fix X = false in (if (((X;not) || true);and) then false else true)',
                       'true ; fix X = false in (if (((X;not) || true);and) then (((X;not) || true);and) else true)']

        expected_output = [(tf.constant([[2], [2], [2], [2], [2]], dtype=tf.uint8),
                            tf.constant([[2], [2], [2], [2]], dtype=tf.uint8)),
                           (tf.constant([[3], [3], [3], [3], [3]], dtype=tf.uint8),
                            tf.constant([[3], [3], [3], [3]], dtype=tf.uint8)),
                           (tf.constant([[4], [4], [4], [4], [4]], dtype=tf.uint8),
                            tf.constant([[4], [4], [4], [4]], dtype=tf.uint8)),
                           (tf.constant([[10], [10], [10], [10], [10]], dtype=tf.uint8),
                            tf.constant([[9], [9], [9], [9]], dtype=tf.uint8)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           ]

        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_par_comp(self):
        expressions = ['a || b || c', 'a || (b || c)', '(a || b) || c', 'sub1 || add1 || one', '(if a then b else false) || b']
        '(fix X = false in (X ; |> or)) || (fix X = true in ((X || a);and))'
        expected_output = [
            (tf.constant([[True, False, False], [False, True, False], [False, False, True], [True, False, False], [True, False, False]], dtype=tf.bool),
             tf.constant([[True, False, False], [False, True, False], [True, True, False], [False, False, True]], dtype=tf.bool)),
            (tf.constant([[True, False, False], [False, True, False], [False, False, True], [True, False, False], [True, False, False]], dtype=tf.bool),
             tf.constant([[True, False, False], [False, True, False], [True, True, False], [False, False, True]], dtype=tf.bool)),
            (tf.constant([[True, False, False], [False, True, False], [False, False, True], [True, False, False], [True, False, False]], dtype=tf.bool),
             tf.constant([[True, False, False], [False, True, False], [True, True, False], [False, False, True]], dtype=tf.bool)),
            (tf.constant([[0, 2, 1], [1, 3, 1], [3, 5, 1], [0, 2, 1], [0, 2, 1]], dtype=tf.uint8),
             tf.constant([[0, 2, 1], [1, 3, 1], [2, 4, 1], [3, 5, 1]], dtype=tf.uint8)),
            (tf.constant([[False, False], [False, True], [False, False], [False, False], [False, False]], dtype=tf.bool),
             tf.constant([[False, False], [False, True], [False, True], [False, False]], dtype=tf.bool)),
        ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_univariate_function_def(self):
        expressions = ["""
                        def test(X){
                        (true || X);and;not
                        } in test(a)
                        """, """
                        def test(X){
                        (true || X);and;not
                        } in test(test(a))
                        """, 'def f(Y){Y} in fix X = false in (if true then f(X) else true)']
        expected_output = [(tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [False], [False], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_multivariate_function_def(self):
        expressions = ["""
                def test(X, Y){
                (Y || X);and;not
                } in test(a, b)
                """, """
                def test(X, Y){
                (Y || X);and;not
                } in test(test(a, b), b)
                """]
        expected_output = [(tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [False], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [True]], dtype=tf.bool)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_let_expr(self):
        expressions = ["""
        let x = add1;add1, y = sub1 in
        (x || y);add
        """]
        expected_output = [(tf.constant([[3], [5], [9], [3], [3]], dtype=tf.uint8),
                            tf.constant([[3], [5], [7], [9]], dtype=tf.uint8))
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_operator(self):
        expressions = ['and(a, b, c)',
                       'and(a, b, or(c, true))',
                       'and(a, b) || true(-(add1, sub1))',
                       '+(add1, sub1, id)',
                       '++(add1, sub1)',
                       '~(true)',
                       """
                        def test(X, Y){
                        (Y || X);and;not
                        } in test(a, and(b, c))
                       """,
                       """
                        def test(X, Y){
                        (Y || and(X, c));and;not
                        } in test(a, b)
                        """,
                       """
                        and(a, def test(X, Y){
                        (Y || X);and;not
                        } in test(b, c))
                        """
                       ]

        expected_output = [(tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [True], [False]], dtype=tf.bool)),
                           (tf.constant([[False, True], [False, True], [False, True], [False, True], [False, True]], dtype=tf.bool),
                            tf.constant([[False, True], [False, True], [True, True], [False, True]], dtype=tf.bool)),
                           (tf.constant([[3], [6], [12], [3], [3]], dtype=tf.uint8),
                            tf.constant([[3], [6], [9], [12]], dtype=tf.uint8)),
                           (tf.constant([[2], [4], [8], [2], [2]], dtype=tf.uint8),
                            tf.constant([[2], [4], [6], [8]], dtype=tf.uint8)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [False], [False], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [False]], dtype=tf.bool)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_ite(self):
        expressions = ['if true then add1 else sub1', 'if false then add1 else sub1', 'if (gsum;less10) then add1 else sub1']
        expected_output = [(tf.constant([[2], [3], [5], [2], [2]], dtype=tf.uint8),
                            tf.constant([[2], [3], [4], [5]], dtype=tf.uint8)),
                           (tf.constant([[0], [1], [3], [0], [0]], dtype=tf.uint8),
                            tf.constant([[0], [1], [2], [3]], dtype=tf.uint8)),
                           (tf.constant([[2], [3], [5], [2], [2]], dtype=tf.uint8),
                            tf.constant([[0], [1], [2], [3]], dtype=tf.uint8)),
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_fix(self):
        expressions = ['fix X = true in X', 'fix X = false in (X ; id)', 'fix X = false in (X ; not; not)',
                       'fix X = false in (X ; |> or)', 'fix X = true in  (X ; |> or)',
                       'fix X = false in ((X || a);or)', 'fix X = true in ((a || (X;|>or));and)',
                       'fix Y = false in (Y ; |> or)', 'fix X = 100 in (X ; div2)',
                       'fix X = true in (a ; ((X || not);and))', 'fix X = false in (((X || a);or) ; ((X || not);or))',
                       'fix X = true in (((a;not) || (X;|>or));and)', 'fix X = true in (let y = a;not in ((y || (X;|>or));and))',
                       'def test(y){ ((a;not) || (y;|>or));and } in fix X = true in (test(X))',
                       'fix X = true in (if X then false else false)',
                       'fix X = false in (if true then X else true)',
                       'fix X = true in (if false then false else X)',
                       'fix X = false in (if true then X else X)',
                       'fix X = true in (if X then X else false)',
                       'fix X = true in (if X then false else X)',
                       'fix X = false in (if X then X else X)',
                       'fix X = false in (fix Y = true in (X || Y);and)',
                       'fix X = false in (fix Y = false in (X || Y || true);or)',
                       'fix X = false in (fix Y = X in Y)', 'fix X = false in (fix Y = X in Y;not;not)',
                       'fix X = true in (fix Y = true in (X || Y || a);and)',
                       'fix Z = true in (fix X = false in (fix Y = false in (X || Y || Z);or))',
                       'fix X = true in ((fix Y = false in (X || Y);and) || (fix Z = true in (X || Z);and));or',
                       'fix X = false in (fix Y = true in ((X;not) || Y);or)',
                       'fix X = false in (fix Y = true in X)',
                       'fix X = false in (if true then (((if true then X else X) || X);or) else true)',
                       'fix X = false in (if true then (((if true then X else false) || X);or) else true)',
                       'fix X = false in (if true then (((if X then false else false) || X);or) else true)',
                       'fix X = false in (if true then ((true || X);or) else true)',
                       'fix X = false in (if true then (X;not;not) else true)',
                       'fix X = false in (if true then (fix Y = true in ((X || Y);and)) else true)',
                       'fix X = false in (if true then (fix Y = X in Y) else true)',
                       'fix X = true in (fix Y = false in ((X;id) || Y);and)'
                       ]
        expected_output = [(tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [False], [False], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=tf.float32),
                            tf.constant([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=tf.float32)),
                           (tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [True], [True], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [True], [False], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [False], [False], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [False], [True], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool))
                           ]
        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_repeat(self):
        expressions = ['repeat X = false in X;not for 3', 'repeat X = false in X;not for 4', 'repeat X = one in X;add1 for 3',
                       'repeat X = false in (if true then (fix Y = true in ((X || Y);and)) else true) for 2']

        expected_output = [(tf.constant([[True], [True], [True], [True], [True]], dtype=tf.bool),
                            tf.constant([[True], [True], [True], [True]], dtype=tf.bool)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool)),
                           (tf.constant([[4], [4], [4], [4], [4]], dtype=tf.uint8),
                            tf.constant([[4], [4], [4], [4]], dtype=tf.uint8)),
                           (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                            tf.constant([[False], [False], [False], [False]], dtype=tf.bool))
                           ]

        for expr, expected in zip(expressions, expected_output):
            self.run_model(self.datasets, self.compilers, expr, expected)

    def test_memoize(self):
        expr = 'a || ((a || b);or) || (b ; |> or) || (fix X = false in ((a || X) ; or)) || (a ; not)'
        mem_expected_n_layers = [11, 12, 12, 13]
        non_mem_expected_n_layers = [15, 16, 16, 17]
        expected_mg_layers = 9
        expected_outputs = (tf.constant(
            [[True, True, True, True, False], [False, True, False, False, True], [False, False, True, False, True], [True, True, False, True, False],
             [True, True, True, True, False]], dtype=tf.bool),
                            tf.constant([[True, True, True, True, False], [False, True, True, False, True], [True, True, False, True, False],
                                         [False, False, True, False, True]], dtype=tf.bool))
        loaders = [SingleGraphLoader(self.datasets[0], epochs=1),
                   MultipleGraphLoader(self.datasets[1], batch_size=2, shuffle=False, epochs=1)] + \
                  [SingleGraphLoader(self.datasets[2], epochs=1),
                   MultipleGraphLoader(self.datasets[3], batch_size=2, shuffle=False, epochs=1)]
        for compiler, loader, mem_expected_n_layer, non_mem_expected_n_layer, expected_output in zip(self.compilers, loaders, mem_expected_n_layers,
                                                                                                     non_mem_expected_n_layers,
                                                                                                     [expected_outputs[0], tf.concat(expected_outputs, 0),
                                                                                                      expected_outputs[0], tf.concat(expected_outputs, 0)]):
            mem_model = compiler.compile(expr, memoize=True)
            non_mem_model = compiler.compile(expr, memoize=False)
            for inputs in loader.load():
                self.assertAllEqual(mem_model.call([inputs], training=False), expected_output)
                self.assertAllEqual(non_mem_model.call([inputs], training=False), expected_output)
            self.assertEqual(len(mem_model.layers), mem_expected_n_layer)
            self.assertEqual(len(mem_model.mg_layers), expected_mg_layers)
            self.assertEqual(len(non_mem_model.layers), non_mem_expected_n_layer)
            self.assertEqual(len(mem_model.mg_layers), expected_mg_layers)

    def test_train_fixpoint(self):
        datasets, _ = setup_test_datasets(use_labels=True)
        expr = 'fix X = 0 in +((X;dense), %)'
        expected_outputs = (tf.constant([[2], [4], [8], [2], [2]], dtype=tf.float32),
                            tf.constant([[2], [4], [6], [8]], dtype=tf.float32))
        loaders = [SingleGraphLoader(datasets[0], epochs=None),
                   MultipleGraphLoader(datasets[1], batch_size=2, shuffle=False, epochs=None)] + \
                  [SingleGraphLoader(datasets[2], epochs=None),
                   MultipleGraphLoader(datasets[3], batch_size=2, shuffle=False, epochs=None)]
        for compiler, loader, expected_output in zip(self.compilers, loaders, [expected_outputs[0], tf.concat(expected_outputs, 0),
                                                                               expected_outputs[0], tf.concat(expected_outputs, 0)]):
            model = compiler.compile(expr)
            model.compile(optimizer='sgd', loss='mse')
            model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100, verbose=0)
            x, y = loader.load().__iter__().__next__()
            self.assertAllClose(model.call(x, training=False), expected_output, atol=0.01, rtol=0.01)

    def test_trace(self):
        for api in ['call', 'predict', 'predict_on_batch']:
            expr = 'fix X = true in ((a || (X;|>or));and)'
            expected_outputs = (tf.constant([[False], [False], [False], [False], [False]], dtype=tf.bool),
                                tf.constant([[False], [False], [False], [False]], dtype=tf.bool))
            loaders = [SingleGraphLoader(self.datasets[0], epochs=1),
                       MultipleGraphLoader(self.datasets[1], batch_size=2, shuffle=False, epochs=1)] + \
                      [SingleGraphLoader(self.datasets[2], epochs=1),
                       MultipleGraphLoader(self.datasets[3], batch_size=2, shuffle=False, epochs=1)]
            for compiler, loader, expected_output in zip(self.compilers, loaders, [expected_outputs[0], tf.concat(expected_outputs, 0),
                                                                                   expected_outputs[0], tf.concat(expected_outputs, 0)]):
                model = compiler.compile(expr)
                try:
                    traced_model, _ = compiler.trace(model, api)
                except ValueError:
                    continue
                if api == 'call':
                    for inputs in loader.load():
                        self.assertAllEqual(model(inputs), expected_output)
                elif api == 'predict':
                    self.assertAllEqual(model.predict(loader.load()), expected_output)
                else:
                    for inputs, in loader.load():
                        self.assertAllEqual(model.predict_on_batch(inputs), expected_output)
