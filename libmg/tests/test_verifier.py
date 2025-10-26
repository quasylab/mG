from libmg import Dataset, Phi, Sigma, PsiLocal, PsiGlobal, CompilerConfig, NodeConfig, EdgeConfig, MGCompiler, SingleGraphLoader, MultipleGraphLoader, MGExplainer
from scipy.sparse import coo_matrix
from spektral.data import Graph
from spektral.utils import gcn_filter
import numpy as np
import tensorflow as tf

from libmg.verifier.lirpa_domain import interpreter, run_abstract_model, check_soundness
from libmg.verifier.graph_abstraction import AbstractionSettings, NoAbstraction, EdgeAbstraction, BisimAbstraction



class DatasetTest(Dataset):
    g1 = (np.array([[0.5, 0.7, 1], [-0.5, 1, 0.5], [3.1, 2.3, 4], [1.1, 1.3, 1.4], [0.1, 0, 0.2]], dtype=np.float32),
          coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1], ([0, 0, 1, 1, 2, 2, 3, 4], [1, 2, 2, 3, 1, 3, 4, 1])),
                     shape=(5, 5), dtype=np.float32),
          np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.uint8))

    def __init__(self, edges=False, labels=False, **kwargs):
        self.edges = edges
        self.labels = labels
        super().__init__("libmg_test_dataset", **kwargs)

    def read(self):
        graphs = []
        x1, a1, y1 = self.g1
        g1 = Graph(x1, a1, None, y1)
        graphs.append(g1)
        return graphs


def preprocess_gcn_mg(graph):
    new_a = gcn_filter(graph.a)
    graph.e = np.expand_dims(new_a.data, axis=-1)
    new_a.data = np.ones_like(new_a.data)
    graph.a = new_a.tocoo()
    return graph

class CastTo:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, graph):
        if graph.x is not None and graph.x.dtype != self.dtype:
            graph.x = graph.x.astype(self.dtype)
        if graph.a is not None and graph.a.dtype != self.dtype:
            graph.a = graph.a.astype(self.dtype)
        if graph.e is not None and graph.e.dtype != self.dtype:
            graph.e = graph.e.astype(self.dtype)
        if graph.y is not None and graph.y.dtype != self.dtype:
            graph.y = graph.y.astype(self.dtype)
        return graph





def get_gcn(dataset, expr):
    # Define mg model
    prod = Phi(lambda i, e, j: i * e)
    sm = Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))
    dense = PsiLocal.make_parametrized('dense', lambda channels: tf.keras.layers.Dense(int(channels), activation='relu', use_bias=True))
    lin = PsiLocal.make_parametrized('dense', lambda channels: tf.keras.layers.Dense(int(channels), activation=None, use_bias=True))
    out = PsiLocal.make('out', tf.keras.layers.Dense(dataset.n_labels, activation='linear', use_bias=True))
    sum_pool = PsiGlobal(single_op=lambda x: tf.reduce_sum(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='SumPooling')
    mean_pool = PsiGlobal(single_op=lambda x: tf.reduce_mean(x, axis=0, keepdims=False), multiple_op=lambda x, i: tf.math.segment_sum(x, i), name='MeanPooling')
    if len(dataset) > 1:
        config = CompilerConfig.xaei_config(NodeConfig(tf.float32, dataset.n_node_features), EdgeConfig(tf.float32, 1), tf.float32, {'float': 0.000001})
    else:
        config = CompilerConfig.xae_config(NodeConfig(tf.float32, dataset.n_node_features), EdgeConfig(tf.float32, 1), tf.float32, {'float': 0.000001})
    compiler = MGCompiler({'dense': dense, 'out': out, 'lin': lin, 'sum': sum_pool, 'mean': mean_pool}, {'+': sm}, {'x': prod}, config)
    model = compiler.compile(expr)
    return model


def get_abstract_model(model, abs_settings):
    interpreter.set_concrete_layers(model.mg_layers)
    interpreter.set_graph_abstraction(abs_settings.graph_abstraction)
    return interpreter.run(model.expr)

def print_bounds(lb, ub, pred, truth):
    pred = pred[0]
    pred_classes = np.argmax(pred.numpy(), axis=1)
    truth_classes = np.argmax(truth.numpy(), axis=1)
    # lb = lb[0] if lb.shape[0] == 1 else lb
    # ub = ub[0] if ub.shape[0] == 1 else ub
    n_nodes = lb.shape[0]
    n_classes = lb.shape[1]  # eq. to n_node_features
    for i in range(n_nodes):
        print(f'Node {i} top-1 prediction {pred_classes[i]} ground-truth {truth_classes[i]}')
        for j in range(n_classes):
            indicator = '(ground-truth)' if j == truth_classes[i] else ''
            print('f_{j}(x_0): {l:8.3f} <= {p:8.3f} <= {u:8.3f} {ind}'.format(
                j=j, l=lb[i][j].item(), u=ub[i][j].item(), p=pred[i][j], ind=indicator))
    print()



# Node task
def test_verifier():
    dataset = DatasetTest(transforms=[preprocess_gcn_mg, CastTo(np.float32)])

    model = get_gcn(dataset, '<x|+ ; dense[32] ; <x|+ ; out')

    # abs_settings = AbstractionSettings(0.1, 0.1, NoAbstraction(optimized_gcn=True), 'backward')
    abs_settings = AbstractionSettings(0, 0, BisimAbstraction('fw'), 'backward')
    abstract_model = get_abstract_model(model, abs_settings)
    # run_node_task(model, abstract_model, dataset, abs_settings)

    for concrete_graph_np, concrete_graph_tf in zip(dataset, MultipleGraphLoader(dataset, node_level=True, epochs=1, shuffle=False).load()):
        ### Setting up graph
        abs_x, abs_a, abs_e = abs_settings.abstract(concrete_graph_np)

        ### Run
        abs_lb, abs_ub = run_abstract_model(abstract_model, abs_x, abs_a, abs_e, abs_settings.algorithm)

        ### Concretize
        lb, ub = abs_settings.concretize(abs_lb, abs_ub)

        (conc_x, conc_a, conc_e, _), y = concrete_graph_tf

        pred = model((conc_x, conc_a, conc_e))
        print(pred)
        print(lb, ub)
        check_soundness(pred, lb, ub)
        print_bounds(lb, ub, pred, y)