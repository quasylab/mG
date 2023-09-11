from lark.exceptions import VisitError
from lark.visitors import Interpreter
import tensorflow as tf
from scipy.sparse import coo_matrix
from spektral.data import Graph
from multiprocessing.pool import ThreadPool

from libmg.functions import PsiLocal, Phi, Sigma, Constant
from libmg.compiler import GNNCompiler, Context
from libmg.grammar import mg_parser, mg_reconstructor
from libmg.utils import unpack_inputs
from libmg.visualizer import print_graph


def explanation_nodes(explanation):
    return tf.squeeze(tf.where(explanation), axis=-1).numpy().tolist()


def get_original_ids_func(explanation):
    sorted_node_ids = sorted(explanation_nodes(explanation))
    return lambda x: sorted_node_ids[x]


def make_graph(explanation, hierarchy, old_graph, labels):
    node_feats, adj, edge_feats, _ = unpack_inputs(old_graph)

    new_node_feats = tf.boolean_mask(node_feats, explanation).numpy()
    new_labels = tf.boolean_mask(labels, explanation).numpy() if labels is not None else None

    nodes = set(explanation_nodes(explanation))
    edges = adj.indices.numpy().tolist()
    with ThreadPool() as pool:
        edge_mask = pool.map(lambda edge: edge[0] in nodes and edge[1] in nodes, edges)
    new_adj = tf.sparse.retain(adj, edge_mask)
    new_adj = coo_matrix((new_adj.values.numpy(), (new_adj.indices.numpy()[:, 0], new_adj.indices.numpy()[:, 1])),
                         shape=adj.shape)

    new_edge_feats = tf.boolean_mask(edge_feats, edge_mask).numpy() if edge_feats is not None else None

    hierarchy = tf.cast(hierarchy[explanation], dtype=tf.int64).numpy().tolist()

    new_graph = Graph(x=new_node_feats, a=new_adj, e=new_edge_feats, y=new_labels, hierarchy=hierarchy)

    return new_graph


# TODO: maybe remove the context and just use the dictionaries (to resolve after ite is decided)
class ExplainerMG(Interpreter):
    INF = 1e38
    localize_node = lambda y: PsiLocal(
        lambda x: tf.one_hot(indices=[int(y)], depth=tf.shape(x)[0], axis=0, on_value=0, off_value=ExplainerMG.INF,
                             dtype=tf.float32))
    id = PsiLocal(lambda x: x)
    proj3 = Phi(lambda i, e, j: j)
    or_agg = Sigma(lambda m, i, n, x: tf.minimum(tf.math.unsorted_segment_min(m, i, n) + 1, x))
    or_fun = PsiLocal(lambda x: tf.math.reduce_min(x, axis=1, keepdims=True))
    all_nodes_expr = 'fix X = id in (((X;|p3>or) || (X;<p3|or));or)'

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.query_node = None
        self.context = Context()
        self.compiler = GNNCompiler(
            psi_functions={'node': ExplainerMG.localize_node, 'id': ExplainerMG.id, 'or': ExplainerMG.or_fun},
            sigma_functions={'or': ExplainerMG.or_agg},
            phi_functions={'p3': ExplainerMG.proj3},
            config=model.config)

    def explain(self, query_node, inputs, labels, open_browser=True):
        # Build the model
        self.query_node = query_node
        self.context.clear()
        try:
            right_branch = self.visit(mg_parser.parse(self.model.expr))
        except VisitError:
            right_branch = mg_parser.parse(ExplainerMG.all_nodes_expr)
        left_branch = mg_parser.parse('node[' + str(self.query_node) + ']')
        explainer_tree = mg_parser.parse('left ; right')
        explainer_tree.children[0] = left_branch
        explainer_tree.children[1] = right_branch
        explainer_expr = mg_reconstructor.reconstruct(explainer_tree)
        explainer_model = self.compiler.compile(explainer_expr)

        # Run the model
        hierarchy = tf.squeeze(explainer_model.call(inputs))
        explanation = tf.math.less(hierarchy, ExplainerMG.INF)
        graph = make_graph(explanation, hierarchy, inputs, labels)
        print_graph(graph, node_names_func=get_original_ids_func(explanation), hierarchical=True, show_labels=True,
                    open_browser=open_browser)

    def get_layer(self, name):
        return self.model.mg_layers.get(hash(self.context.get(name)))

    def atom_op(self, tree):
        layer = self.get_layer(tree)
        if layer is None:
            new_op = tree
        elif isinstance(layer.psi, PsiLocal):
            new_op = mg_parser.parse('id')
        else:  # not local neither a variable
            raise VisitError('atom_op', tree, 'Nonlocal psi function')
        return new_op

    def lhd(self, _):
        new_op = mg_parser.parse('|p3>or')
        return new_op

    def rhd(self, _):
        new_op = mg_parser.parse('<p3|or')
        return new_op

    def composition(self, tree):
        left, right = tree.children
        phi = self.visit(left)
        self.context.push(left)
        psi = self.visit(right)
        self.context.pop()
        new_op = tree.copy()
        new_op.children = [phi, psi]
        return new_op

    def parallel(self, tree):
        children = self.visit_children(tree)
        new_op = mg_parser.parse('SUBST;or')
        new_op.children[0] = tree.copy()
        new_op.children[0].children = children
        return new_op

    def ite(self, tree):
        '''
        layer = self.get_layer(tree)
        new_op = mg_parser.parse('left || right')
        new_op.children[0] = tree.children[0]
        if layer.branch is None:
            raise ValueError("Model was not run!")
        if layer.branch.numpy():
            new_op.children[1] = tree.children[1]
        else:
            new_op.children[1] = tree.children[2]
        return self.visit(new_op)
        '''
        raise VisitError('ite', tree, 'If-Then-Else expression')

    def __default__(self, tree):  # local var expr, fun def, fun call, fix, repeat
        new_op = tree.copy()
        new_op.children = self.visit_children(tree)
        return new_op
