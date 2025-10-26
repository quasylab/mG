from typing import Tuple, Optional

import numpy as np
import scipy.sparse
from lark import v_args
from lark.visitors import Interpreter
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, register_custom_op
import torch
import torch.nn as nn
from auto_LiRPA.operators import Bound, Interval, BoundMul
from libmg.compiler.compiler import Context
from libmg.language import mg_parser, mg_reconstructor
from keras import layers
from tqdm import tqdm

from libmg.verifier.graph_abstraction import GraphAbstraction


# def forward_lub(ml1, mu1, l1, u1, ml2, mu2, l2, u2):
#     mask = torch.flatten(l1[0]) <= torch.flatten(l2[0])
#     expanded_mask = mask[:, None]
#     ml1 = torch.transpose(torch.reshape(ml1, (1, 15, -1,)), 1, 2)
#     ml2 = torch.transpose(torch.reshape(ml2, (1, 15, -1,)), 1, 2)
#     lb = torch.where(expanded_mask, ml1[0], ml2[0])
#     mask = torch.flatten(u1[0]) >= torch.flatten(u2[0])
#     expanded_mask = mask[:, None]
#     mu1 = torch.transpose(torch.reshape(mu1, (1, 15, -1,)), 1, 2)
#     mu2 = torch.transpose(torch.reshape(mu2, (1, 15, -1,)), 1, 2)
#     ub = torch.where(expanded_mask, mu1[0], mu2[0])
#     return torch.reshape(torch.transpose(lb, 0, 1), (1, 15, 5, 3)), torch.reshape(torch.transpose(ub, 0, 1), (1, 15, 5, 3))

# Helper functions
def get_node_labels(x, node_id):
    lb = x[0][0][node_id, :]
    ub = x[1][0][node_id, :]
    return lb, ub


def get_edge_labels(e, edge_id):
    lb = e[0][edge_id, :]
    ub = e[1][edge_id, :]
    return lb, ub


def interval_to_bounded_tensor(lb, ub):
    ptb = PerturbationLpNorm(norm=np.inf, x_L=lb, x_U=ub)
    return BoundedTensor(torch.zeros_like(lb), ptb)


# Phi functions
def conc_phi_product(i, e, j):
    return i * e


def intv_phi_product(i: Tuple[torch.Tensor, torch.Tensor], e: Tuple[torch.Tensor, torch.Tensor], j: Tuple[torch.Tensor, torch.Tensor]):
    x, y = i, e
    r0, r1, r2, r3 = x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1]
    lower = torch.min(torch.min(r0, r1), torch.min(r2, r3))
    upper = torch.max(torch.max(r0, r1), torch.max(r2, r3))
    return lower, upper


def poly_phi_product(i, e, j):
    il, ih = i
    el, eh = e
    argmin = np.argmin([el * il, el * ih, eh * il, eh * ih])
    emin, emax = (el, eh) if argmin <= 1 else (eh, el)
    return emin, emax


# Sigma functions
def conc_sigma_sum(m, x):
    return torch.stack(m).sum(dim=0)


def intv_sigma_sum(m, x):
    lbs = [msg[0] for msg in m]
    ubs = [msg[1] for msg in m]
    return sum(lbs), sum(ubs)


def poly_sigma_sum(m, x):
    lbs = [msg[0] for msg in m]
    ubs = [msg[1] for msg in m]
    return sum(lbs), sum(ubs)


# Psi functions
def make_layer(w, bias, activation):
    in_features = w.shape[0]
    out_features = w.shape[1]
    lin = torch.nn.Linear(in_features, out_features, bias=True)
    lin.weight.data = torch.tensor(w).transpose(0, 1)
    lin.bias.data = torch.tensor(bias)
    if activation == 'relu':
        act: torch.nn.Module = torch.nn.ReLU()
    else:
        act = torch.nn.Identity()
    return lin, act


class Pool(nn.Module):
    def __init__(self, pool_fn, abstraction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_fn = pool_fn
        self.abstraction = abstraction

    def forward(self, x):
        x1 = self.abstraction.handle_pooling(x)
        x2 = self.pool_fn(x1, dim=1, keepdim=True)
        return x2


def make_pooling(pool, abstraction):
    pool_fn = torch.sum if pool == 'sum' else torch.mean
    return Pool(pool_fn, abstraction)


# Least upper bound
def least_upper_bound(lt1, ut1, lt2, ut2):
    olb = torch.minimum(lt1, lt2)
    oub = torch.maximum(ut1, ut2)
    return olb, oub


def backward_lub(ml1, mu1, l1, u1, ml2, mu2, l2, u2):
    mask = torch.flatten(l1[0]) <= torch.flatten(l2[0])
    expanded_mask = mask[:, None, None]
    lb = torch.where(expanded_mask, ml1[0], ml2[0])
    mask = torch.flatten(u1[0]) >= torch.flatten(u2[0])
    expanded_mask = mask[:, None, None]
    ub = torch.where(expanded_mask, mu1[0], mu2[0])
    return lb.unsqueeze(0), ub.unsqueeze(0)


# Abstraction functions
def abstract_x(value: np.ndarray, delta: float = 0, x_L: Optional[np.ndarray] = None, x_U: Optional[np.ndarray] = None) -> BoundedTensor | torch.Tensor:
    tensor = torch.tensor(value).unsqueeze(0)
    xL = torch.tensor(x_L).unsqueeze(0) if x_L is not None else x_L
    xU = torch.tensor(x_U).unsqueeze(0) if x_U is not None else x_U
    if delta != 0 and xL is not None and xU is not None:
        xL = xL - delta
        xU = xU + delta
        perturbation = PerturbationLpNorm(x_L=xL, x_U=xU)
        tensor = BoundedTensor(tensor, perturbation)
    elif delta != 0 or (xL is not None and xU is not None):
        perturbation = PerturbationLpNorm(eps=delta, x_L=xL, x_U=xU)
        tensor = BoundedTensor(tensor, perturbation)
    return tensor


def abstract_e(value: np.ndarray, delta: float = 0, x_L: Optional[np.ndarray] = None, x_U: Optional[np.ndarray] = None) -> BoundedTensor | torch.Tensor:
    tensor = torch.tensor(value)
    xL = torch.tensor(x_L) if x_L is not None else x_L
    xU = torch.tensor(x_U) if x_U is not None else x_U
    if delta != 0 and xL is not None and xU is not None:
        xL = xL - delta
        xU = xU + delta
        perturbation = PerturbationLpNorm(x_L=xL, x_U=xU)
        tensor = BoundedTensor(tensor, perturbation)
    elif delta != 0 or (xL is not None and xU is not None):
        perturbation = PerturbationLpNorm(eps=delta, x_L=xL, x_U=xU)
        tensor = BoundedTensor(tensor, perturbation)
    return tensor


def abstract_adj(mat: scipy.sparse.coo_matrix) -> torch.Tensor:
    rows = mat.row
    cols = mat.col
    data = mat.data.astype(np.int32)
    tensor = torch.tensor(np.array([[rows, cols, data]]))
    return tensor


# Concretization
def run_abstract_model(model, x, a, e, algorithm, verbose=False):
    abs_model = BoundedModule(model, (torch.empty_like(x), a, torch.empty_like(e)), device=a.device, verbose=verbose)
    lb, ub = abs_model.compute_bounds(x=(x, a, e), method=algorithm)
    # print(lirpa_model.save_intermediate())
    return lb.detach()[0], ub.detach()[0]


# Function store
class Transformer:
    def __init__(self, concrete_transformer, interval_transformer, poly_transformer, id_element):
        self.concrete_transformer = concrete_transformer
        self.interval_transformer = interval_transformer
        self.poly_transformer = poly_transformer
        self.id_element = id_element

    def identity_like(self, t):
        tensor = torch.full_like(t, self.id_element)
        return tensor, tensor


fucts = {'x': Transformer(conc_phi_product, intv_phi_product, poly_phi_product, 1.),
         '+': Transformer(conc_sigma_sum, intv_sigma_sum, poly_sigma_sum, 0.)}


# Message-passing procedures
def concrete_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, x, a, e):
    if use_optimized_gcn:
        return concrete_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)
    else:
        return concrete_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)


@torch.no_grad()
def concrete_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    phi = function_store_phi.concrete_transformer
    sigma = function_store_sigma.concrete_transformer
    n_nodes = x.shape[1]
    n_edges = e.shape[0]
    index_targets = a[0][tgt_idx]  # Nodes receiving the message
    index_sources = a[0][src_idx]  # Nodes sending the message (ie neighbors)
    x = x[0]
    # Message
    messages: list[list[torch.Tensor]] = [[] for _ in range(n_nodes)]  # list of lists of messages
    for idx in range(n_edges):
        messages[index_targets[idx]].append(phi(x[index_sources[idx], :], e[idx], x[index_targets[idx], :]))
    # Aggregate
    embeddings = [sigma(m, x[i, :]) for i, m in enumerate(messages)]
    out = torch.stack(embeddings).unsqueeze(0)
    # Update
    return out


@torch.no_grad()
def concrete_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    x = x[0]
    # Message
    return torch.matmul(e.transpose(-1, -2) if src_idx == 0 else e, x)


def interval_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, x, a, e):
    if use_optimized_gcn:
        return interval_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)
    else:
        return interval_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e)


@torch.no_grad()
def interval_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    abs_phi = function_store_phi.interval_transformer
    abs_sigma = function_store_sigma.interval_transformer
    a = a[0][0]  # not an interval
    n_nodes = x[0].shape[1]
    index_targets = a[tgt_idx]  # Nodes receiving the message
    index_sources = a[src_idx]  # Nodes sending the message (ie neighbors)
    edge_status = a[-1]
    messages: list[list[torch.Tensor]] = [[] for _ in range(n_nodes)]  # list of lists of messages

    n_edges = e[0].shape[0]
    # Message
    for idx in range(n_edges):
        certain_edge = True if edge_status[idx] == 1 else False
        if certain_edge is True:
            messages[index_targets[idx]].append(abs_phi(get_node_labels(x, index_sources[idx]), get_edge_labels(e, idx),
                                                        get_node_labels(x, index_targets[idx])))
        else:
            msg = abs_phi(get_node_labels(x, index_sources[idx]), get_edge_labels(e, idx), get_node_labels(x, index_targets[idx]))
            bot = function_store_sigma.identity_like(get_node_labels(x, index_sources[idx])[0])
            lub = least_upper_bound(*msg, *bot)
            messages[index_targets[idx]].append(lub)

    # Aggregate
    embeddings = [abs_sigma(m, get_node_labels(x, i)) for i, m in enumerate(messages)]
    lb, ub = torch.stack([emb[0] for emb in embeddings]), torch.stack([emb[1] for emb in embeddings])
    # Update
    return torch.unsqueeze(lb, 0), torch.unsqueeze(ub, 0)


@torch.no_grad()
def interval_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, x, a, e):
    v = [e, x]

    # This will convert an Interval object to tuple.
    # We need to add perturbation property later.
    vlb, vub = zip(*v)
    v_lb = list(vlb)
    v_ub = list(vub)
    v_lb[0] = v_lb[0].transpose(-2, -1) if src_idx == 0 else v_lb[0]

    # vlb0 = v_lb[0].clone()
    # vlb0[edge_status < 0] = torch.clamp(v_lb[0][edge_status < 0], max=0)
    # v_lb[0] = vlb0

    v_lb[1] = v_lb[1].transpose(-2, -1)
    v_ub[0] = v_ub[0].transpose(-2, -1) if src_idx == 0 else v_ub[0]

    # vub0 = v_ub[0].clone()
    # vub0[edge_status < 0] = torch.clamp(v_ub[0][edge_status < 0], min=0)
    # v_ub[0] = vub0

    v_ub[1] = v_ub[1].transpose(-2, -1)
    # After preprocess the lower and upper bounds, we make them Intervals again.
    v = [Interval.make_interval(bounds[0], bounds[1], bounds[2])
         for bounds in zip(v_lb, v_ub, v)]

    x_l, x_u = v[0][0].unsqueeze(-2), v[0][1].unsqueeze(-2)
    y_l, y_u = v[1][0].unsqueeze(-3), v[1][1].unsqueeze(-3)
    # Reuse the multiplication bounds and sum over results.
    lower, upper = BoundMul.interval_propagate_both_perturbed(*[(x_l, x_u), (y_l, y_u)])
    lower, upper = torch.sum(lower, -1), torch.sum(upper, -1)

    l, u = lower, upper
    return l, u


def bwd_message_passing(src_idx, tgt_idx, function_store_phi, function_store_sigma, use_optimized_gcn, last_lA, last_uA, x, a, e):
    if use_optimized_gcn:
        return bwd_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e)
    else:
        return bwd_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e)


@torch.no_grad()
def bwd_message_passing_general(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
    """ Backward mode bound propagation """
    abs_phi = function_store_phi.poly_transformer
    abs_sigma = function_store_sigma.poly_transformer
    n_nodes = x.output_shape[1]
    n_vars_post = last_lA.shape[0]
    n_vars_pre = last_lA.shape[-2] * last_lA.shape[-1]
    n_node_features_pre = n_vars_pre // n_nodes
    n_node_features_post = n_vars_post // n_nodes
    index_targets = a.forward_value[0][tgt_idx]  # Nodes receiving the message
    index_sources = a.forward_value[0][src_idx]  # Nodes sending the message (ie neighbors)
    edge_status = a.forward_value[0][-1]
    elow, ehigh = e.interval[0], e.interval[1]
    xlow, xhigh = x.interval[0], x.interval[1]

    class NodeValues:
        def __init__(self, xl, xh):
            self.xlow = xl
            self.xhigh = xh
            self.current_nonzero_idx = torch.empty((0,))
            self.indices = {}
            self.lows = {}
            self.highs = {}

        def get_node_values(self, nonzero_idx):
            if not torch.equal(self.current_nonzero_idx, nonzero_idx):
                self.current_nonzero_idx = nonzero_idx
                self.indices, self.lows, self.highs = {}, {}, {}
                for nzi in nonzero_idx:
                    nzi = nzi.item()
                    _node = nzi // n_node_features_pre
                    offset = nzi % n_node_features_pre
                    edge_idx = np.where(index_targets == _node)[0]
                    self.indices[(_node, offset)] = []
                    self.lows[(_node, offset)] = []
                    self.highs[(_node, offset)] = []
                    for idx in edge_idx:
                        certain_edge = True if edge_status[idx] == 1 else False
                        sidx = index_sources[idx]
                        self.indices[(_node, offset)].append(sidx)
                        if certain_edge is True:
                            el, eh = elow[idx], ehigh[idx]
                            src_node = index_sources[idx]
                            tgt_node = index_targets[idx]
                            il, ih = self.xlow[0][src_node][offset], self.xhigh[0][src_node][offset]
                            jl, jh = self.xlow[0][tgt_node][offset], self.xhigh[0][tgt_node][offset]
                            emin, emax = abs_phi((il, ih), (el, eh), (jl, jh))
                            self.lows[(_node, offset)].append(emin)
                            self.highs[(_node, offset)].append(emax)
                        else:
                            el, eh = elow[idx], ehigh[idx]
                            src_node = index_sources[idx]
                            tgt_node = index_targets[idx]
                            il, ih = self.xlow[0][src_node][offset], self.xhigh[0][src_node][offset]
                            jl, jh = self.xlow[0][tgt_node][offset], self.xhigh[0][tgt_node][offset]
                            emin, emax = abs_phi((il, ih), (el, eh), (jl, jh))
                            bot_low, bot_high = function_store_sigma.identity_like(elow[idx])
                            self.lows[(_node, offset)].append(torch.min(emin, bot_low))
                            self.highs[(_node, offset)].append(torch.max(emax, bot_high))

            nonzero_idx = [nnz.item() for nnz in nonzero_idx]
            values_low = torch.zeros((1, n_vars_pre))
            values_high = torch.zeros((1, n_vars_pre))
            for j in nonzero_idx:
                values_low.zero_()
                values_high.zero_()
                node = j // n_node_features_pre
                offset = j % n_node_features_pre
                idx_array = torch.tensor(self.indices[(node, offset)])
                pos_array = idx_array * n_node_features_pre + offset
                values_low[0][pos_array] = torch.tensor(self.lows[(node, offset)])
                values_high[0][pos_array] = torch.tensor(self.highs[(node, offset)])
                yield values_low, values_high

    reshaped_last_la = last_lA.view((n_vars_post, 1, -1))
    reshaped_last_ua = last_uA.view((n_vars_post, 1, -1))
    node_val_obj = NodeValues(xlow, xhigh)
    id_value_low, id_value_high = function_store_sigma.identity_like(reshaped_last_la[0])

    lA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
    uA = torch.empty((last_lA.shape[0], 1, n_nodes * n_node_features_pre))
    node_iterator = tqdm(range(n_nodes))
    for node in node_iterator:
        for v_offset in range(n_node_features_post):
            v = node * n_node_features_post + v_offset
            non_zero_indices_l = torch.nonzero(reshaped_last_la[v], as_tuple=True)[1]
            non_zero_indices_u = torch.nonzero(reshaped_last_ua[v], as_tuple=True)[1]
            if non_zero_indices_u.nelement() != 0 and non_zero_indices_l.nelement() != 0:
                prev_values_la = reshaped_last_la[v][0][non_zero_indices_l]
                prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
                node_val = node_val_obj.get_node_values(non_zero_indices_u)
                rec_value_la = reshaped_last_la[v]
                rec_value_ua = reshaped_last_ua[v]
                emb_low, emb_high = id_value_low, id_value_high
                for coeff_la, coeff_ua, (mat_la, mat_ua) in zip(prev_values_la, prev_values_ua, node_val):
                    if coeff_la >= 0:
                        msg_low = mat_la * coeff_la
                    else:
                        msg_low = mat_ua * coeff_la
                    if coeff_ua >= 0:
                        msg_high = mat_ua * coeff_ua
                    else:
                        msg_high = mat_la * coeff_ua
                    emb_low, emb_high = abs_sigma([(emb_low, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_low, emb_high), (id_value_low, id_value_high)], (rec_value_la, rec_value_ua))
                embeddings_low, embeddings_high = emb_low, emb_high
            elif non_zero_indices_l.nelement() != 0:
                embeddings_high = torch.zeros((1, n_vars_pre))
                prev_values_la = reshaped_last_ua[v][0][non_zero_indices_l]
                node_val = node_val_obj.get_node_values(non_zero_indices_l)
                rec_value_la = reshaped_last_la[v]
                emb_low = id_value_low
                for coeff_la, (mat_la, mat_ua) in zip(prev_values_la, node_val):
                    if coeff_la >= 0:
                        msg_low, msg_high = mat_la * coeff_la, mat_ua * coeff_la
                    else:
                        msg_low, msg_high = mat_ua * coeff_la, mat_la * coeff_la
                    emb_low, emb_high = abs_sigma([(emb_low, emb_low), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_low, emb_low), (id_value_low, id_value_high)], (rec_value_la, rec_value_la))
                embeddings_low = emb_low
            elif non_zero_indices_u.nelement() != 0:
                embeddings_low = torch.zeros((1, n_vars_pre))
                prev_values_ua = reshaped_last_ua[v][0][non_zero_indices_u]
                node_val = node_val_obj.get_node_values(non_zero_indices_u)
                rec_value_ua = reshaped_last_ua[v]
                emb_high = id_value_high
                for coeff_ua, (mat_la, mat_ua) in zip(prev_values_ua, node_val):
                    if coeff_ua >= 0:
                        msg_low, msg_high = mat_la * coeff_ua, mat_ua * coeff_ua
                    else:
                        msg_low, msg_high = mat_ua * coeff_ua, mat_la * coeff_ua
                    emb_low, emb_high = abs_sigma([(emb_high, emb_high), (msg_low, msg_high)], (id_value_low, id_value_high))
                emb_low, emb_high = abs_sigma([(emb_high, emb_high), (id_value_low, id_value_high)], (rec_value_ua, rec_value_ua))
                embeddings_high = emb_high
            else:
                embeddings_low = torch.zeros((1, n_vars_pre))
                embeddings_high = torch.zeros((1, n_vars_pre))
            lA[v] = embeddings_low
            uA[v] = embeddings_high

    lAx = lA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))
    uAx = uA.view((last_lA.shape[0], 1, n_nodes, n_node_features_pre))

    return [(lAx, uAx), (None, None), (None, None)], 0., 0.


@torch.no_grad()
def bwd_message_passing_gcn_optimized(src_idx, tgt_idx, function_store_phi, function_store_sigma, last_lA, last_uA, x, a, e):
    """ Backward mode bound propagation """
    if src_idx == 1:
        e.lower = e.lower.transpose(-1, -2)
        e.upper = e.upper.transpose(-1, -2)

    input_lb = [e.lower, x.lower]
    input_ub = [e.upper, x.upper]

    input_lb[0] = input_lb[0].transpose(-2, -1)
    input_ub[0] = input_ub[0].transpose(-2, -1)
    input_lb[1] = input_lb[1].transpose(-2, -1)
    input_ub[1] = input_ub[1].transpose(-2, -1)

    input_lb[0] = input_lb[0].unsqueeze(-2)
    input_ub[0] = input_ub[0].unsqueeze(-2)
    input_lb[1] = input_lb[1].unsqueeze(-3)
    input_ub[1] = input_ub[1].unsqueeze(-3)

    x_l, _, y_l, y_u = input_lb[0], input_ub[0], input_lb[1], input_ub[1]

    alpha_l, beta_l, gamma_l = y_l, x_l, -y_l * x_l
    alpha_u, beta_u, gamma_u = y_u, x_l, -y_u * x_l

    gamma_l = torch.sum(gamma_l, dim=-1)
    gamma_u = torch.sum(gamma_u, dim=-1)

    @torch.jit.script
    def propagate_A_xy(last_A, alpha_pos, alpha_neg,
                       beta_pos, beta_neg):
        # last_uA has size (batch, spec, output)
        last_A_pos = last_A.clamp(min=0).unsqueeze(-1)
        last_A_neg = last_A.clamp(max=0).unsqueeze(-1)
        # alpha_u has size (batch, spec, output, input)
        # uA_x has size (batch, spec, input).
        A_x = (alpha_pos.transpose(-1, -2).matmul(last_A_pos) +
               alpha_neg.transpose(-1, -2).matmul(last_A_neg)).squeeze(-1)
        # beta_u has size (batch, spec, output, input)
        # uA_y is for weight matrix, with parameter size (output, input)
        # uA_y has size (batch, spec, output, input). This is an element-wise multiplication.
        # TODO (for zhouxing/qirui): generalize multiply_by_A_signs() to calculate A_x,
        # so last_A_pos and last_A_neg are not needed. This saves memory.
        d_pos = beta_pos.contiguous()
        d_neg = beta_neg.contiguous()
        A = last_A.unsqueeze(-1)

        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)

        # Initialize output tensor
        A_new = torch.zeros((A_pos.shape[0], A_pos.shape[1], A_pos.shape[3], A_pos.shape[2]), device=A_pos.device)

        # Loop over i (the summation axis)
        for i in range(A_pos.shape[2]):
            # Extract weights for this i across all j
            w_pos = d_pos[0, i, 0, :]  # shape: [80]
            w_neg = d_neg[0, i, 0, :]  # shape: [80]

            # Extract A_pos and A_neg slice at i
            A_pos_i = A_pos[:, 0, i, :, 0]  # shape: [560, 32]
            A_neg_i = A_neg[:, 0, i, :, 0]  # shape: [560, 32]

            # Reshape for broadcasting: [560, 1, 32, 1]
            A_pos_i = A_pos_i.unsqueeze(1).unsqueeze(-1)
            A_neg_i = A_neg_i.unsqueeze(1).unsqueeze(-1)

            # Reshape weights: [1, 1, 1, 80]
            w_pos = w_pos.view(1, 1, 1, A_pos.shape[2])
            w_neg = w_neg.view(1, 1, 1, A_pos.shape[2])

            # Accumulate weighted contribution
            A_new += A_pos_i * w_pos + A_neg_i * w_neg

        A_y = A_new

        return A_x, A_y

    def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos, alpha_neg, beta_neg, gamma_neg):
        if last_A is None:
            return None, None, 0

        A_x, A_y = propagate_A_xy(
            last_A, alpha_pos, alpha_neg, beta_pos, beta_neg)

        # last_uA has size (batch, spec, output)
        # gamma_u has size (batch, output, 1)
        # ubias has size (batch, spec, 1)
        bias = (torch.einsum('sb...,b...->sb', last_A.clamp(min=0), gamma_pos) + torch.einsum('sb...,b...->sb', last_A.clamp(max=0), gamma_neg))
        return A_x, A_y, bias

    lA_x, lA_y, lbias = _bound_oneside(
        last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
    uA_x, uA_y, ubias = _bound_oneside(
        last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

    results = [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    lA_y = results[0][1][0].transpose(-1, -2) if results[0][1][0] is not None else None
    uA_y = results[0][1][1].transpose(-1, -2) if results[0][1][1] is not None else None
    lA_e = results[0][0][0].transpose(-1, -2) if results[0][0][0] is not None else None
    uA_e = results[0][0][1].transpose(-1, -2) if results[0][0][1] is not None else None

    if isinstance(results[1], tuple):
        lbias = (results[1][0], results[1][1].transpose(-1, -2))
    else:
        lbias = results[1]
    if isinstance(results[2], tuple):
        ubias = (results[2][0], results[2][1].transpose(-1, -2))
    else:
        ubias = results[2]

    # for edge abs only
    # edge_status = a.forward_value[0]
    # uncertain_edges = edge_status < 0
    #
    # pos_mask = uncertain_edges & (lA_e > 0)
    # pos_indices = torch.nonzero(pos_mask, as_tuple=False)
    # pos_i = pos_indices[:, 2]
    # pos_j = pos_indices[:, 3]
    # neg_mask = uncertain_edges & (lA_e < 0)
    # neg_indices = torch.nonzero(neg_mask, as_tuple=False)
    # neg_i = neg_indices[:, 2]
    # neg_j = neg_indices[:, 3]
    #
    # weight_pos = (torch.minimum(e.lower[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_pos = weight_pos[pos_i, pos_j]
    # weight_neg = (torch.maximum(e.lower[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_neg = weight_neg[neg_i, neg_j]
    #
    # lA_e[pos_mask] *= selected_weight_pos
    # lA_e[neg_mask] *= selected_weight_neg
    #
    #
    # pos_mask = uncertain_edges & (uA_e > 0)
    # pos_indices = torch.nonzero(pos_mask, as_tuple=False)
    # pos_i = pos_indices[:, 2]
    # pos_j = pos_indices[:, 3]
    # neg_mask = uncertain_edges & (uA_e < 0)
    # neg_indices = torch.nonzero(neg_mask, as_tuple=False)
    # neg_i = neg_indices[:, 2]
    # neg_j = neg_indices[:, 3]
    #
    # weight_pos = (torch.maximum(e.upper[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_pos = weight_pos[pos_i, pos_j]
    # weight_neg = (torch.minimum(e.upper[0], torch.tensor(0.)) != 0.).float()
    # selected_weight_neg = weight_neg[neg_i, neg_j]
    #
    # uA_e[pos_mask] *= selected_weight_pos
    # uA_e[neg_mask] *= selected_weight_neg

    if src_idx == 1:
        lA_e = lA_e.transpose(-1, -2) if lA_e is not None else None
        uA_e = uA_e.transpose(-1, -2) if uA_e is not None else None

    return [(lA_y, uA_y), (None, None), (lA_e, uA_e)], lbias, ubias


# PREIMAGE
class Pre(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, define the arguments and attributes of the operator.
        "custom::SigmaSum" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Pre', x, a, e, phi_s=phis, sigma_s=sigmas, use_optimized_gcn_s=str(use_optimized_gcn)).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = i * e in this case. """
        return concrete_message_passing(0, 1, fucts[phis], fucts[sigmas], use_optimized_gcn, x, a, e)


class PreImage(nn.Module):
    def __init__(self, phi_f, sigma_f, use_optimized_gcn):
        super().__init__()
        self.phi_f = phi_f
        self.sigma_f = sigma_f
        self.use_optimized_gcn = use_optimized_gcn

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Pre.apply(x, a, e, self.phi_f, self.sigma_f, self.use_optimized_gcn)


class BoundPre(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.phi_f = attr['phi']
        self.sigma_f = attr['sigma']
        self.use_optimized_gcn = attr['use_optimized_gcn'] == 'True'
        self.requires_input_bounds = [0]

    def forward(self, x, a, e):
        return concrete_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # abs_fwd_phi = fucts[self.phi_f][2]
        # abs_fwd_sigma = fucts[self.sigma_f][2]
        # n_nodes = x.lb.shape[1]
        # n_edges = e.lb.shape[1]
        # index_targets = a.lb[0][1]  # Nodes receiving the message
        # index_sources = a.lb[0][0]  # Nodes sending the message (ie neighbors)
        # e = e.lb[0]
        # lwx, uwx = x.lw[0], x.uw[0]
        # emb_list_l = []
        # emb_list_u = []
        # for k in range(dim_in):
        #     lw = lwx[k]
        #     uw = uwx[k]
        #     messages_l = [[] for _ in range(n_nodes)]  # list of lists of messages
        #     messages_u = [[] for _ in range(n_nodes)]
        #     for idx in range(n_edges):
        #         sidx = index_sources[idx]
        #         tidx = index_targets[idx]
        #         messages_l[index_targets[idx]].append(abs_fwd_phi(lw[sidx], e[idx], lw[tidx]))
        #         messages_u[index_targets[idx]].append(abs_fwd_phi(uw[sidx], e[idx], uw[tidx]))
        #
        #     embeddings_l = torch.stack([abs_fwd_sigma(m, lw[i]) for i, m in enumerate(messages_l)])
        #     embeddings_u = torch.stack([abs_fwd_sigma(m, uw[i]) for i, m in enumerate(messages_u)])
        #     emb_list_l.append(embeddings_l)
        #     emb_list_u.append(embeddings_u)
        #
        # embeddings_l = torch.stack(emb_list_l).unsqueeze(0)
        # embeddings_u = torch.stack(emb_list_u).unsqueeze(0)
        #
        # return LinearBound(lw=embeddings_l, lb=x.lb, uw=embeddings_u, ub=x.ub)

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        ((lAx, uAx), _, (lAe, uAe)), lbias, ubias = bwd_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn,
                                                                        last_lA, last_uA, x, a, e)
        return [(lAx, uAx), (None, None), (lAe, uAe)], lbias, ubias

    def interval_propagate(self, *v):
        """ IBP computation """
        x, a, e = v
        return interval_message_passing(0, 1, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)


register_custom_op("custom::Pre", BoundPre)

########################################################################################################################################################


# POSTIMAGE
class Post(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, define the arguments and attributes of the operator.
        "custom::SigmaSum" is the name of the new operator, "x" is an argument
        of the operator, "const_i" is an attribute which stands for "c" in the operator.
        There can be multiple arguments and attributes. For attribute naming,
        use a suffix such as "_i" to specify the data type, where "_i" stands for
        integer, "_t" stands for tensor, "_f" stands for float, etc. """
        return g.op('custom::Post', x, a, e, phi_s=phis, sigma_s=sigmas, use_optimized_gcn_s=str(use_optimized_gcn)).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, phis, sigmas, use_optimized_gcn):
        """ In this function, implement the computation for the operator, i.e.,
        f(x) = i * e in this case. """
        return concrete_message_passing(1, 0, fucts[phis], fucts[sigmas], use_optimized_gcn, x, a, e)


class PostImage(nn.Module):
    def __init__(self, phi_f, sigma_f, use_optimized_gcn):
        super().__init__()
        self.phi_f = phi_f
        self.sigma_f = sigma_f
        self.use_optimized_gcn = use_optimized_gcn

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Post.apply(x, a, e, self.phi_f, self.sigma_f, self.use_optimized_gcn)


class BoundPost(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.phi_f = attr['phi']
        self.sigma_f = attr['sigma']
        self.use_optimized_gcn = attr['use_optimized_gcn'] == 'True'

    def forward(self, x, a, e):
        return concrete_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError
        # abs_fwd_phi = fucts[self.phi_f][2]
        # abs_fwd_sigma = fucts[self.sigma_f][2]
        # n_nodes = x.lb.shape[1]
        # n_edges = e.lb.shape[1]
        # index_targets = a.lb[0][1]  # Nodes receiving the message
        # index_sources = a.lb[0][0]  # Nodes sending the message (ie neighbors)
        # e = e.lb[0]
        # lwx, uwx = x.lw[0], x.uw[0]
        # emb_list_l = []
        # emb_list_u = []
        # for k in range(dim_in):
        #     lw = lwx[k]
        #     uw = uwx[k]
        #     messages_l = [[] for _ in range(n_nodes)]  # list of lists of messages
        #     messages_u = [[] for _ in range(n_nodes)]
        #     for idx in range(n_edges):
        #         sidx = index_sources[idx]
        #         tidx = index_targets[idx]
        #         messages_l[index_sources[idx]].append(abs_fwd_phi(lw[tidx], e[idx], lw[sidx]))
        #         messages_u[index_sources[idx]].append(abs_fwd_phi(uw[tidx], e[idx], uw[sidx]))
        #
        #     embeddings_l = torch.stack([abs_fwd_sigma(m, lw[i]) for i, m in enumerate(messages_l)])
        #     embeddings_u = torch.stack([abs_fwd_sigma(m, uw[i]) for i, m in enumerate(messages_u)])
        #     emb_list_l.append(embeddings_l)
        #     emb_list_u.append(embeddings_u)
        #
        # embeddings_l = torch.stack(emb_list_l).unsqueeze(0)
        # embeddings_u = torch.stack(emb_list_u).unsqueeze(0)
        #
        # return LinearBound(lw=embeddings_l, lb=x.lb, uw=embeddings_u, ub=x.ub)

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        return bwd_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, last_lA, last_uA, x, a, e)

    def interval_propagate(self, *v):
        """ IBP computation """
        x, a, e = v
        return interval_message_passing(1, 0, fucts[self.phi_f], fucts[self.sigma_f], self.use_optimized_gcn, x, a, e)


register_custom_op("custom::Post", BoundPost)

########################################################################################################################################################


# ITE-CROSS
class Ite(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, b, a, e, iftrue, iffalse, shp):
        return g.op('custom::Ite', x, b, a, e, shp_t=shp, iftrue_s=mg_reconstructor.reconstruct(iftrue.expr),
                    iffalse_s=mg_reconstructor.reconstruct(iffalse.expr)).setType(x.type())

    @staticmethod
    def forward(ctx, x, b, a, e, iftrue, iffalse, shp):
        if torch.all(b):
            return iftrue(x, a, e)
        else:
            return iffalse(x, a, e)


class Choice(nn.Module):
    def __init__(self, iftrue, iffalse):
        super().__init__()
        self.iftrue = iftrue
        self.iffalse = iffalse

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        b, x = x
        return Ite.apply(x, b, a, e, self.iftrue, self.iffalse, x.shape)


class BoundChoice(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        x, b, a, e = inputs
        self.x = x
        self.a = a
        self.e = e
        self.iftrue = BoundedModule(interpreter.run(attr['iftrue']), (torch.empty_like(torch.zeros(attr['shp'])), a.value,
                                                                      torch.empty_like(e.value)), device=x.device, verbose=True)
        self.iffalse = BoundedModule(interpreter.run(attr['iffalse']), (torch.empty_like(torch.zeros(attr['shp'])), a.value,
                                                                        torch.empty_like(e.value)), device=x.device, verbose=True)

    def forward(self, x, b, a, e):
        return torch.cond(torch.all(b), self.iftrue, self.iffalse, (x, a, e))

    def bound_forward(self, dim_in, x, b, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, x, b, a, e, **kwargs):
        """ Backward mode bound propagation """
        lb, ub = b.lower, b.upper
        x = interval_to_bounded_tensor(x.lower, x.upper)
        if torch.all(lb):  # if lower bound is all True, guaranteed iftrue branch
            _, _, A_dict = self.iftrue.compute_bounds((x, a.value, e.value), C=torch.transpose(last_lA, 0, 1), method='backward', return_A=True,
                                                      need_A_only=True, needed_A_dict={self.iftrue.final_name: [self.iftrue.final_name]})
            lA, uA = A_dict[self.iftrue.final_name][self.iftrue.final_name]['lA'], A_dict[self.iftrue.final_name][self.iftrue.final_name]['uA']
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0
        elif not torch.all(ub):  # if upper bound is not all True, guaranteed iffalse branch
            _, _, A_dict = self.iffalse.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward',
                                                       return_A=True, need_A_only=True, needed_A_dict={self.iffalse.final_name: [self.iffalse.final_name]})
            lA, uA = A_dict[self.iffalse.final_name][self.iffalse.final_name]['lA'], A_dict[self.iffalse.final_name][self.iffalse.final_name]['uA']
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0
        else:
            tmpl, tmpu, A_dict = self.iftrue.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward',
                                                            return_A=True, need_A_only=False, needed_A_dict={self.iftrue.final_name: [self.iftrue.final_name]})
            tlA, tuA = A_dict[self.iftrue.final_name][self.iftrue.final_name]['lA'], A_dict[self.iftrue.final_name][self.iftrue.final_name]['uA']
            fmpl, fmpu, A_dict = self.iffalse.compute_bounds((x, self.a.value, self.e.value), C=torch.transpose(last_lA, 0, 1), method='backward',
                                                             return_A=True, need_A_only=False,
                                                             needed_A_dict={self.iffalse.final_name: [self.iffalse.final_name]})
            flA, fuA = A_dict[self.iffalse.final_name][self.iffalse.final_name]['lA'], A_dict[self.iffalse.final_name][self.iffalse.final_name]['uA']
            lA, uA = backward_lub(tlA, tuA, tmpl, tmpu, flA, fuA, fmpl, fmpu)
            return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0

    def interval_propagate(self, *v):
        """ IBP computation """
        x, b, a, e = v
        lb, ub = b
        x = interval_to_bounded_tensor(*x)
        if torch.all(lb):  # if lower bound is all True, guaranteed iftrue branch
            return self.iftrue.compute_bounds((x, self.a.value, self.e.value), method='IBP')
        elif not torch.all(ub):  # if upper bound is not all True, guaranteed iffalse branch
            return self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='IBP')
        else:  # lub of iftrue and iffalse
            ift = self.iftrue.compute_bounds((x, self.a.value, self.e.value), method='IBP')
            iff = self.iffalse.compute_bounds((x, self.a.value, self.e.value), method='IBP')
            tlb, tub = ift[0], ift[1]
            flb, fub = iff[0], iff[1]
            return least_upper_bound(tlb, tub, flb, fub)


register_custom_op("custom::Ite", BoundChoice)

########################################################################################################################################################


# FIX
class Fix(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, a, e, body, atol, rtol, shp):
        return g.op('custom::Fix', x, a, e, shp_t=shp, body_s=mg_reconstructor.reconstruct(body.expr), atol_f=atol, rtol_f=rtol).setType(x.type())

    @staticmethod
    def forward(ctx, x, a, e, body, atol, rtol, shp):
        x_old = x
        x = body(x_old, a, e)
        while not torch.allclose(x_old, x, atol=atol, rtol=rtol) and not torch.any(torch.isnan(x)):
            x_old = x
            x = body(x_old, a, e)
        return x


class FixPoint(nn.Module):
    def __init__(self, body, atol, rtol):
        super().__init__()
        self.body = body
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, a, e):
        """ Use `.apply` to call the defined custom operator."""
        return Fix.apply(x, a, e, self.body, self.atol, self.rtol, x.shape)


class BoundFix(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        x, a, e = inputs
        self.x = x
        self.a = a
        self.e = e
        self.atol = attr['atol']
        self.rtol = attr['rtol']
        interpreter.set_tolerance(self.atol, self.rtol)
        self.body = BoundedModule(interpreter.run(attr['body']), (torch.empty_like(torch.zeros(attr['shp'])), a.value, torch.empty_like(e.value)),
                                  device=x.device, verbose=True)

    def forward(self, x, a, e):
        x_old = x
        x = self.body(x_old, a, e)
        while not torch.allclose(x_old, x, atol=self.atol, rtol=self.rtol) and not torch.any(torch.isnan(x)):
            x_old = x
            x = self.body(x_old, a, e)
        return x

    def bound_forward(self, dim_in, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        raise NotImplementedError

    def bound_backward(self, last_lA, last_uA, x, a, e, **kwargs):
        """ Backward mode bound propagation """
        R = x.interval
        T = R
        R = interval_to_bounded_tensor(*R)
        *R, A_dict = self.body.compute_bounds((R, a.value, e.value), method='backward', C=torch.transpose(last_lA, 0, 1), return_A=True,
                                              needed_A_dict={self.body.final_name: [self.body.final_name]})
        R = least_upper_bound(*T, *R)
        while not torch.allclose(T[0], R[0], atol=self.atol, rtol=self.rtol) and not torch.allclose(T[1], R[1], atol=self.atol, rtol=self.rtol):
            T = R
            R = interval_to_bounded_tensor(*R)
            *R, A_dict = self.body.compute_bounds((R, a.value, e.value), C=A_dict[self.body.final_name][self.body.final_name]['lA'], method='backward',
                                                  return_A=True, needed_A_dict={self.body.final_name: [self.body.final_name]})
            R = least_upper_bound(*T, *R)
        lA, uA = A_dict[self.body.final_name][self.body.final_name]['lA'], A_dict[self.body.final_name][self.body.final_name]['uA']
        return [(torch.transpose(lA, 0, 1), torch.transpose(uA, 0, 1)), (None, None), (None, None), (None, None)], 0, 0

    def interval_propagate(self, *v):
        """ IBP computation """
        R, a, e = v
        T = R
        R = interval_to_bounded_tensor(*R)
        # R = least_upper_bound(*R, *self.body.compute_bounds((self.x.value, self.a.value, self.e.value), method='IBP'))
        R = self.body.compute_bounds((R, self.a.value, self.e.value), method='IBP')
        while not torch.allclose(T[0], R[0], atol=self.atol, rtol=self.rtol) and not torch.allclose(T[1], R[1], atol=self.atol, rtol=self.rtol):
            T = R
            R = interval_to_bounded_tensor(*R)
            # R = least_upper_bound(*R, *self.body.compute_bounds((new_x, self.a.value, self.e.value), method='IBP'))
            R = self.body.compute_bounds((R, self.a.value, self.e.value))
        return R


register_custom_op("custom::Fix", BoundFix)

########################################################################################################################################################
# class PreGCN(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def forward(self, x, a, e):
#         return torch.matmul(e.transpose(-1, -2), x)
#
#
# class PostGCN(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def forward(self, x, a, e):
#         return torch.matmul(e, x)


# Abstract interpreter
class LirpaInterpreter(Interpreter):
    def __init__(self):
        self.graph_abstraction: GraphAbstraction
        self.atol = 0.000001
        self.rtol = 0
        self.mg_layers: dict
        self.context = Context()
        self.use_optimized_gcn = None

    def set_graph_abstraction(self, graph_abstraction):
        self.graph_abstraction = graph_abstraction

    def set_concrete_layers(self, mg_layers):
        self.mg_layers = mg_layers

    def set_tolerance(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol

    def get_concrete_layer(self, tree):
        return self.mg_layers[hash(self.context.get(tree))]

    def get_abstract_layer(self, conc_op):
        psi = conc_op.psi
        if isinstance(psi.f, layers.Dense):  # Dense layer
            layer = psi.f
            return make_layer(layer.trainable_variables[0].value.numpy(), layer.trainable_variables[1].value.numpy(), layer.activation.__name__)
        else:  # Pooling layer
            return make_pooling('sum' if psi.fname == 'SumPooling' else 'mean', self.graph_abstraction)

    def run(self, expr):  # inputs in tf format
        tree = mg_parser.parse(expr) if isinstance(expr, str) else expr
        output = self.visit(tree)
        return output

    @v_args(inline=True)
    def label(self, tree):
        return str(tree)

    @v_args(inline=True)
    def id(self):
        return torch.nn.Identity()

    def atom_op(self, tree):
        concrete_op = self.get_concrete_layer(tree)
        abstract_op = self.get_abstract_layer(concrete_op)

        class Atom(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)

            def forward(self, x, a, e):
                for layer in self.layers:
                    x = layer(x)
                return x

        if isinstance(abstract_op, list) or isinstance(abstract_op, tuple):
            return Atom(abstract_op)
        else:
            return Atom([abstract_op])

    def lhd(self, tree):
        phi, sigma = tree.children
        phi, sigma = self.visit(phi), self.visit(sigma)
        return PreImage(phi, sigma, self.graph_abstraction.optimized_gcn)

    def rhd(self, tree):
        phi, sigma = tree.children
        phi, sigma = self.visit(phi), self.visit(sigma)
        return PostImage(phi, sigma, self.graph_abstraction.optimized_gcn)

    def sequential_composition(self, tree):
        left, right = tree.children

        class Sequential(torch.nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = torch.nn.ModuleList(layers)

            def forward(self, x, a, e):
                for layer in self.layers:
                    x = layer(x, a, e)
                return x
        phi = self.visit(left)
        self.context.push(left)
        psi = self.visit(right)
        self.context.pop()
        return Sequential([phi, psi])

    @v_args(inline=True)
    def parallel_composition(self, left, right):

        class Parallel(torch.nn.Module):
            def __init__(self, ll, r):
                super().__init__()
                self.ll = ll
                self.r = r

            def forward(self, x, a, e):
                y = self.ll(x, a, e)
                z = self.r(x, a, e)
                return y, z
        left = self.visit(left)
        right = self.visit(right)
        return Parallel(left, right)

    @v_args(inline=True)
    def choice(self, left, right):
        iftrue = self.visit(left)
        iftrue.expr = left
        iffalse = self.visit(right)
        iffalse.expr = right
        return Choice(iftrue, iffalse)

    @v_args(inline=True)
    def star(self, body):
        loop = self.visit(body)
        loop.expr = body
        return FixPoint(loop, self.atol, self.rtol)


interpreter = LirpaInterpreter()


# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PostImage('phi_prod', 'sigma_sum')
#         self.lin1 = torch.nn.Linear(3, 5, bias=False)
#         self.relu = torch.nn.ReLU()
#         self.pre2 = PostImage('phi_prod', 'sigma_sum')
#         self.lin2 = torch.nn.Linear(5, 2, bias=False)
#         self.sfmax = torch.nn.Softmax()
#
#     def forward(self, x, a, e):
#         x1 = self.pre1(x, a, e)
#         x2 = self.lin1(x1)
#         x3 = self.relu(x2)
#         x4 = self.pre2(x3, a, e)
#         x5 = self.lin2(x4)
#         return x5

# PAR
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.post1 = PostImage('phi_prod', 'sigma_sum')
#
#     def forward(self, x, a, e):
#         x1 = self.pre1(x, a, e)
#         x2 = self.post1(x, a, e)
#         return torch.add(x1, x2)


# ITE
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.post1 = PostImage('phi_prod', 'sigma_sum')
#         self.ite = Choice(self.pre1, self.post1)
#
#     def forward(self, x, xb, a, e):
#         x1 = self.ite(x, xb, a, e)
#         return x1

# FIX
# class MyModel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pre1 = PreImage('phi_prod', 'sigma_sum')
#         self.fix = FixPoint(self.pre1, 0.000001, 0)
#
#     def forward(self, x, a, e):
#         x1 = self.fix(x, a, e)
#         return x1


def check_soundness(pred, lb, ub):
    eps = interpreter.atol
    pred = pred[0] if isinstance(pred, tuple) else pred
    pred = pred[0] if pred.shape.ndims == 3 else pred
    lb = lb[0] if lb.ndim == 3 else lb
    ub = ub[0] if ub.ndim == 3 else ub
    for i, (prow, lrow, urow) in enumerate(zip(pred, lb, ub)):
        for j, (p, l, u) in enumerate(zip(prow, lrow, urow)):
            assert l - eps <= p <= u + eps, "Unsound result at {0},{1}: {2} <!= {3} <!= {4}".format(i, j, l, p, u)
    print('Soundness check passed')
