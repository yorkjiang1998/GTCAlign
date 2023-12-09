# 作者：York
# 时间：2022/5/31 16:48

from model import *
import numpy as np
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from utils import *
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity

def log(iteration, acc, MAP, top1, top10):
    print("iteration is: {:d}, Accuracy: {:.4f}, MAP: {:.4f}, Precision_1: {:.4f}, Precision_10: {:.4f}".format(iteration + 1, acc, MAP, top1, top10))

def refine_alignment(model, s_sadj, t_sadj, args,s_topo, t_topo):
    s_sadj = s_sadj.to_dense()
    t_sadj = t_sadj.to_dense()
    CloAlign_S, groundtruth = refine(model, s_sadj, t_sadj, 0.8, args,s_topo, t_topo)
    return CloAlign_S, groundtruth

def refine(model, source_A_hat, target_A_hat, threshold, args,s_topo_, t_topo_):
    refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), False)
    S_max = None
    source_outputs = model(refinement_model(source_A_hat, 's'), 's')
    target_outputs = model(refinement_model(target_A_hat, 't'), 't')

    ground_truth = np.genfromtxt('./data/' + args.dataset + '/' + args.dataset + '_ground_True.txt', dtype=np.int32)
    ground_truth = list(ground_truth)
    full_di = dict(ground_truth)
    if args.dataset in ['douban', 'allmv_tmdb']:
        full_dic = dict([val, key] for key, val in full_di.items())
    elif args.dataset in ['ppi', 'flickr_myspace','citeseer', 'flickr','dblp','acm_dblp','nell','bgs_new','elliptic','fq_tw']:
        full_dic = full_di
    acc, S = get_acc(source_outputs, target_outputs, full_dic, args.alphas, just_S=True)
    score = np.max(S, axis=1).mean()
    alpha_source_max = None
    alpha_target_max = None
    if 1:
        refinement_model.score_max = score
        alpha_source_max = refinement_model.alpha_source
        alpha_target_max = refinement_model.alpha_target
        S_max = S
    source_candidates, target_candidates = [], []
    alpha_source_max = refinement_model.alpha_source + 0
    alpha_target_max = refinement_model.alpha_target + 0
    s_topo, t_topo = s_topo_, t_topo_
    score = []
    for iteration in range(args.r_epochs):
        source_candidates, target_candidates, len_source_candidates, count_true_candidates, source_candidates_5, target_candidates_5 = topo_refine(
            source_outputs, target_outputs, threshold, full_dic, source_A_hat, target_A_hat, s_topo, t_topo, args)
        refinement_model.alpha_source[source_candidates] *= args.stable_xs[0]
        refinement_model.alpha_source[source_candidates_5] *= args.stable_xs[1]
        refinement_model.alpha_target[target_candidates] *= args.stable_xs[0]
        refinement_model.alpha_target[target_candidates_5] *= args.stable_xs[1]

        source_outputs = model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = model(refinement_model(target_A_hat, 't'), 't')

        acc, S = get_acc(source_outputs, target_outputs, full_dic, args.alphas, just_S=True)
        top = [1, 10]
        acc, MAP, top1, top10 = get_statistics(S, full_dic, top, use_greedy_match=False,
                                               get_all_metric=True)
        log(iteration, acc, MAP, top1, top10)
        sc = np.max(S, axis=1).mean()
        score.append(sc)
        if iteration == args.r_epochs - 1:
            print("Numcandidate: {}, num_true_candidate: {}".format(len_source_candidates, count_true_candidates))
    print("Done refinement!")
    refinement_model.alpha_source = alpha_source_max
    refinement_model.alpha_target = alpha_target_max
    CloAlign = S_max
    return CloAlign, full_dic

def get_acc(source_outputs, target_outputs, test_dict=None, alphas=None, just_S=False):
    global acc, MAP, top1, top10
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))
    accs = ""
    for i in range(0, len(source_outputs)):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_numpy = S.detach().cpu().numpy()
        if test_dict is not None:
            if not just_S:
                acc = get_statistics(S_numpy, test_dict)
                accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += alphas[i] * S_numpy
        else:
            Sf += S_numpy
    if test_dict is not None:
        if not just_S:
            accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf


def get_statistics(alignment_matrix, groundtruth, top, groundtruth_matrix=None,  use_greedy_match=False,
                   get_all_metric=False):
    if use_greedy_match:
        print("This is greedy match accuracy")
        pred = greedy_match(alignment_matrix)
    else:
        pred = get_nn_alignment_matrix(alignment_matrix)
    acc = compute_accuracy(pred, groundtruth)
    if get_all_metric:
        MAP, Hit, AUC = compute_MAP_Hit_AUC(alignment_matrix, groundtruth)
        pred_top_1 = top_k(alignment_matrix, top[0])
        top1 = compute_precision_k(pred_top_1, groundtruth)
        pred_top_10 = top_k(alignment_matrix, top[1])
        top10 = compute_precision_k(pred_top_10, groundtruth)
        return acc, MAP, top1, top10
    return acc

def compute_accuracy(pred, gt):
    n_matched = 0
    if type(gt) == dict:
        for key, value in gt.items():
            if pred[key, value] == 1:
                n_matched += 1
        return n_matched / len(gt)

    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt == 1).sum()
    return n_matched / n_nodes

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            try:
                if top_k_matrix[key, value] == 1:
                    n_matched += 1
            except:
                n_matched += 1
        return n_matched / len(gt)

    gt_candidates = np.argmax(gt, axis=1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1

    n_nodes = (gt == 1).sum()
    return n_matched / n_nodes


def greedy_match(S):
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m, n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while (matched <= min_size):
        ipos = ix[index - 1]
        jc = int(np.ceil(ipos / m))
        ic = ipos - (jc - 1) * m
        if ic == 0: ic = 1
        if (used_rows[ic - 1] == 0 and used_cols[jc - 1] == 0):
            row[matched - 1] = ic - 1
            col[matched - 1] = jc - 1
            max_list[matched - 1] = x[index - 1]
            used_rows[ic - 1] = 1
            used_cols[jc - 1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result


def get_nn_alignment_matrix(alignment_matrix):
    # Sparse
    row = np.arange(len(alignment_matrix))
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result

def compute_MAP_Hit_AUC(alignment_matrix, gt):
    MAP = 0
    AUC = 0
    Hit = 0
    for key, value in gt.items():
        ele_key = alignment_matrix[key].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == value:
                ra = i + 1  # r1
                MAP += 1 / ra
                Hit += (alignment_matrix.shape[1] + 1) / alignment_matrix.shape[1]
                AUC += (alignment_matrix.shape[1] - ra) / (alignment_matrix.shape[1] - 1)
                break
    n_nodes = len(gt)
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, Hit, AUC


def top_k(S, k=1):
    top = np.argsort(-S)[:, :k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx, elm] = 1
    return result


def get_candidate(source_outputs, target_outputs, threshold, full_dict):
    List_S = get_similarity_matrices(source_outputs, target_outputs)[1:]
    source_candidates = []
    target_candidates = []
    count_true_candidates = 0
    if len(List_S) < 2:
        print("The current model doesn't support refinement for number of GNN layer smaller than 2")
        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

    # num_source_nodes = len(self.source_dataset.G.nodes())
    num_source_nodes = source_outputs[0].shape[0]
    # num_target_nodes = len(self.target_dataset.G.nodes())
    num_target_nodes = target_outputs[0].shape[1]
    for i in range(min(num_source_nodes, num_target_nodes)):
        node_i_is_stable = True
        for j in range(len(List_S)):
            if List_S[j][i].argmax() != List_S[j - 1][i].argmax() or List_S[j][i].max() < threshold:
                node_i_is_stable = False
                break
        if node_i_is_stable:
            tg_candi = List_S[-1][i].argmax()
            source_candidates.append(i)
            target_candidates.append(tg_candi)
            try:
                if full_dict[i] == tg_candi:
                    count_true_candidates += 1
            except:
                continue
    return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(
        source_candidates), count_true_candidates


def topo_refine(source_outputs, target_outputs, threshold, full_dict, s_sadj, t_sadj, s_topo, t_topo, args):
    List_S = get_similarity_matrices(source_outputs, target_outputs)[1:]
    source_candidates = []
    source_candidates_5 = []
    target_candidates = []
    target_candidates_5 = []
    count_true_candidates = 0
    num_source_nodes = s_sadj.shape[0]
    num_target_nodes = t_sadj.shape[0]
    for i in range(min(num_source_nodes, num_target_nodes)):
        node_i_is_stable = True
        for j in range(len(List_S)): # The process of realizing global topological consistency
            if List_S[j][i].argmax() != List_S[j - 1][i].argmax() or List_S[j][i].max() < threshold:
                node_i_is_stable = False
                t = t_s(List_S[j][i], List_S[j-1][i], args.top_k)
                if len(t) > 0:
                    for k in t:
                        x = []
                        y = []
                        for t_topo_index in range(len(t_topo)):
                            try:
                                x.append(s_topo[t_topo_index][i])
                                y.append(t_topo[t_topo_index][k])
                            except Exception:
                                x.append(0.0001)
                                y.append(0.0001)
                        x_topo_all = 0.0
                        y_topo_all = 0.0
                        for add in range(len(y)):
                            x_topo_all += x[add]
                            y_topo_all += y[add]
                        if (args.topo_diff[1] >= x_topo_all / y_topo_all >= args.topo_diff[0]) or (
                                    args.topo_diff[1] >= y_topo_all / x_topo_all >= args.topo_diff[0]):
                            source_candidates_5.append(i)
                            target_candidates_5.append(k)
                        break
                break
        if node_i_is_stable:
            tg_candi = List_S[-1][i].argmax()
            source_candidates.append(i)
            target_candidates.append(tg_candi)
            # try:
            if full_dict[i] == tg_candi:
                count_true_candidates += 1
            # except:
            #     continue
    return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(
        source_candidates), count_true_candidates , torch.LongTensor(source_candidates_5), torch.LongTensor(target_candidates_5)

def t_s(s_matrix, t_matrix, k):
    s_matrix = np.array(s_matrix.cpu().detach())
    t_matrix = np.array(t_matrix.cpu().detach())
    s_index = s_matrix.argsort()[::-1].tolist()[:k]
    t_index = t_matrix.argsort()[::-1].tolist()[:k]
    equal_node = []
    for i in s_index:
        for j in t_index:
            if i == j:
                equal_node.append(i)
                return equal_node
    if len(equal_node) == 0:
        return []


def get_t_s_matrix(s_matrix, t_matrix, k):
    s_matrix = np.array(s_matrix.cpu().detach())
    t_matrix = np.array(t_matrix.cpu().detach())

    return

def get_similarity_matrices(source_outputs, target_outputs):
    """
    Construct Similarity matrix in each layer
    :params source_outputs: List of embedding at each layer of source graph
    :params target_outputs: List of embedding at each layer of target graph
    """
    list_S = []
    for i in range(len(source_outputs)):
        source_output_i = source_outputs[i]
        target_output_i = target_outputs[i]
        S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
        #S = get_cos_matrix(source_output_i.cpu().detach().numpy(), target_output_i.cpu().detach().numpy())
        #S = torch.from_numpy(S).cuda()
        list_S.append(S)
    return list_S


def get_cos_matrix(source_output, target_output):
    return cosine_similarity(source_output, target_output)
