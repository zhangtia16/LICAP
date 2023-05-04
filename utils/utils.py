# the following is modified from https://github.com/GRAPH-0/RGTN-NIE
import dgl
import numpy as np
import pickle
import random
import torch
from sklearn.metrics import f1_score
from .metric import ndcg, spearman_sci, overlap, median_AE
from scipy import stats
import pdb
import os


def convert_to_gpu(*data, device):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    return tuple(res)


def set_random_seed(seed=0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    dgl.random.seed(seed)

    


def load_model(model, model_path):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def count_parameters_in_KB(model):
    """
    count the size of trainable parameters in model (KB)
    :param model: model
    :return:
    """
    param_num = np.sum(np.prod(v.size()) for v in model.parameters()) / 1e3
    return param_num


def get_rank_metrics(predicts, labels, NDCG_k):
    """
    calculate NDCG@k metric
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :return:
    """
    return ndcg(labels, predicts, NDCG_k), spearman_sci(labels, predicts), overlap(labels, predicts, NDCG_k)


def rank_evaluate(predicts, labels, NDCG_k, loss_func):
    """
    evaluation used for validation or test
    :param predicts: Tensor, shape (N, 1)
    :param labels: Tensor, shape (N, 1)
    :param loss_func: loss function
    :return:
    """
    with torch.no_grad():
        loss = loss_func(predicts, labels)
    ndcg_score, spear_score, overlap_score = get_rank_metrics(predicts, labels, NDCG_k)

    medianAE_score = median_AE(predicts, labels)
    
    return loss, ndcg_score, spear_score, overlap_score, medianAE_score



def load_fb15k_rel_data(data_path, cross_validation_shift=0, dataset_name='FB15k_rel'):
    """
    load fb15k data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    labels = data['labels']

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/fb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']
    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    rel_num = (max(edge_types) + 1).item()

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    # # log transform for labels
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # # semi-supervised
    # labeled_train_idx = np.random.choice(train_idx, int(labeled_ratio*len(train_idx)),replace=False)
    # unlabeled_train_idx = np.array([i for i in train_idx if i not in labeled_train_idx])

    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def load_imdb_s_rel_data(data_path, cross_validation_shift=0, dataset_name='IMDB_S_rel'):
    """
    load imdb rel data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'two' in dataset_name:
        node_feat1 = torch.from_numpy(pickle.load(open('./datasets/imdb_s_node2vec.pk', 'rb')))
        node_feat2 = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    elif 'concat' in dataset_name:
        node_feat1 = torch.from_numpy(pickle.load(open('./datasets/imdb_s_node2vec.pk', 'rb')))
        node_feat2 = pickle.load(open('./datasets/imdb_s_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], 1)
    else:
        node_feats = torch.from_numpy(pickle.load(open('datasets/imdb_s_node2vec.pk', 'rb')))

    # edge list
    edges = data['edges']
    labels = data['labels'].float()
    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    # rel_num = (max(edge_types) + 1).item()
    rel_num = 30

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # log transform for labels
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    if 'two' in dataset_name:
        return hg, edge_types, None, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, None, rel_num, node_feats, labels, train_idx, val_idx, test_idx


def load_tmdb_rel_data(data_path, cross_validation_shift=0, dataset_name='TMDB_rel'):
    """
    load tmdb rel data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    labels = data['labels'].float()
    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    # rel_num = (max(edge_types) + 1).item()
    rel_num = 34

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/tmdb_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], 1)
    else:
        node_feats = data['features']

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # log transform for labels
    labels = torch.log(1 + labels)
    
    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))


    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')


    # # semi-supervised
    # labeled_train_idx = np.random.choice(train_idx, int(labeled_ratio*len(train_idx)),replace=False)
    # unlabeled_train_idx = np.array([i for i in train_idx if i not in labeled_train_idx])

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels,train_idx,  val_idx, test_idx



def load_ga16k_rel_data(data_path, cross_validation_shift=0, dataset_name='GA16k_rel'):
    """
    load ga16k data
    :param data_path: str, data file path
    :param cross_validation_shift: int, shift of data split
    :return:
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # edge list
    edges = data['edges']
    labels = data['labels']

    if 'semantic' in dataset_name:
        node_feats = pickle.load(open('./datasets/ga_lang.pk', 'rb'))
        node_feats = torch.from_numpy(node_feats).float()
    elif 'concat' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/ga_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
        node_feats = torch.cat([node_feat1, node_feat2], dim=1)
    elif 'two' in dataset_name:
        node_feat1 = data['features']
        node_feat2 = pickle.load(open('./datasets/ga_lang.pk', 'rb'))
        node_feat2 = torch.from_numpy(node_feat2).float()
    else:
        node_feats = data['features']

    invalid_masks = data['invalid_masks']
    edge_types = data['edge_types']
    rel_num = (max(edge_types) + 1).item()

    # construct a heterogeneous graph
    hg = dgl.graph(edges)

    # generate edge norm
    g = hg.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    node_norm = torch.from_numpy(norm).view(-1, 1)
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    edge_norm = g.edata['norm']

    # # log transform for labels
    labels = torch.log(1 + labels)

    # split dataset
    float_mask = np.ones(hg.number_of_nodes()) * -1.
    label_mask = (invalid_masks == 0)
    float_mask[label_mask] = np.random.RandomState(seed=0).permutation(np.linspace(0, 1, label_mask.sum()))

    # train_idx, val_idx, test_idx
    # 70% for train, 10% for val, 20% for test
    if cross_validation_shift == 0:
        test_idx = np.where((0. <= float_mask) & (float_mask <= 0.2))[0]
        val_idx = np.where((0.2 < float_mask) & (float_mask <= 0.3))[0]
        train_idx = np.where(float_mask > 0.3)[0]
    elif cross_validation_shift == 1:
        test_idx = np.where((0.2 <= float_mask) & (float_mask <= 0.4))[0]
        val_idx = np.where((0.4 < float_mask) & (float_mask <= 0.5))[0]
        train_idx = np.where((float_mask > 0.5) | ((0 <= float_mask) & (float_mask < 0.2)))[0]
    elif cross_validation_shift == 2:
        test_idx = np.where((0.4 <= float_mask) & (float_mask <= 0.6))[0]
        val_idx = np.where((0.6 < float_mask) & (float_mask <= 0.7))[0]
        train_idx = np.where((float_mask > 0.7) | ((0 <= float_mask) & (float_mask < 0.4)))[0]
    elif cross_validation_shift == 3:
        test_idx = np.where((0.6 <= float_mask) & (float_mask <= 0.8))[0]
        val_idx = np.where((0.8 < float_mask) & (float_mask <= 0.9))[0]
        train_idx = np.where((float_mask > 0.9) | ((0 <= float_mask) & (float_mask < 0.6)))[0]
    elif cross_validation_shift == 4:
        test_idx = np.where((0.8 <= float_mask) & (float_mask <= 1.0))[0]
        val_idx = np.where((0 <= float_mask) & (float_mask <= 0.1))[0]
        train_idx = np.where((0.1 < float_mask) & (float_mask < 0.8))[0]
    else:
        raise ValueError(f'wrong value for parameter {cross_validation_shift}')

    # # semi-supervised
    # labeled_train_idx = np.random.choice(train_idx, int(labeled_ratio*len(train_idx)),replace=False)
    # unlabeled_train_idx = np.array([i for i in train_idx if i not in labeled_train_idx])

    if 'two' in dataset_name:
        return hg, edge_types, edge_norm, rel_num, node_feat1, node_feat2, labels, train_idx, val_idx, test_idx
    return hg, edge_types, edge_norm, rel_num, node_feats, labels, train_idx, val_idx, test_idx




def load_data(data_path, dataset_name, cross_validation_shift=0):
    """
    load dataset based on the input dataset name
    :param data_path: str, data file path
    :param dataset_name: dataset name
    :param cross_validation_shift: int, shift of data split
    : my mod label_transform
    :return:
    """
    
    if dataset_name.startswith('FB15k'):
        return load_fb15k_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('IMDB_S'):
        return load_imdb_s_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('TMDB'):
        return load_tmdb_rel_data(data_path, cross_validation_shift, dataset_name)
    elif dataset_name.startswith('GA16k'):
        return load_ga16k_rel_data(data_path, cross_validation_shift, dataset_name)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset_name))


def get_centrality(graph):
    g = graph.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    theta = 1e-4
    centrality = torch.log(in_deg + theta)
    return centrality