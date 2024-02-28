import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import pickle as pk
import warnings

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed, load_data, get_rank_metrics, rank_evaluate, convert_to_gpu
from utils.metric import overlap
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

def main(args):
    torch.set_num_threads(5) 

    warnings.filterwarnings('ignore')

    set_random_seed(0)

    ndcg_scores = []
    spearmans = []
    rmses = []
    overlaps = []
    medAEs = []

    ndcg_scores_20 = []
    ndcg_scores_50 = []
    ndcg_scores_200 = []
    overlaps_20 = []
    overlaps_50 = []
    overlaps_200 = []

    # set the save path
    save_root = 'pretrain/results/' + args.dataset + '_simple_supervised_/'

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):
        # print('-------------------------------------')
        # print('Cross:{}'.format(cross_id))
        # print('------Dataset Loading')

                
        g, edge_types, _, rel_num, struct_feat, semantic_feat, labels, train_idx, val_idx, test_idx = \
            load_data(args.data_path, args.dataset, cross_id)



        #features = torch.cat((struct_feat,semantic_feat),dim=1)
        features = struct_feat

        # create model
        loss_fcn = torch.nn.MSELoss()

        if args.sv_model == 'mlp':
            model = MLPRegressor()
        elif args.sv_model == 'lr':
            model = LinearRegression()

        model.fit(features[train_idx], labels[train_idx])
        pred = model.predict(features[test_idx])

        pred = torch.Tensor(pred)
        test_loss, test_ndcg, test_spearman, test_overlap, test_medianAE = \
                rank_evaluate(pred, labels[test_idx], args.list_num, loss_fcn)
        _, test_ndcg_20, _, test_overlap_20, _ = \
                rank_evaluate(pred, labels[test_idx], 20, loss_fcn)
        _, test_ndcg_50, _, test_overlap_50, _  = \
                rank_evaluate(pred, labels[test_idx], 50, loss_fcn)
        _, test_ndcg_200, _, test_overlap_200, _  = \
                rank_evaluate(pred, labels[test_idx], 200, loss_fcn)


        ndcg_scores.append(test_ndcg)
        spearmans.append(test_spearman)
        rmses.append(torch.sqrt(test_loss).item())
        overlaps.append(test_overlap)
        medAEs.append(test_medianAE)

        ndcg_scores_20.append(test_ndcg_20)
        ndcg_scores_50.append(test_ndcg_50)
        ndcg_scores_200.append(test_ndcg_200)
        overlaps_20.append(test_overlap_20)
        overlaps_50.append(test_overlap_50)
        overlaps_200.append(test_overlap_200)

    
    ndcg_scores = np.array(ndcg_scores)
    spearmans = np.array(spearmans)
    rmses = np.array(rmses)
    overlaps = np.array(overlaps)
    medAEs = np.array(medAEs)

    ndcg_scores_20 = np.array(ndcg_scores_20)
    ndcg_scores_50 = np.array(ndcg_scores_50)
    ndcg_scores_200 = np.array(ndcg_scores_200)
    overlaps_20 = np.array(overlaps_20)
    overlaps_50 = np.array(overlaps_50)
    overlaps_200 = np.array(overlaps_200)



    
    print()
    print(args)
    print('sv model | dataset')
    print('{} | {}'.format(args.sv_model, args.dataset))
    print('---------------------')
    print('ndcg_20 | spearmans | over_20 | rmse | medianAE')
    print('{:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f}'.format(ndcg_scores_20.mean(), np.std(ndcg_scores_20), spearmans.mean(), np.std(spearmans), overlaps_20.mean(), np.std(overlaps_20), rmses.mean(), np.std(rmses),medAEs.mean() ,np.std(medAEs)))
    print('ndcg_50 | spearmans | over_50 | rmse | medianAE')
    print('{:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f}'.format(ndcg_scores_50.mean(), np.std(ndcg_scores_50), spearmans.mean(), np.std(spearmans), overlaps_50.mean(), np.std(overlaps_50), rmses.mean(), np.std(rmses),medAEs.mean() ,np.std(medAEs)))
    print('ndcg_100 | spearmans | over_100 | rmse | medianAE')
    print('{:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f}'.format(ndcg_scores.mean(), np.std(ndcg_scores), spearmans.mean(), np.std(spearmans), overlaps.mean(), np.std(overlaps), rmses.mean(), np.std(rmses),medAEs.mean() ,np.std(medAEs)))
    print('ndcg_200 | spearmans | over_200 | rmse | medianAE')
    print('{:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.4f}±{:.4f}'.format(ndcg_scores_200.mean(), np.std(ndcg_scores_200), spearmans.mean(), np.std(spearmans), overlaps_200.mean(), np.std(overlaps_200), rmses.mean(), np.std(rmses),medAEs.mean() ,np.std(medAEs)))
    print('---------------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SV')
    parser.add_argument("--dataset", type=str, default='FB15k_rel',
                        help="The input dataset. Can be FB15k")
    parser.add_argument("--data_path", type=str, default='datasets/fb15k_rel.pk',
                        help="path of dataset")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    # parser.add_argument("--epochs", type=int, default=10000,
    #                     help="number of training epochs")
    # parser.add_argument('--min-epoch', type=int, default=-1,
    #                     help='the least epoch for training, avoiding stopping at the start time')
    # parser.add_argument("--num-heads", type=int, default=4,
    #                     help="number of hidden attention heads")
    # parser.add_argument("--num-out-heads", type=int, default=4,
    #                     help="number of output attention heads")
    # parser.add_argument("--num-layers", type=int, default=2,
    #                     help="number of hidden layers")
    # parser.add_argument("--num-hidden", type=int, default=16,
    #                     help="number of hidden units")
    # parser.add_argument("--residual", action="store_true", default=False,
    #                     help="use residual connection")
    # parser.add_argument("--feat-drop", type=float, default=0.)
    # parser.add_argument("--in-drop", type=float, default=.3,
    #                     help="input feature dropout")
    # parser.add_argument("--attn-drop", type=float, default=.3,
    #                     help="attention dropout")
    # parser.add_argument("--lr", type=float, default=0.005,
    #                     help="learning rate")
    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                     help="weight decay")
    # parser.add_argument('--negative-slope', type=float, default=0.2,
    #                     help="the negative slope of leaky relu")
    # parser.add_argument('--early-stop', action='store_true', default=True,
    #                     help="indicates whether to use early stop or not")
    # parser.add_argument('--patience', type=int, default=1000,
    #                     help="indicates whether to use early stop or not")
    # parser.add_argument('--scale', action="store_true", default=False,
    #                     help="utilize centrality to scale scores")
    # parser.add_argument('--pred-dim', type=int, default=10,
    #                     help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='gat-two_checkpoint.pt',
                        help='the path to save the best model')

    # parser.add_argument('--loss-lambda', type=float, default=0.5,
    #                     help='the weight to add unsupervised loss')
    # parser.add_argument('--norm', action="store_true", default=False)
    # parser.add_argument('--edge-mode', type=str, default='MUL')
    # parser.add_argument('--spm', action="store_true", default=False)
    # parser.add_argument('--loss-alpha', type=float, default=0.3)
    parser.add_argument('--list-num', type=int, default=100)


    # #######
    # parser.add_argument('--loss-eta1', type=float, default=1.0)
    # parser.add_argument('--loss-eta2', type=float, default=1.0)
    # parser.add_argument('--important-ratio-k', type=float, default=1.0)
    # parser.add_argument('--temp', type=float, default=0.05)
    # parser.add_argument('--normal-important-ratio', type=float, default=9.0)
    parser.add_argument('--sv-model', type=str, default='mlp')
 
    args = parser.parse_args()

    main(args)
