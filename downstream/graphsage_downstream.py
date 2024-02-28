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
from utils.utils import set_random_seed, load_data, get_rank_metrics, rank_evaluate, convert_to_gpu, get_centrality
from utils.metric import overlap
from baseline_models.graphsage import GraphSAGE
from collections import Counter
import matplotlib.pyplot as plt

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
    save_root = 'pretrain/results/' + args.dataset + '_' + args.pretrain_model +  '_pretrain_graphsage/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for cross_id in range(args.cross_num):
        g, edge_types, _, rel_num, struct_feat, semantic_feat, labels, train_idx, val_idx, test_idx = \
            load_data(args.data_path, args.dataset, cross_id)

        


        # load pretrain data
        feat_save_root = '/workspace1/zty/pretrain_data'

        dataset_name = args.dataset.rstrip('_two').rstrip('_rel').lower()
        struct_pretrain_data_root = '/workspace1/zty/pretrain_data/pregat_pretrain_struct/'
        semantic_pretrain_data_root = '/workspace1/zty/pretrain_data/pregat_pretrain_semantic/'
        

        imp_ratio_path = 'imp_ratio_'+ str(args.important_ratio) + '/'
        pretrain_patience_path = 'patience_'+ str(args.pretrain_patience) + '/'
        dataset_path = dataset_name + '/'

        
        struct_root =  struct_pretrain_data_root + imp_ratio_path + pretrain_patience_path + dataset_path
        semantic_root =  semantic_pretrain_data_root + imp_ratio_path + pretrain_patience_path + dataset_path

        suffix = str(args.lr) + 'r' + str(args.loss_eta2) + '_' + str(cross_id) + '.pkl'


        if args.pretrain_model == 'gat_struct':
            print('Pretrain from GAT_struct')
            # feat_pretrained_path = '/workspace1/zty/pretrain/gat_pretrain/pretrain_feat/' + args.dataset.strip('_two') + '_features_pretrained_lr' + suffix
            # struct_feat = pk.load(open(feat_pretrained_path, 'rb'))
            

        elif args.pretrain_model == 'pregat_struct':
            feat_pretrained_path = struct_root + dataset_name + '_struct_pregat_pretrained_lr' + suffix
            # print(feat_pretrained_path)
            features = pk.load(open(feat_pretrained_path, 'rb'))

        elif args.pretrain_model == 'pregat_semantic':
            feat_pretrained_path = semantic_root + dataset_name + '_semantic_pregat_pretrained_lr' + suffix
            #print(feat_pretrained_path)
            features = pk.load(open(feat_pretrained_path, 'rb'))
            
        
        elif args.pretrain_model == 'pregat_concat':
            struct_feat_pretrained_path = struct_root + dataset_name + '_struct_pregat_pretrained_lr' + suffix
            struct_feat = pk.load(open(struct_feat_pretrained_path, 'rb'))

            semantic_feat_pretrained_path = semantic_root + dataset_name + '_semantic_pregat_pretrained_lr' + suffix
            semantic_feat = pk.load(open(semantic_feat_pretrained_path, 'rb'))

            features = torch.cat((struct_feat,semantic_feat),dim=1)
            feat_pretrained_path = struct_feat_pretrained_path + '||' + semantic_feat_pretrained_path
            print(features.shape)

        elif args.pretrain_model == 'null_semantic':
            #print('Pretrain from null_semantic')
            feat_pretrained_path = 'null_semantic'
            features = semantic_feat
        elif args.pretrain_model == 'null_concat':
            #print('Pretrain from null_concat')
            feat_pretrained_path = 'null_concat'
            features = torch.cat((struct_feat,semantic_feat),dim=1)
            print(features.shape)
        elif args.pretrain_model == 'null_struct':
            #print('Pretrain from null_struct')
            feat_pretrained_path = 'null_struct'
            features = struct_feat
        else:
            print('Error:pretrain model no existing!')
        

        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()    

        # # cuda
        # imp_node_coeff = imp_node_coeff.cuda() 
          

        # g = data[0]
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            g = g.int().to(args.gpu)

        num_feats = features.shape[1]
        n_edges = g.number_of_edges()

        
        # add self loop
        g = dgl.add_self_loop(g)
        new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
        edge_types = torch.cat([edge_types, new_edge_types], 0)
        rel_num += 1
        n_edges = g.number_of_edges()



        # create model
        loss_fcn = torch.nn.MSELoss()

        model = GraphSAGE(num_feats, 64 , 1)


        model_path = save_root + str(cross_id) + '_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.patience, save_path=model_path, min_epoch=args.min_epoch)
        if cuda:
            model.cuda()
            edge_types = edge_types.cuda()

        # use optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # initialize graph
        dur = []
        for epoch in range(args.epochs):
            model.train()
            if epoch >= 3:
                t0 = time.time()
            # train
            logits = model(g, features)
            loss = loss_fcn(logits[train_idx], labels[train_idx].unsqueeze(-1))
            
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)

            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            train_ndcg = get_rank_metrics(logits[train_idx], labels[train_idx], 100)

            # val
            model.eval()
            with torch.no_grad():
                val_logits = model(g, features)
                val_loss, val_ndcg, val_spm, val_overlap, val_medianAE = rank_evaluate(val_logits[val_idx], labels[val_idx].unsqueeze(-1), args.list_num, loss_fcn)

                test_loss, test_ndcg, test_spm, test_overlap, test_medianAE = rank_evaluate(val_logits[test_idx], labels[test_idx].unsqueeze(-1), args.list_num, loss_fcn)

            if args.early_stop:
                stop = stopper.step(-val_loss, epoch, model)  
                if stop:
                    # print('best epoch :', stopper.best_epoch)
                    break

            # if epoch % 100 == 0:
            #     print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TestSPM {:.4f} | TestNDCG {:.4f} | TestOverlap {:.4f}".format(epoch, np.mean(dur), loss.item(), test_spm, test_ndcg, test_overlap))



        # print()
        if args.early_stop:
            model.load_state_dict(torch.load(model_path,map_location='cpu'))
    
        # test
        model.eval()
        with torch.no_grad():
            test_logits = model(g, features)  # [node*1]
            test_loss, test_ndcg, test_spearman, test_overlap, test_medianAE = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), args.list_num, loss_fcn)
            _, test_ndcg_20, _, test_overlap_20, _ = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 20, loss_fcn)
            _, test_ndcg_50, _, test_overlap_50, _  = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 50, loss_fcn)
            _, test_ndcg_200, _, test_overlap_200, _  = \
                rank_evaluate(test_logits[test_idx], labels[test_idx].unsqueeze(-1), 200, loss_fcn)


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
    print('---------------------')
    print(args.save_path)
    print(feat_pretrained_path)
    print('---------------------')
    print('important_ratio | pretrain-patience | lr | loss_ratio')
    print('{} | {} | {} | {}'.format(args.important_ratio, args.pretrain_patience, args.lr, args.loss_eta2))
    print('---------------------')
    print('pretrain | epoch | patience')
    print('{} | {} | {}'.format(args.pretrain_model, args.epochs, args.patience))
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

    parser = argparse.ArgumentParser(description='PreGAT-Pretrain-GraphSAGE')
    parser.add_argument("--dataset", type=str, default='FB15k_rel',
                        help="The input dataset. Can be FB15k")
    parser.add_argument("--data_path", type=str, default='datasets/fb15k_rel.pk',
                        help="path of dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=1000,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--save-path', type=str, default='gat-two_checkpoint.pt',
                        help='the path to save the best model')
    parser.add_argument('--list-num', type=int, default=100)


    #######
    parser.add_argument('--loss-eta1', type=float, default=1.0)
    parser.add_argument('--loss-eta2', type=float, default=1.0)
    parser.add_argument('--important-ratio', type=float, default=0.1)
    parser.add_argument('--pretrain-model', type=str, default='pregat')
    parser.add_argument('--pretrain-patience', type=int, default=20)
 
    args = parser.parse_args()

    main(args)