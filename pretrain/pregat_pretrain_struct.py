import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import pickle as pk


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed, load_data
from collections import Counter
import matplotlib.pyplot as plt

from pretrain.licap_loss import Top_Bin_Loss, Finer_Bin_Loss
from utils.licap_utils import find_imp_idx, get_imp_bin_idx2node_idx, return_bin_idx2bin_coeff, return_imp_node_coeff
from pretrain.pretrain_model import Predicate_GAT

def main(args):
    print(args)

    set_random_seed(0)

    print(args.save_path)
    # set the save path
    dataset_name = args.dataset.rstrip('_two').rstrip('_rel').lower()
    save_root = 'pretrain/results/' + dataset_name + '_pregat_pretrain_struct/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    

    pretrain_data_root = '/workspace1/zty/pretrain_data/pregat_pretrain_struct/'
    imp_ratio_path = 'imp_ratio_'+ str(args.important_ratio) + '/'
    pretrain_patience_path = 'patience_'+ str(args.pretrain_patience) + '/'
    dataset_path = dataset_name + '/'
    feat_save_root =  pretrain_data_root + imp_ratio_path + pretrain_patience_path + dataset_path
    if not os.path.exists(feat_save_root):
        os.makedirs(feat_save_root)


    

        # if args.loss_eta2 == 0 and args.loss_eta1 == 1:
        #     loss_sign = 'l1'
        # elif args.loss_eta2 == 1 and args.loss_eta1 == 0:
        #     loss_sign = 'l2'
        # else:
        #     print('wrong input')

        # feat_pretrained_path = 'pretrain/pretrain_loss/' + args.dataset + '_relgat_features_pretrained_lr' + str(args.lr) + loss_sign +'_' + str(cross_id) + '.pkl'



    for cross_id in range(args.cross_num):


        feat_pretrained_path = feat_save_root + dataset_name + '_struct_pregat_pretrained_lr' + str(args.lr) + 'r' + str(args.loss_eta2) + '_' + str(cross_id) + '.pkl'

                
        g, edge_types, _, rel_num, struct_feat, semantic_feat, labels, train_idx, val_idx, test_idx = \
            load_data(args.data_path, args.dataset, cross_id)

        ###
        train_important_idx, train_normal_idx, important_ratio, normal_ratio, train_important_border, train_normal_border = \
            find_imp_idx(labels, train_idx, val_idx, test_idx, args.list_num, args.important_ratio, args.normal_important_ratio)


        imp_bin_idx2node_idx, imp_bin_idx2isvalid = get_imp_bin_idx2node_idx(labels, args.dataset, train_idx, train_important_border)  
        
        imp_node_idx2bin_idx={value:key for key,value_list in imp_bin_idx2node_idx.items() for value in value_list }
        imp_bin_idx2coeff = return_bin_idx2bin_coeff(imp_bin_idx2isvalid)
        
        imp_node_coeff = return_imp_node_coeff(train_important_idx, imp_node_idx2bin_idx, imp_bin_idx2coeff)

         
        
        # cuda
        torch.cuda.set_device(args.gpu)
        struct_feat = struct_feat.cuda()
        labels = labels.cuda()   
        imp_node_coeff = imp_node_coeff.cuda() 
          

        # g = data[0]
        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            g = g.int().to(args.gpu)

        num_feats = struct_feat.shape[1]
        n_edges = g.number_of_edges()

        if cross_id == 0:
            print('---------Dataset Statistics---------')
            print("Edges {} | Rel num {} | Train samples {} | Val samples {} | Test samples {} | Important Train samples {} | Important ratio {:.4f}".format(n_edges,rel_num,len(train_idx),len(val_idx),len(test_idx),len(train_important_idx),important_ratio))
            print('-------------Pretraining------------')
        
        print('cross:{}'.format(cross_id))
        print('------------')
        
        # add self loop
        g = dgl.add_self_loop(g)
        new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
        edge_types = torch.cat([edge_types, new_edge_types], 0)
        rel_num += 1
        n_edges = g.number_of_edges()

        

        print('Model Pretraining')

        # create model
        loss_fcn = torch.nn.MSELoss()
        loss_1_fcn = Top_Bin_Loss(args.temp)
        loss_2_fcn = Finer_Bin_Loss(args.temp)

        hidden_dim = 8
        num_heads = 8
        rel_dim = 10

        model = Predicate_GAT(args, g, num_feats, hidden_dim , args.pretrain_dim, num_heads, rel_num, rel_dim, 
            loss_fcn, loss_1_fcn,loss_2_fcn, 
            train_important_idx, train_normal_idx , imp_bin_idx2node_idx, imp_node_idx2bin_idx, imp_node_coeff)


    


        model_path = save_root + str(cross_id) + '_' + args.save_path
        if args.early_stop:
            stopper = EarlyStopping_simple(patience=args.pretrain_patience, save_path=model_path, min_epoch=args.min_epoch)
            
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
            features_pretrained, loss, loss_1, loss_2 = model(struct_feat, edge_types,  train_idx)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            if epoch >= 3:
                dur.append(time.time() - t0)

            # # val
            # model.eval()
            # with torch.no_grad():
            #     features_pretrained, loss, loss_1, loss_2 = model(struct_feat, edge_types, val_idx)

            if args.early_stop:
                stop = stopper.step(-loss, epoch, model)
                if stop:
                    # print('best epoch :', stopper.best_epoch)
                    break

            if epoch % 10 == 0:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | loss_1 {:.4f} | loss_2 {:.4f}".format(epoch, np.mean(dur), loss.item(), loss_1.item(),loss_2.item()))
     


        print()

        print(feat_pretrained_path)
        print('pregat_features_pretrained.shape:',features_pretrained.shape)     
        pk.dump(features_pretrained, open(feat_pretrained_path, 'wb'))

        print('Pretrained vectors saved!')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RELGAT-Pretrain')
    parser.add_argument("--dataset", type=str, default='GA16k_rel_two',
                        help="The input dataset.")
    parser.add_argument("--data_path", type=str, default='datasets/ga16k_rel.pk',
                        help="path of dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of pretraining epochs")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=4,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=1000,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--scale', action="store_true", default=False,
                        help="utilize centrality to scale scores")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='gat-two_checkpoint.pt',
                        help='the path to save the best model')

    parser.add_argument('--loss-lambda', type=float, default=0.5,
                        help='the weight to add unsupervised loss')
    parser.add_argument('--list-num', type=int, default=100)


    #######
    parser.add_argument('--loss-eta1', type=float, default=1.0)
    parser.add_argument('--loss-eta2', type=float, default=1.0)
    parser.add_argument('--important-ratio', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--normal-important-ratio', type=float, default=5.0)
    parser.add_argument('--pretrain-patience', type=int, default=20)
    parser.add_argument('--pretrain-dim', type=int, default=64)
 
    args = parser.parse_args()

    main(args)
