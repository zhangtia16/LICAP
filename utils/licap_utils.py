import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from scipy.stats import binom

def get_bin_length(dataset_name):
    if dataset_name.startswith('FB15k'):
        bin_length = 1.0
    elif dataset_name.startswith('IMDB_S'):
        bin_length = 1.0
    elif dataset_name.startswith('TMDB'):
        bin_length = 0.5
    elif dataset_name.startswith('GA16k'):
        bin_length = 1.0
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset_name))
    return bin_length

def visualize_labels(labels, dataset_name, figure_name, save_path):
    bin_length = get_bin_length(dataset_name)
    labels = np.array(labels.cpu())/bin_length
    _, ax = plt.subplots(figsize=(6, 3), sharex='all', sharey='all')
    ax.hist(labels, bins=range(int(np.floor(min(labels))), int(np.ceil(max(labels))) + 1, 1))

    plt.title(figure_name)
    plt.tight_layout()
    plt.savefig(save_path)


def get_imp_bin_idx2node_idx(labels, dataset_name, target_idx, train_important_border):
    bin_length = get_bin_length(dataset_name)
    
    labels = np.array(labels - train_important_border) 
    
    imp_bin_idx2node_idx = {}
    bin_idx = 0
    while np.max(labels[target_idx]) > bin_idx*bin_length:
        if bin_idx == 0:
            imp_bin_idx2node_idx[bin_idx] = np.intersect1d(np.where((bin_idx*bin_length <= labels) & (labels <= (bin_idx+1)*bin_length))[0],target_idx,assume_unique=True)
        else:
            imp_bin_idx2node_idx[bin_idx] = np.intersect1d(np.where((bin_idx*bin_length < labels) & (labels <= (bin_idx+1)*bin_length))[0],target_idx,assume_unique=True)
        bin_idx += 1
    
    imp_bin_idx2isvalid = {}
    for key in imp_bin_idx2node_idx.keys():
        if len(imp_bin_idx2node_idx[key]) > 0:
            imp_bin_idx2isvalid[key] = True
        else:
            imp_bin_idx2isvalid[key] = False
    return imp_bin_idx2node_idx, imp_bin_idx2isvalid


def return_centroids(embed_all, bin_idx2node_idx, train_imp_idx):
    sign = True
    bin2centroids = {}
    for bin_idx in bin_idx2node_idx.keys():
        
        node_idx = bin_idx2node_idx[bin_idx]
        
        if (len(node_idx) > 0):
            centroid = torch.mean(embed_all[node_idx],dim=0).unsqueeze(0)
            bin2centroids[bin_idx] = centroid
            
            if sign:
                imp_bin_centroids = centroid
                sign = False
            else:
                imp_bin_centroids = torch.cat((imp_bin_centroids, centroid), dim=0)
        else:
            bin2centroids[bin_idx] = None
    return imp_bin_centroids, bin2centroids


def return_bin_idx2bin_coeff(bin_idx2isvalid):
    bin_min = min(bin_idx2isvalid.keys())
    bin_max = max(bin_idx2isvalid.keys())
    print('bin_max , bin_min',bin_max , bin_min)
    bin_coeff = {}
    for bin_idx in bin_idx2isvalid.keys():
        if bin_idx2isvalid[bin_idx]:
            N = 2*max(bin_idx, (bin_max-bin_idx))
            coeff = []
            for target_bin_idx in bin_idx2isvalid.keys():
                if bin_idx2isvalid[target_bin_idx]:
                    coeff.append(binom.pmf(k=N/2+abs(target_bin_idx-bin_idx), n=N, p=0.5))
            coeff = torch.tensor(coeff)
            coeff = (coeff/torch.max(coeff)).unsqueeze(0)
            bin_coeff[bin_idx] = coeff
        else:
            bin_coeff[bin_idx] = None

    return bin_coeff # [train_num, bin_num]

def return_imp_node_centroid(train_node_idx2bin_idx, bin2centroids, imp_train_idx):
    sign = True
    for train_node_idx in imp_train_idx:
        bin_idx = train_node_idx2bin_idx[train_node_idx]
        centroid = bin2centroids[bin_idx]
        if sign:
            imp_node_centroid = centroid
            sign = False
        else:
            imp_node_centroid = torch.cat((imp_node_centroid, centroid), dim=0)
    return imp_node_centroid




def return_imp_node_coeff(imp_idx, imp_node_idx2bin_idx, bin_idx2coeff):
    sign = True
    for train_node_idx in imp_idx:
        node_coeff = bin_idx2coeff[imp_node_idx2bin_idx[train_node_idx]]
        if sign:
            imp_node_coeff = node_coeff
            sign = False
        else:
            imp_node_coeff = torch.cat((imp_node_coeff, node_coeff), dim=0)
    return imp_node_coeff

        

        

# licap

def find_imp_idx(labels, train_idx, val_idx, test_idx, list_num, important_ratio, normal_important_ratio):
    
    train_num = len(train_idx)
    test_num = len(test_idx)
    
    normal_ratio = important_ratio * normal_important_ratio

    train_label_sort, _ =  torch.sort(labels[train_idx], descending=True)

    train_important_border = train_label_sort[int(np.round(important_ratio*train_num))]
    train_important_idx = np.intersect1d(np.where((labels >= train_important_border))[0], train_idx)

    train_normal_border = train_label_sort[np.min([int(np.round(normal_ratio*train_num)),len(train_idx)-1])]

    train_normal_idx = np.intersect1d(np.where((labels < train_important_border) & (labels >= train_normal_border))[0], train_idx)


    train_normal_idx = np.random.choice(train_normal_idx,size=int(np.round(0.5*len(train_normal_idx))),replace=False)
    
    
    
    return train_important_idx, train_normal_idx, important_ratio, normal_ratio, train_important_border, train_normal_border





def calculate_anchor_cos_sim(embed, anchor_embed, bin_idx2indices, save_path):
    bin_idx2cos_sim = {}
    for bin_idx in bin_idx2indices.keys():
        indices = bin_idx2indices[bin_idx]
        cos_sims = F.cosine_similarity(embed[indices],anchor_embed,dim=1)
        bin_idx2cos_sim[bin_idx] = torch.mean(cos_sims)

    for bin_idx in bin_idx2cos_sim.keys():
        plt.title('Mean Cosine Similarity')
        if bin_idx == max(bin_idx2cos_sim.keys()):
            c = 'red'
        else:
            c = 'blue'
        plt.bar(bin_idx,bin_idx2cos_sim[bin_idx].cpu(), color=c)
        plt.savefig(save_path)
    return bin_idx2cos_sim



def embedding_tsne(embedding, labels, idx, train_node_idx2bin_idx, labels_important, train_important_border, train_normal_border, dataset_name, cross_id, figure_type,save_path=None):
    if save_path == None:
        save_path='./pics/tsne_cl_train_' + dataset_name + '_' + figure_type +  '_' + str(cross_id) + '.png'

    labels_tsne = np.array(labels.cpu())
    labels_important = np.array(labels_important.cpu())
    embedding = embedding.cpu().detach().numpy()
    

    
    if figure_type == 'important':
        # labels_tsne = np.append(labels_tsne,[-1])
        # embedding = torch.cat((embedding, important_anchor_embed.unsqueeze(0)),dim=0)
        color_list = [2*np.min(labels_important)-np.max(labels_important)] * (len(labels_tsne))
        for idx in range(len(labels_tsne)):
            if labels_tsne[idx] >= train_important_border:
                color_list[idx] = labels_tsne[idx]
            else:
                pass
        mycmap = plt.cm.get_cmap('seismic')
    elif figure_type == 'bin':
        bin_idx_list = []
        for key in idx: 
            bin_idx_list.append(train_node_idx2bin_idx[key])
        color_list = np.round(np.array(bin_idx_list))
        mycmap = plt.cm.get_cmap('tab20')
    else:
        color_list = labels_tsne
        mycmap = None
    # if type == 'important':
    #     for idx in range(len(labels_tsne)):
    #         if labels_tsne[idx] >= train_important_border:
    #             labels_tsne[idx] += 30
    #         elif

    tsne = TSNE(n_components=2,random_state=0,init='pca',learning_rate='auto')
    x_tsne = tsne.fit_transform(embedding)
    x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
    x_tsne = (x_tsne - x_min) / (x_max - x_min)

    plt.figure()
    plt.xticks([]) 
    plt.yticks([])
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=color_list, s=5, cmap=mycmap)
    plt.savefig(save_path,dpi=500)


