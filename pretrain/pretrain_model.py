import torch
from torch import nn
import torch.nn.functional as F
import dgl
from utils.licap_utils import return_centroids, return_imp_node_centroid

class Predicate_GAT_Layer(nn.Module):
    def __init__(self, g, in_dim , out_dim, rel_dim):
        super(Predicate_GAT_Layer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim+rel_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        z2 = torch.cat([edges.src['z'], edges.data['p'], edges.dst['z']], dim=1) # [num_edges, 8+10+8]
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # 归一化每一条入边的注意力系数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, h, p):
        z = self.fc(h)  # h [num_nodes, num_feats] -> [num_nodes, hid_num]
        self.g.ndata['z'] = z # 每个节点的特征
        self.g.edata['p'] = p # 每个边的特征


        self.g.apply_edges(self.edge_attention) # 为每一条边获得其注意力系数
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHead_Predicate_GAT_Layer(nn.Module):
    def __init__(self, g, in_dim , out_dim, rel_dim, num_heads=1, merge='cat'):
        super(MultiHead_Predicate_GAT_Layer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(Predicate_GAT_Layer(g, in_dim, out_dim, rel_dim))
        self.merge = merge


    def forward(self, h, p):
        head_out = [attn_head(h, p) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class Predicate_GAT(nn.Module):
    def __init__(self, args, g, in_dim, hidden_dim , out_dim, num_heads, rel_num, rel_dim, 
                 loss_function=None,
                 loss_1_fcn=None,
                 loss_2_fcn=None,
                 train_important_idx=None,
                 train_normal_idx=None,
                 imp_bin_idx2node_idx=None,
                 imp_node_idx2bin_idx=None,
                 imp_node_coeff=None):
        super(Predicate_GAT, self).__init__()
        self.layer1 = MultiHead_Predicate_GAT_Layer(g , in_dim, hidden_dim, rel_dim, num_heads)
        self.layer2 = MultiHead_Predicate_GAT_Layer(g, hidden_dim*num_heads, out_dim, rel_dim, 1)

        self.loss_fn = loss_function
        self.loss_1_fcn = loss_1_fcn
        self.loss_2_fcn = loss_2_fcn
        

        self.train_important_idx = train_important_idx
        self.train_normal_idx = train_normal_idx
        
        self.imp_bin_idx2node_idx = imp_bin_idx2node_idx
        self.imp_node_idx2bin_idx = imp_node_idx2bin_idx
        self.imp_node_coeff = imp_node_coeff

        self.loss_eta1 = args.loss_eta1
        self.loss_eta2 = args.loss_eta2

        # Relation Embedding
        self.rel_emb = nn.Embedding(rel_num, rel_dim) 


    def forward(self, node_feats, edge_types, idx=None):

        edge_feats = self.rel_emb(edge_types) # [num_edges, rel_dim]

        h = self.layer1(node_feats, edge_feats)
        h = F.elu(h)
        h = self.layer2(h, edge_feats)

        # caculate_loss            
        embed_all = h # [N_node, dim]
                     
        imp_bin_centroids, bin2centroids = return_centroids(embed_all, self.imp_bin_idx2node_idx, self.train_important_idx) # [bin_num, dim]
                
        imp_node_centroid = return_imp_node_centroid(self.imp_node_idx2bin_idx, bin2centroids, self.train_important_idx) # [imp_num, dim]    
        embed_normal = embed_all[self.train_normal_idx] ### mod sample by constant
                
        loss_1 = self.loss_eta1 * self.loss_1_fcn(embed_all[self.train_important_idx], embed_normal)

        loss_2 = self.loss_eta2 * self.loss_2_fcn(embed_all[self.train_important_idx], imp_bin_centroids, imp_node_centroid, self.imp_node_coeff)

        loss = loss_1 + loss_2

        return h, loss, loss_1, loss_2

# batch-version