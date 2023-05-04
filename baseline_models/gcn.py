import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim , out_dim):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        logits = self.layer2(g, h)

        return logits