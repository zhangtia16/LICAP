import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim , out_dim, rel_num):
        super(RGCN, self).__init__()
        self.layer1 = RelGraphConv(in_dim, hidden_dim, rel_num)
        self.layer2 = RelGraphConv(hidden_dim, out_dim, rel_num)

    def forward(self, g, features, edge_types):
        h = F.relu(self.layer1(g, features, edge_types))
        logits = self.layer2(g, h, edge_types)

        return logits