import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim , out_dim):
        super(GraphSAGE, self).__init__()
        self.layer1 = SAGEConv(in_dim, hidden_dim, 'mean')
        self.layer2 = SAGEConv(hidden_dim, out_dim, 'mean')

    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        logits = self.layer2(g, h)

        return logits