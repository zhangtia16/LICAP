import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SGConv

class SGC(nn.Module):
    def __init__(self, in_dim, hidden_dim , out_dim):
        super(SGC, self).__init__()
        self.layer1 = SGConv(in_dim, hidden_dim)
        self.layer2 = SGConv(hidden_dim, out_dim)

    def forward(self, g, features):
        h = F.relu(self.layer1(g, features))
        logits = self.layer2(g, h)

        return logits