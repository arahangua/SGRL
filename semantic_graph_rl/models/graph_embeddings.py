import dgl
import torch
import torch.nn as nn

class GraphEmbedding(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

class HeterogeneousGraphEmbedding(nn.Module):
    def __init__(self, in_feats_dict, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(in_feats_dict[etype], hidden_feats)
            for etype in in_feats_dict
        })
        self.conv2 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.GraphConv(hidden_feats, out_feats)
            for etype in in_feats_dict
        })

    def forward(self, g, features_dict):
        h = self.conv1(g, features_dict)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h
