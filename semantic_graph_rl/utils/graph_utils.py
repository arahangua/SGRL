import torch
from typing import Dict

def create_initial_knowledge_graph(in_feats_dict: Dict[str, int]) -> Dict[str, torch.Tensor]:
    # Create an initial knowledge graph based on the input features
    initial_graph = {}
    for node_type, feat_dim in in_feats_dict.items():
        initial_graph[node_type] = torch.randn(10, feat_dim)  # Start with 10 nodes per type
    return initial_graph
