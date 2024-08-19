import torch
import dgl
import numpy as np

def create_sample_knowledge_graph():
    # Define node types
    num_person = 5
    num_movie = 3
    num_genre = 2

    # Create a heterogeneous graph
    graph_data = {
        ('person', 'watches', 'movie'): (torch.randint(0, num_person, (10,)), torch.randint(0, num_movie, (10,))),
        ('movie', 'belongs_to', 'genre'): (torch.randint(0, num_movie, (5,)), torch.randint(0, num_genre, (5,))),
        ('person', 'likes', 'genre'): (torch.randint(0, num_person, (8,)), torch.randint(0, num_genre, (8,)))
    }

    g = dgl.heterograph(graph_data)

    # Add node features
    g.nodes['person'].data['feat'] = torch.randn(num_person, 10)  # 10-dimensional feature for each person
    g.nodes['movie'].data['feat'] = torch.randn(num_movie, 15)   # 15-dimensional feature for each movie
    g.nodes['genre'].data['feat'] = torch.randn(num_genre, 5)    # 5-dimensional feature for each genre

    # Add edge features
    g.edges['watches'].data['rating'] = torch.randn(g.number_of_edges('watches'), 1)
    g.edges['belongs_to'].data['strength'] = torch.randn(g.number_of_edges('belongs_to'), 1)
    g.edges['likes'].data['score'] = torch.randn(g.number_of_edges('likes'), 1)

    return g

if __name__ == "__main__":
    # Create and print the sample knowledge graph
    sample_graph = create_sample_knowledge_graph()
    print(sample_graph)
    print("Number of nodes:", sample_graph.number_of_nodes())
    print("Number of edges:", sample_graph.number_of_edges())
    print("Node types:", sample_graph.ntypes)
    print("Edge types:", sample_graph.etypes)
    print("Node feature shapes:")
    for ntype in sample_graph.ntypes:
        print(f"  {ntype}: {sample_graph.nodes[ntype].data['feat'].shape}")
    print("Edge feature shapes:")
    for etype in sample_graph.etypes:
        print(f"  {etype}: {sample_graph.edges[etype].data.keys()}")
