import matplotlib.pyplot as plt
import networkx as nx

def generate_schematic():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with attributes
    G.add_node("Graph Construction", pos=(0, 4))
    G.add_node("Graph Embedding", pos=(2, 4))
    G.add_node("Policy Learning", pos=(4, 4))
    G.add_node("Environment Interaction", pos=(6, 4))
    G.add_node("Evaluation", pos=(8, 4))

    # Add edges
    G.add_edge("Graph Construction", "Graph Embedding")
    G.add_edge("Graph Embedding", "Policy Learning")
    G.add_edge("Policy Learning", "Environment Interaction")
    G.add_edge("Environment Interaction", "Evaluation")

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='skyblue')

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20)

    # Draw the labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Add title
    plt.title("Semantic Graph Reinforcement Learning Schematic")

    # Save the figure
    plt.savefig("static/semantic_graph_rl_schematic.png")
    plt.show()

if __name__ == "__main__":
    generate_schematic()
