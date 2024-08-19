import shutil
import sys
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.generic.ml import ML
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet

# Check if Graphviz is installed
if not shutil.which("dot"):
    print("Error: Graphviz is not installed or not found in PATH.", file=sys.stderr)
    sys.exit(1)

with Diagram("Semantic Graph Reinforcement Learning Architecture", show=False, direction="LR"):
    users = Users("Users")
    internet = Internet("Internet")

    with Cluster("Graph Construction"):
        graph_db = SQL("Knowledge Graph DB")
        graph_construction = Server("Graph\nConstruction\nService")

    with Cluster("Graph Embedding"):
        graph_embedding = ML("Graph\nEmbedding\nModel")

    with Cluster("Policy Learning"):
        policy_learning = ML("Policy\nLearning\nModel")

    with Cluster("Environment Interaction"):
        env_interaction = Rack("Environment\nInteraction\nService")

    with Cluster("Evaluation"):
        evaluation = Server("Evaluation\nService")

    users >> internet >> graph_construction >> graph_db
    graph_db >> graph_embedding >> policy_learning >> env_interaction >> evaluation
    evaluation >> users
