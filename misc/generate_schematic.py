from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.ml import Sagemaker
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet

with Diagram("Semantic Graph Reinforcement Learning Architecture", show=False, direction="LR"):
    users = Users("Users")
    internet = Internet("Internet")

    with Cluster("Graph Construction"):
        graph_db = RDS("Knowledge Graph DB")
        graph_construction = Server("Graph Construction Service")

    with Cluster("Graph Embedding"):
        graph_embedding = Sagemaker("Graph Embedding Model")

    with Cluster("Policy Learning"):
        policy_learning = Sagemaker("Policy Learning Model")

    with Cluster("Environment Interaction"):
        env_interaction = EC2("Environment Interaction Service")

    with Cluster("Evaluation"):
        evaluation = Server("Evaluation Service")

    users >> internet >> graph_construction >> graph_db
    graph_db >> graph_embedding >> policy_learning >> env_interaction >> evaluation
    evaluation >> users