from diagrams import Diagram, Cluster, Edge
from diagrams.aws.database import RDS
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.ml import Sagemaker as ML
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet
from diagrams.onprem.compute import Server
from diagrams.custom import Custom

with Diagram("Semantic Graph Reinforcement Learning Architecture", show=False, direction="LR"):
    users = Users("Users")                                                                                                                                                                      
    internet = Internet("Internet")                                                                                                                                                             
                                                                                                                                                                                                 
    with Cluster("Graph Construction"):                                                                                                                                                         
        graph_db = RDS("Knowledge Graph DB")                                                                                                                                                    
        graph_construction = Server("Graph\nConstruction\nService")                                                                                                                             
                                                                                                                                                                                                
    with Cluster("Graph Embedding"):                                                                                                                                                            
        graph_embedding = ML("Graph\nEmbedding\nModel")                                                                                                                                         
                                                                                                                                                                                                
    with Cluster("Policy Learning"):                                                                                                                                                            
        policy_learning = ML("Policy\nLearning\nModel")                                                                                                                                         
                                                                                                                                                                                                
    with Cluster("Environment Interaction"):                                                                                                                                                    
        env_interaction = Server("Environment\nInteraction\nService")                                                                                                                             
                                                                                                                                                                                                
    with Cluster("Evaluation"):                                                                                                                                                                 
        evaluation = Server("Evaluation\nService")                                                                                                                                              
                                                                                                                                                                                                 
    users >> internet >> graph_construction >> graph_db                                                                                                                                         
    graph_db >> graph_embedding >> policy_learning >> env_interaction >> evaluation                                                                                                             
    evaluation >> users 
