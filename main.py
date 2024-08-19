import mlflow
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from typing import Dict
from semantic_graph_rl.data.graph_dataset import GraphDataModule, create_heterogeneous_graph, generate_hetero_random_walk, concatenate_hetero_embeddings
from semantic_graph_rl.models.graph_embeddings import HeterogeneousGraphEmbedding, MambaModule
from semantic_graph_rl.models.lightning_rl_agent import LightningGraphRLPolicy, LightningGraphRLAgent
from semantic_graph_rl.utils.nlp_utils import process_text, create_embeddings
from semantic_graph_rl.utils.evaluation import evaluate_graph_expressivity, evaluate_graph_structure, evaluate_rl_performance
from semantic_graph_rl.utils.graph_utils import create_initial_knowledge_graph
import pytorch_lightning as pl

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment.name)

    with mlflow.start_run():
        # Data preparation
        # TODO: Implement data loading and preprocessing
        graphs = cfg.data.graphs
        labels = cfg.data.labels
        in_feats_dict = cfg.data.in_feats_dict
        hidden_feats = cfg.data.hidden_feats
        out_feats = cfg.data.out_feats
        action_space = cfg.data.action_space

        # Create graph dataset and data module
        data_module = GraphDataModule(graphs, labels)

        # Create graph embeddings
        graph_embedding_model = HeterogeneousGraphEmbedding(in_feats_dict, hidden_feats, out_feats)
        mamba_module = MambaModule(out_feats, d_state=cfg.mamba.d_state, d_conv=cfg.mamba.d_conv, expand=cfg.mamba.expand)

        # Create initial knowledge graph
        initial_graph = create_initial_knowledge_graph(in_feats_dict)

        # Create RL policy and agent
        policy = LightningGraphRLPolicy(in_feats_dict, hidden_feats, out_feats, action_space)
        rl_agent = LightningGraphRLAgent(policy, initial_graph)

        # Train the agent
        trainer = pl.Trainer(max_epochs=cfg.experiment.max_epochs, gpus=cfg.experiment.gpus if torch.cuda.is_available() else 0)
        trainer.fit(rl_agent, data_module)

        # Evaluation
        graph = rl_agent.knowledge_graph
        embeddings = graph_embedding_model(graph)
        walks = generate_hetero_random_walk(graph)
        concatenated_embeddings = concatenate_hetero_embeddings(graph, walks)
        mamba_embeddings = mamba_module(concatenated_embeddings)

        expressivity_score = evaluate_graph_expressivity(graph, mamba_embeddings)
        structure_metrics = evaluate_graph_structure(graph)
        rl_performance = evaluate_rl_performance(rl_agent)

        # Log metrics
        mlflow.log_metric("expressivity_score", expressivity_score)
        for key, value in structure_metrics.items():
            mlflow.log_metric(f"structure_{key}", value)
        for key, value in rl_performance.items():
            mlflow.log_metric(f"rl_{key}", value)

        # Save models
        mlflow.pytorch.log_model(graph_embedding_model, "graph_embedding_model")
        mlflow.pytorch.log_model(mamba_module, "mamba_module")
        mlflow.pytorch.log_model(rl_agent, "rl_agent")


if __name__ == "__main__":
    main()
