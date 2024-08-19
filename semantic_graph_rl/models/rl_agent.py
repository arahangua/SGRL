import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

class GraphRLPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(GraphRLPolicy, self).__init__(*args, **kwargs)
        # Custom policy network architecture
        self.graph_embedding = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.action_net = nn.Linear(64, self.action_space.n)
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs, deterministic=False):
        # Custom forward pass
        features = self.extract_features(obs)
        embeddings = self.graph_embedding(features)
        
        action_logits = self.action_net(embeddings)
        value = self.value_net(embeddings)

        return action_logits, value

class GraphRLAgent(PPO):
    def __init__(self, policy, env, **kwargs):
        super(GraphRLAgent, self).__init__(policy, env, **kwargs)
