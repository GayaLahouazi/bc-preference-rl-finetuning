"""Offline stage: behavioural cloning on reward-free expert demonstrations."""

import numpy as np
import torch
from torch import nn


class Policy(nn.Module):
    """Deterministic MLP policy for continuous actions."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, observations):
        return self.net(observations)


def behavioural_cloning(policy, demonstrations, epochs=200, lr=1e-3, batch_size=256, seed=0):
    """Regress the expert's actions on its observations (MSE). Rewards are not used.

    Returns the mean loss per epoch.
    """
    observations = torch.as_tensor(np.concatenate([d.observations for d in demonstrations]), dtype=torch.float32)
    actions = torch.as_tensor(np.concatenate([d.actions for d in demonstrations]), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(observations), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = nn.functional.mse_loss(policy(observations[batch]), actions[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
    return losses


def as_numpy_policy(policy):
    """Wrap a torch policy as a function observation -> action usable by `rollout`."""

    @torch.no_grad()
    def act(observation):
        return policy(torch.as_tensor(observation, dtype=torch.float32)).numpy()

    return act
