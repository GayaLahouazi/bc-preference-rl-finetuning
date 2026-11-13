"""Reward model learned from pairwise trajectory comparisons (Bradley-Terry)."""

import torch
from torch import nn


class RewardModel(nn.Module):
    """r(s, a): a small MLP. The predicted return of a trajectory is the sum over its steps."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observations, actions):
        return self.net(torch.cat([observations, actions], dim=-1)).squeeze(-1)

    def predicted_return(self, trajectory):
        observations = torch.as_tensor(trajectory.observations, dtype=torch.float32)
        actions = torch.as_tensor(trajectory.actions, dtype=torch.float32)
        return self(observations, actions).sum()


def preference_loss(model, pairs, labels):
    """Bradley-Terry negative log-likelihood.

    P(a preferred to b) = sigmoid(R(a) - R(b)) with R the predicted return, so the loss is a
    binary cross-entropy on the return gap. A label of 0.5 (tie) pulls the two returns together.
    """
    gaps = torch.stack([model.predicted_return(a) - model.predicted_return(b) for a, b in pairs])
    targets = torch.as_tensor(labels, dtype=torch.float32)
    return nn.functional.binary_cross_entropy_with_logits(gaps, targets)


def fit_reward(model, pairs, labels, epochs=200, lr=3e-3, weight_decay=1e-4):
    """Fit the reward model on labelled pairs (full batch). Returns the loss per epoch."""
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = preference_loss(model, pairs, labels)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def ranking_accuracy(model, pairs):
    """Share of pairs whose order by predicted return matches the order by true return."""
    correct, counted = 0, 0
    for a, b in pairs:
        if a.true_return == b.true_return:
            continue
        predicted = model.predicted_return(a) > model.predicted_return(b)
        correct += int(bool(predicted) == (a.true_return > b.true_return))
        counted += 1
    return correct / counted if counted else float("nan")
