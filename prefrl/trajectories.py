"""Trajectories: container, rollouts in a Gymnasium environment, demonstration subsets."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """One episode.

    `rewards` holds the true environment rewards. The learner never reads them: they are
    used only by the synthetic teacher (to answer preference queries) and by the evaluation
    (to compute the regret).
    """

    observations: np.ndarray  # (T, obs_dim)
    actions: np.ndarray  # (T, act_dim)
    rewards: np.ndarray  # (T,)

    def __len__(self):
        return len(self.rewards)

    @property
    def true_return(self):
        return float(np.sum(self.rewards))


def rollout(env, policy, seed=None, max_steps=None):
    """Run one episode of `policy` (a function observation -> action) in `env`."""
    observation, _ = env.reset(seed=seed)
    observations, actions, rewards = [], [], []
    done = False
    while not done:
        action = np.asarray(policy(observation), dtype=np.float32)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        observation = next_observation
        done = terminated or truncated or (max_steps is not None and len(rewards) >= max_steps)
    return Trajectory(np.stack(observations), np.stack(actions), np.asarray(rewards, dtype=np.float32))


def collect(env, policy, n_trajectories, seed=0, max_steps=None):
    """Collect `n_trajectories` episodes, seeded seed, seed + 1, ... so datasets are reproducible."""
    return [rollout(env, policy, seed=seed + i, max_steps=max_steps) for i in range(n_trajectories)]


def subsample(trajectories, n, seed=0):
    """Draw `n` trajectories without replacement (the offline datasets of 1, 5, 20, 100 demos)."""
    if n > len(trajectories):
        raise ValueError("asked for %d trajectories, only %d available" % (n, len(trajectories)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(trajectories), size=n, replace=False)
    return [trajectories[i] for i in indices]
