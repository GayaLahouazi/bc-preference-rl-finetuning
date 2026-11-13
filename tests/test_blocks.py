import gymnasium as gym
import numpy as np
import pytest
import torch

from prefrl import (
    Policy,
    RewardModel,
    SyntheticTeacher,
    Trajectory,
    as_numpy_policy,
    behavioural_cloning,
    collect,
    fit_reward,
    ranking_accuracy,
    subsample,
)

OBS_DIM, ACT_DIM, HORIZON = 4, 2, 20


def random_trajectory(rng, weights):
    """Random steps whose true reward is a hidden linear function of (observation, action)."""
    observations = rng.normal(size=(HORIZON, OBS_DIM)).astype(np.float32)
    actions = rng.normal(size=(HORIZON, ACT_DIM)).astype(np.float32)
    rewards = np.concatenate([observations, actions], axis=1) @ weights
    return Trajectory(observations, actions, rewards.astype(np.float32))


def random_pairs(rng, weights, n):
    return [(random_trajectory(rng, weights), random_trajectory(rng, weights)) for _ in range(n)]


def constant_return(value):
    return Trajectory(np.zeros((1, OBS_DIM), np.float32), np.zeros((1, ACT_DIM), np.float32), np.array([value], np.float32))


def test_rational_teacher_prefers_the_higher_return_and_counts_queries():
    teacher = SyntheticTeacher()
    low, high = constant_return(1.0), constant_return(3.0)
    assert teacher.prefer(high, low) == 1.0
    assert teacher.prefer(low, high) == 0.0
    assert teacher.prefer(low, low) == 0.5
    assert teacher.n_queries == 3


def test_noisy_teacher_follows_bradley_terry():
    teacher = SyntheticTeacher(beta=1.0, seed=0)
    low, high = constant_return(0.0), constant_return(1.0)
    share = np.mean([teacher.prefer(high, low) for _ in range(4000)])
    assert share == pytest.approx(1 / (1 + np.exp(-1.0)), abs=0.03)


def test_teacher_rejects_a_non_positive_beta():
    with pytest.raises(ValueError):
        SyntheticTeacher(beta=0.0)


def test_reward_model_learns_to_rank_unseen_trajectories_from_preferences():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    weights = rng.normal(size=OBS_DIM + ACT_DIM)
    train, held_out = random_pairs(rng, weights, 150), random_pairs(rng, weights, 200)
    model = RewardModel(OBS_DIM, ACT_DIM)
    before = ranking_accuracy(model, held_out)
    losses = fit_reward(model, train, SyntheticTeacher().label(train), epochs=150)
    assert losses[-1] < losses[0]
    assert ranking_accuracy(model, held_out) > max(0.85, before)


def test_behavioural_cloning_recovers_a_linear_expert():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    expert = rng.normal(size=(OBS_DIM, ACT_DIM)).astype(np.float32)

    def demonstration():
        observations = rng.normal(size=(HORIZON, OBS_DIM)).astype(np.float32)
        return Trajectory(observations, observations @ expert, np.zeros(HORIZON, np.float32))

    policy = Policy(OBS_DIM, ACT_DIM)
    losses = behavioural_cloning(policy, [demonstration() for _ in range(50)], epochs=150)
    assert losses[-1] < 0.1 * losses[0]
    fresh = rng.normal(size=(500, OBS_DIM)).astype(np.float32)
    error = np.mean((np.stack([as_numpy_policy(policy)(o) for o in fresh]) - fresh @ expert) ** 2)
    assert error < 0.05 * np.mean((fresh @ expert) ** 2)


def test_rollouts_are_reproducible_and_subsets_are_drawn_without_replacement():
    env = gym.make("Pendulum-v1")
    policy = lambda observation: np.array([-observation[1]], dtype=np.float32)  # noqa: E731
    first, second = collect(env, policy, 3, seed=7), collect(env, policy, 3, seed=7)
    assert len(first[0]) == 200 and first[0].observations.shape == (200, 3) and first[0].actions.shape == (200, 1)
    assert all(np.array_equal(a.rewards, b.rewards) for a, b in zip(first, second))
    subset = subsample(first, 2, seed=0)
    assert len(subset) == 2 and subset[0] is not subset[1]
    with pytest.raises(ValueError):
        subsample(first, 4)
