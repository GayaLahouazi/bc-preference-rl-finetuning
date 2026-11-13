"""Building blocks for BC + preference-based RL fine-tuning experiments (work in progress)."""

from .bc import Policy, as_numpy_policy, behavioural_cloning
from .reward_model import RewardModel, fit_reward, preference_loss, ranking_accuracy
from .teacher import SyntheticTeacher
from .trajectories import Trajectory, collect, rollout, subsample

__all__ = [
    "Policy",
    "RewardModel",
    "SyntheticTeacher",
    "Trajectory",
    "as_numpy_policy",
    "behavioural_cloning",
    "collect",
    "fit_reward",
    "preference_loss",
    "ranking_accuracy",
    "rollout",
    "subsample",
]
