"""Synthetic preference teacher: stands in for the human who compares two trajectories."""

import numpy as np


class SyntheticTeacher:
    """Answers "which of these two trajectories is better?" from their true returns.

    With `beta=None` the teacher is perfectly rational: the higher return always wins and a
    tie gives 0.5. With a finite `beta` it follows the Bradley-Terry model,
    P(a preferred to b) = sigmoid(beta * (R(a) - R(b))), so close trajectories get noisy labels.

    The number of answered queries is counted: it is the cost the experiments report.
    """

    def __init__(self, beta=None, seed=0):
        if beta is not None and beta <= 0:
            raise ValueError("beta must be positive (or None for a perfectly rational teacher)")
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.n_queries = 0

    def prefer(self, a, b):
        """Return 1.0 if `a` is preferred, 0.0 if `b` is, 0.5 on a tie (rational teacher only)."""
        self.n_queries += 1
        gap = a.true_return - b.true_return
        if self.beta is None:
            return 0.5 if gap == 0 else float(gap > 0)
        probability = 1.0 / (1.0 + np.exp(-self.beta * gap))
        return float(self.rng.random() < probability)

    def label(self, pairs):
        """Label a list of (a, b) trajectory pairs."""
        return np.asarray([self.prefer(a, b) for a, b in pairs], dtype=np.float32)
