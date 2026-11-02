# Fine-tuning behavioural cloning policies with preference-based RL

> **Status: work in progress.** Repository set up; implementation and experiments are under way.

Course project for **IAR** (*Intelligence Artificielle pour la Robotique*, AI for robotics), M2 AI2D, Sorbonne Université, 2026–2027. Supervisor: Silvia Tulli.

It studies the approach of:

> M. Macuglia, P. Friedrich, G. Ramponi. *Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning.* ICLR 2026.
> [OpenReview](https://openreview.net/forum?id=oIiQZfnSxP) · [arXiv:2509.26605](https://arxiv.org/abs/2509.26605) · [authors' code (BRIDGE)](https://github.com/pfriedric/bridge)

## The paper in brief

Robots are often taught by imitation: a policy is trained by **behavioural cloning (BC)** on expert demonstrations that carry no reward. BC gives a safe starting point, but it is limited by the quality and coverage of the demonstrations. Writing a reward function to improve it further is hard. Instead, a human can compare pairs of trajectories and say which one is better.

The paper proposes a **two-stage** framework:

1. **Offline:** learn an initial policy from reward-free expert demonstrations.
2. **Online:** fine-tune it with **preference-based RL**. A reward model is learned from pairwise trajectory comparisons (Bradley-Terry model), and the policy is optimised against it.

Their algorithm, **BRIDGE**, uses the demonstrations to build a confidence set over policies that constrains and speeds up online preference learning. The paper's central theoretical claim is a regret bound that **decreases with the number of offline demonstrations**.

## Objective of this project

Test that claim empirically:

- Vary the size of the offline dataset (e.g. **1, 5, 20 and 100 expert trajectories**).
- Run on simulated environments (Gymnasium / MuJoCo, as in the paper).
- Replace the human with a **synthetic teacher** that prefers the trajectory with the higher true return.
- Measure the **regret** and the **number of preference queries** needed to reach a target performance.
- Check whether the empirical trend matches the **shape predicted by the bound**.

## Plan

- [ ] Read the paper and the authors' code; write down the bound and how it depends on the number of demonstrations
- [ ] Environments and expert policies; generate demonstration datasets of 1, 5, 20 and 100 trajectories
- [ ] Synthetic preference teacher (prefers the higher-return trajectory)
- [ ] Offline stage (BC) + online preference fine-tuning (BRIDGE); baselines: BC alone, online preference-based RL alone
- [ ] Metrics: cumulative regret, number of queries to reach a target return; several seeds
- [ ] Fit the empirical curves against the predicted shape of the bound
- [ ] Report and results

## Stack

Python, PyTorch, Gymnasium (MuJoCo).

## Author

**Gaya Lahouazi**, M2 AI2D (Sorbonne Université).
