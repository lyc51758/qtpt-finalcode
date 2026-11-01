"""Diagnostic stitching toy MDP for QTPT.

This micro-environment is designed to illustrate why bootstrapped TD targets
can help beyond Monte Carlo regression or behavior imitation. The offline data
contains only partial trajectory fragments. No complete optimal trajectory is
included, but the fragments can be connected through Bellman backup.

Expected qualitative behavior:
    QTPT    : can propagate value across fragments via TD bootstrapping.
    QTPT-MC : uses observed segment returns and cannot propagate beyond them.
    SPT     : imitates observed actions without value propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LEFT, RIGHT = 0, 1
S_START, S_MID, S_GL, S_GR, S_BAD = 0, 1, 2, 3, 4
N_STATES, N_ACTIONS = 5, 2


def step(state: int, action: int, env: int) -> tuple[int, float, bool]:
    """One-step deterministic dynamics. Returns next_state, reward, done."""
    if state == S_START:
        next_state = S_MID if action == 0 else S_BAD
        return next_state, 0.0, next_state == S_BAD

    if state == S_MID:
        next_state = S_GL if action == 0 else S_GR
        if env == LEFT:
            reward = 1.0 if next_state == S_GL else 0.0
        else:
            reward = 1.0 if next_state == S_GR else 0.0
        return next_state, reward, True

    return state, 0.0, True


@dataclass
class Trajectory:
    env_id: int
    states: list[int]
    actions: list[int]
    rewards: list[float]


def collect_dataset(
    n_envs: int = 200,
    n_traj_A_per_env: int = 50,
    n_traj_B_per_env: int = 50,
    seed: int = 0,
) -> list[Trajectory]:
    """Collect partial fragments without any full start-to-goal trajectory.

    Type-A fragments start at S_START and contain a single transition. They can
    show how S_START connects to S_MID but never observe the downstream reward.

    Type-B fragments start at S_MID and contain a single transition to one of
    the two terminal goals, revealing which goal is correct for the task.
    """
    rng = np.random.default_rng(seed)
    data: list[Trajectory] = []

    for env_idx in range(n_envs):
        env = env_idx % 2

        for _ in range(n_traj_A_per_env):
            action = int(rng.integers(0, N_ACTIONS))
            _, reward, _ = step(S_START, action, env)
            data.append(Trajectory(env, [S_START], [action], [reward]))

        for _ in range(n_traj_B_per_env):
            action = int(rng.integers(0, N_ACTIONS))
            _, reward, _ = step(S_MID, action, env)
            data.append(Trajectory(env, [S_MID], [action], [reward]))

    rng.shuffle(data)
    return data


class TinyICTransformer(nn.Module):
    """Small context-conditioned Transformer used only for this diagnostic."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        token_dim = N_STATES + N_ACTIONS + 1
        self.input_proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.q_head = nn.Linear(d_model, N_ACTIONS)

    @staticmethod
    def encode_token(state: int, action: int, reward: float) -> torch.Tensor:
        state_one_hot = F.one_hot(torch.tensor(state), N_STATES).float()
        action_one_hot = F.one_hot(torch.tensor(action), N_ACTIONS).float()
        return torch.cat([state_one_hot, action_one_hot, torch.tensor([reward])], dim=-1)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(context)
        hidden = self.transformer(hidden)
        return self.q_head(hidden[:, -1, :])


def make_qtpt_target(
    model: TinyICTransformer,
    reward: float,
    next_state: int,
    done: bool,
    context: torch.Tensor,
) -> float:
    """TD target: reward + max_a Q(next_state, a, context)."""
    if done:
        return reward

    next_token = model.encode_token(next_state, 0, 0.0).unsqueeze(0).unsqueeze(0)
    next_context = torch.cat([context, next_token], dim=1)
    with torch.no_grad():
        q_next = model(next_context)
    return float(reward + q_next.max().item())


def make_mc_target(traj: Trajectory, t: int) -> float:
    """Monte Carlo return over the observed fragment only."""
    return float(sum(traj.rewards[t:]))


def spt_loss(model: TinyICTransformer, action: int, context: torch.Tensor) -> torch.Tensor:
    """Behavior cloning loss on the observed action."""
    logits = model(context)
    return F.cross_entropy(logits, torch.tensor([action]))


def train(
    method: Literal["QTPT", "QTPT-MC", "SPT"],
    data: list[Trajectory],
    n_steps: int = 5_000,
    lr: float = 3e-4,
    seed: int = 0,
) -> TinyICTransformer:
    torch.manual_seed(seed)
    model = TinyICTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    by_env: dict[int, list[Trajectory]] = {}
    for traj in data:
        by_env.setdefault(traj.env_id, []).append(traj)

    for step_idx in range(n_steps):
        traj = data[int(rng.integers(0, len(data)))]
        state = traj.states[0]
        action = traj.actions[0]
        reward = traj.rewards[0]

        same_env = by_env[traj.env_id]
        context_size = min(8, len(same_env))
        indices = rng.choice(len(same_env), size=context_size, replace=False)
        tokens = [
            model.encode_token(
                same_env[i].states[0],
                same_env[i].actions[0],
                same_env[i].rewards[0],
            )
            for i in indices
        ]
        tokens.append(model.encode_token(state, 0, 0.0))
        context = torch.stack(tokens).unsqueeze(0)

        if method == "SPT":
            loss = spt_loss(model, action, context)
        else:
            if method == "QTPT":
                next_state, _, done = step(state, action, traj.env_id)
                target = make_qtpt_target(model, reward, next_state, done, context)
            else:
                target = make_mc_target(traj, 0)

            prediction = model(context)[0, action]
            loss = (prediction - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_idx % 1000 == 0:
            print(f"[{method}] step={step_idx} loss={loss.item():.4f}")

    return model


def evaluate(
    model: TinyICTransformer,
    n_eval_envs: int = 100,
    seed: int = 999,
) -> float:
    """Evaluate by asking the model to execute the two-step stitched policy."""
    rng = np.random.default_rng(seed)
    returns = []

    for env_idx in range(n_eval_envs):
        env = env_idx % 2
        context_tokens = []

        for _ in range(8):
            mid_action = int(rng.integers(0, N_ACTIONS))
            _, mid_reward, _ = step(S_MID, mid_action, env)
            context_tokens.append(model.encode_token(S_MID, mid_action, mid_reward))

        context_tokens.append(model.encode_token(S_START, 0, 0.0))
        context = torch.stack(context_tokens).unsqueeze(0)

        with torch.no_grad():
            first_action = int(model(context).argmax().item())

        next_state, _, done = step(S_START, first_action, env)
        if done:
            returns.append(0.0)
            continue

        second_query = model.encode_token(next_state, 0, 0.0).unsqueeze(0).unsqueeze(0)
        context = torch.cat([context, second_query], dim=1)
        with torch.no_grad():
            second_action = int(model(context).argmax().item())

        _, reward, _ = step(next_state, second_action, env)
        returns.append(reward)

    return float(np.mean(returns))


def main() -> None:
    print("Collecting offline fragments without full start-to-goal trajectories.")
    data = collect_dataset(seed=0)
    print(f"Collected {len(data)} one-step fragments.")

    for method in ("QTPT", "QTPT-MC", "SPT"):
        print(f"\n=== Training {method} ===")
        model = train(method, data, n_steps=5_000, seed=0)
        score = evaluate(model)
        print(f"{method} success rate: {score:.3f}")


if __name__ == "__main__":
    main()
