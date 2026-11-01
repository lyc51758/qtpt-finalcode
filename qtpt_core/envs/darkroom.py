import numpy as np
from collections import namedtuple


ActionSpace = namedtuple('ActionSpace', ['n'])


class DarkroomEnv:
    """
    DarkRoom grid-world environment.

    The agent navigates a dim x dim grid to find a hidden goal.
    Reward is 1 when the agent is at the goal, 0 otherwise.

    Actions: 0=right(x+1), 1=left(x-1), 2=up(y+1), 3=down(y-1), 4=stay
    State: normalized coordinates [x/(dim-1), y/(dim-1)]
    """

    def __init__(self, dim, goal, horizon):
        self.dim = dim
        self._goal = tuple(goal)
        self._horizon = horizon
        self.action_space = ActionSpace(n=5)
        self._step_count = 0
        # Start at random position
        self.x = np.random.randint(0, dim)
        self.y = np.random.randint(0, dim)

    @property
    def goal(self):
        return self._goal

    @property
    def horizon(self):
        return self._horizon

    def _normalize_state(self, x, y):
        d = self.dim - 1 if self.dim > 1 else 1
        return np.array([x / d, y / d], dtype=np.float64)

    def _denormalize_state(self, state):
        d = self.dim - 1 if self.dim > 1 else 1
        x = int(round(state[0] * d))
        y = int(round(state[1] * d))
        return x, y

    def reset(self):
        self.x = np.random.randint(0, self.dim)
        self.y = np.random.randint(0, self.dim)
        self._step_count = 0
        return self._normalize_state(self.x, self.y)

    def _apply_action_index(self, action_idx, x, y):
        """Apply action index and return new (x, y) clipped to grid."""
        if action_idx == 0:    # right
            x = min(x + 1, self.dim - 1)
        elif action_idx == 1:  # left
            x = max(x - 1, 0)
        elif action_idx == 2:  # up
            y = min(y + 1, self.dim - 1)
        elif action_idx == 3:  # down
            y = max(y - 1, 0)
        # action_idx == 4: stay
        return x, y

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: one-hot numpy array of shape [5], or integer action index.

        Returns:
            (next_state, reward, done, info)
        """
        if isinstance(action, (np.ndarray, list)):
            action = np.asarray(action).flatten()
            if action.shape[0] == self.action_space.n:
                action_idx = int(np.argmax(action))
            else:
                action_idx = int(action[0]) if action.size > 1 else int(action)
        else:
            action_idx = int(action)

        self.x, self.y = self._apply_action_index(action_idx, self.x, self.y)
        self._step_count += 1

        reward = 1.0 if (self.x, self.y) == self._goal else 0.0
        done = self._step_count >= self._horizon
        state = self._normalize_state(self.x, self.y)

        return state, reward, done, {}

    def transit(self, state, action):
        """
        Transition function (does not modify internal state).

        Args:
            state: normalized [x/(dim-1), y/(dim-1)]
            action: integer action index

        Returns:
            (next_state, reward)
        """
        x, y = self._denormalize_state(state)
        action_idx = int(action) if not isinstance(action, (np.ndarray, list)) else int(np.argmax(action))
        nx, ny = self._apply_action_index(action_idx, x, y)
        reward = 1.0 if (nx, ny) == self._goal else 0.0
        return self._normalize_state(nx, ny), reward

    def opt_action(self, state):
        """
        Return the optimal action index to move toward the goal.

        Args:
            state: normalized [x/(dim-1), y/(dim-1)]

        Returns:
            action index (int)
        """
        x, y = self._denormalize_state(state)
        gx, gy = self._goal

        if x < gx:
            return 0  # right
        elif x > gx:
            return 1  # left
        elif y < gy:
            return 2  # up
        elif y > gy:
            return 3  # down
        else:
            return 4  # stay (already at goal)

    def sample_state(self):
        """Return a random normalized state on the grid."""
        x = np.random.randint(0, self.dim)
        y = np.random.randint(0, self.dim)
        return self._normalize_state(x, y)

    def sample_action(self):
        """Return a random one-hot action."""
        action = np.zeros(self.action_space.n)
        action[np.random.randint(0, self.action_space.n)] = 1
        return action


class DarkroomEnvVec:
    """
    Vectorized DarkRoom environment for batched evaluation.
    """

    def __init__(self, dim, goals, horizon):
        self.dim = dim
        self.goals = [tuple(g) for g in goals]
        self.horizon = horizon
        self.n_envs = len(goals)
        self.action_space = ActionSpace(n=5)

        self.xs = np.random.randint(0, dim, size=self.n_envs)
        self.ys = np.random.randint(0, dim, size=self.n_envs)
        self._step_count = 0

    def _normalize_states(self):
        d = self.dim - 1 if self.dim > 1 else 1
        return np.stack([self.xs / d, self.ys / d], axis=-1).astype(np.float64)

    def reset(self):
        self.xs = np.random.randint(0, self.dim, size=self.n_envs)
        self.ys = np.random.randint(0, self.dim, size=self.n_envs)
        self._step_count = 0
        return self._normalize_states()

    def step(self, actions):
        """
        Batched step.

        Args:
            actions: array of shape [n_envs, 5] (one-hot) or [n_envs] (indices)

        Returns:
            (next_states, rewards, dones, infos)
        """
        if len(actions.shape) == 2:
            action_indices = np.argmax(actions, axis=-1)
        else:
            action_indices = actions.astype(int)

        # Apply actions
        # right
        mask = action_indices == 0
        self.xs[mask] = np.minimum(self.xs[mask] + 1, self.dim - 1)
        # left
        mask = action_indices == 1
        self.xs[mask] = np.maximum(self.xs[mask] - 1, 0)
        # up
        mask = action_indices == 2
        self.ys[mask] = np.minimum(self.ys[mask] + 1, self.dim - 1)
        # down
        mask = action_indices == 3
        self.ys[mask] = np.maximum(self.ys[mask] - 1, 0)

        self._step_count += 1
        goals_arr = np.array(self.goals)
        rewards = ((self.xs == goals_arr[:, 0]) & (self.ys == goals_arr[:, 1])).astype(np.float64)
        dones = np.full(self.n_envs, self._step_count >= self.horizon)

        return self._normalize_states(), rewards, dones, {}
