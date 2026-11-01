"""
Dataset classes for QTPT experiments
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class LinearBanditDataset(Dataset):
    """
    Dataset for Linear Bandit experiments
    
    Args:
        trajectories: List of trajectories, each is (states, actions, rewards, action_indexs)
        device: torch.device
        action_num: Number of actions
        horizon: Time horizon
    """
    
    def __init__(self, trajectories, device, action_num, horizon=200):
        self.trajectories = trajectories
        self.device = device
        self.action_num = action_num
        self.horizon = horizon
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        states, actions, rewards, action_indexs = self.trajectories[idx]
        
        # Get action set from first state (it's the same for all timesteps in linear bandit)
        action_set = torch.tensor(states[0], dtype=torch.float32).to(self.device).reshape(-1)
        
        # Convert trajectory to tensors
        # states: list of action sets (same for all t in linear bandit)
        # actions: list of action vectors
        # rewards: list of scalar rewards
        # action_indexs: list of action indices
        
        # Build context: sequence of (action, reward) pairs
        context_actions = []
        context_rewards = []
        
        for t in range(min(len(actions), self.horizon)):
            # Convert action to one-hot
            action_one_hot = torch.zeros(self.action_num, dtype=torch.float32)
            if action_indexs[t] < self.action_num:
                action_one_hot[action_indexs[t]] = 1.0
            context_actions.append(action_one_hot)
            
            # Reward
            context_rewards.append(torch.tensor([rewards[t]], dtype=torch.float32))
        
        # Pad to horizon if needed
        while len(context_actions) < self.horizon:
            context_actions.append(torch.zeros(self.action_num, dtype=torch.float32))
            context_rewards.append(torch.tensor([0.0], dtype=torch.float32))
        
        # Stack to tensors
        context_actions = torch.stack(context_actions).to(self.device)  # [horizon, action_num]
        context_rewards = torch.stack(context_rewards).to(self.device)  # [horizon, 1]
        
        return {
            'action_set': action_set,  # [action_num * context_dim]
            'context_actions': context_actions,  # [horizon, action_num]
            'context_rewards': context_rewards  # [horizon, 1]
        }


class DarkRoomDataset(Dataset):
    """
    Dataset for Darkroom experiments
    
    Args:
        data_path: Path to pickle file containing trajectories
        config: Configuration dictionary
    """
    
    def __init__(self, data_path, config):
        import pickle
        with open(data_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        self.config = config
        self.device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        # Extract data from trajectory
        context_states = torch.tensor(traj['context_states'], dtype=torch.float32).to(self.device)
        context_actions = torch.tensor(traj['context_actions'], dtype=torch.float32).to(self.device)
        context_rewards = torch.tensor(traj['context_rewards'], dtype=torch.float32).to(self.device)
        
        # Reshape rewards if needed
        if context_rewards.dim() == 1:
            context_rewards = context_rewards.unsqueeze(-1)
        
        return {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_rewards': context_rewards,
            'query_state': torch.tensor(traj.get('query_state', [0, 0]), dtype=torch.float32).to(self.device),
            'optimal_action': torch.tensor(traj.get('optimal_action', [0]), dtype=torch.float32).to(self.device),
            'goal': traj.get('goal', [0, 0])
        }

