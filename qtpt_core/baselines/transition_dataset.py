"""
Transition dataset utilities for standard offline RL baselines.
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class TransitionDataset(Dataset):
    """
    Flatten trajectories into independent transitions for IQL/CQL-style baselines. Each transition is converted to the Transformer input format with seq_len=1.
    """
    
    def __init__(self, trajectories, device, action_num):
        """
        Args:
            trajectories: List of trajectories, each is (states, actions, rewards, action_indexs)
            device: torch.device
            action_num: Number of actions
        """
        self.device = device
        self.action_num = action_num
        
        # Flatten trajectories into transitions.
        self.transitions = []
        for states, actions, rewards, action_indexs in trajectories:
            # In linear bandits, the action set is shared across timesteps.
            action_set = np.array(states[0])  # [action_num, dim]
            
            # Build one transition per timestep.
            for t in range(len(actions)):
                if t < len(rewards) and t < len(action_indexs):
                    # Current state is the action set in linear bandits.
                    # actions[t]
                    # rewards[t]
                    # next_state is the same action set.
                    
                    action_one_hot = np.zeros(action_num)
                    if action_indexs[t] < action_num:
                        action_one_hot[action_indexs[t]] = 1.0
                    
                    self.transitions.append({
                        'state': action_set,  # [action_num, dim] - action set
                        'action': action_one_hot,  # [action_num] - one-hot
                        'reward': rewards[t],  # scalar
                        'next_state': action_set,  # [action_num, dim] - next_state = state
                        'action_index': action_indexs[t]  # scalar
                    })
        
        print(f"Expanded {len(trajectories)} trajectories into {len(self.transitions)} transitions")
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        transition = self.transitions[idx]
        
        # Convert fields to tensors.
        state = torch.tensor(transition['state'], dtype=torch.float32).to(self.device)  # [action_num, dim]
        action = torch.tensor(transition['action'], dtype=torch.float32).to(self.device)  # [action_num]
        reward = torch.tensor([transition['reward']], dtype=torch.float32).to(self.device)  # [1]
        next_state = torch.tensor(transition['next_state'], dtype=torch.float32).to(self.device)  # [action_num, dim]
        action_index = transition['action_index']
        
        # Transformertransition
        # seq_len=1
        # linear banditstateaction_set
        action_set_flat = state.reshape(-1)  # [action_num * dim]
        next_action_set_flat = next_state.reshape(-1)  # [action_num * dim]
        
        # 1
        # Transformercontext_actions [seq_len, action_num], context_rewards [seq_len, 1]
        context_actions = action.unsqueeze(0)  # [1, action_num]
        context_rewards = reward.unsqueeze(0)  # [1, 1]
        
        return {
            'action_set': action_set_flat,  # [action_num * dim] - action_set
            'context_actions': context_actions,  # [1, action_num] - seq_len=1
            'context_rewards': context_rewards,  # [1, 1] - seq_len=1
            'action_index': action_index,  # scalar - 
            'next_action_set': next_action_set_flat  # [action_num * dim] - action_set
        }

