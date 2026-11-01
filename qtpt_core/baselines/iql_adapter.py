"""
IQL (Implicit Q-Learning) Adapter for QTPT Experiments

IQL is adapted to the QTPT interface for fair comparison.

Reference: https://github.com/ikostrikov/implicit_q_learning
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from model import Transformer, ValueNetwork
from dataset import LinearBanditDataset, DarkRoomDataset


class IQLAdapter:
    """
    Implicit Q-learning wrapper using the same Transformer components as QTPT.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Args:
            config: QTPT-compatible configuration.
            device: torch.device.
        """
        self.config = config
        self.device = device
        self.temperature = 3.0
        self.expectile = 0.7
        
        # Q-network.
        q_config = config.copy()
        q_config['pred_q'] = True
        self.q_network = Transformer(q_config, device)
        
        # Value network.
        v_config = config.copy()
        v_config['test'] = False
        self.v_network = ValueNetwork(v_config, device)
        self.v_network.test = False
        
        # Policy network.
        policy_config = config.copy()
        policy_config['pred_q'] = False
        self.policy_network = Transformer(policy_config, device)
        
        self.q_network.to(device)
        self.v_network.to(device)
        self.policy_network.to(device)
        
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
        self.v_optimizer = optim.Adam(
            self.v_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
        
    def expectile_loss(self, pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
        """
        IQL expectile regression loss.
        
        Args:
            pred: predicted values.
            target: target values.
            tau: expectile parameter.
        """
        error = target - pred
        weight = torch.where(error > 0, tau, 1 - tau)
        return weight * (error ** 2)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Run one IQL training step on transition-style offline data.
        
        Args:
            batch: transition batch (s, a, r, s').
            batch['context_actions']: [batch_size, 1, num_actions] - seq_len=1
            batch['context_rewards']: [batch_size, 1, 1] - seq_len=1
            
        Returns:
            Dictionary of scalar loss values.
        """
        # Move tensors to the target device.
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Run one IQL training step on transition-style offline data. (s, a, r, s')
        # seq_len=1
        
        # 1. V (expectile)
        # V(s)  E_{a~}[Q(s,a)]expectile
        with torch.no_grad():
            q_values = self.q_network(batch)  # [batch_size, 1, num_actions]
            # Q
            actions = batch['context_actions'].argmax(dim=-1)  # [batch_size, 1]
            q_selected = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, 1] -> [batch_size]
        
        v_values = self.v_network(batch)  # ValueNetwork [batch_size, 1, 1]  [batch_size, 1]
        if v_values.dim() == 3:
            v_values = v_values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        elif v_values.dim() == 2:
            v_values = v_values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        else:
            raise ValueError(f"Unexpected v_values shape: {v_values.shape}")
        
        v_loss = self.expectile_loss(v_values, q_selected, self.expectile).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # 2. Q (TD: Q(s,a) = r + V(s'))
        # s'V
        with torch.no_grad():
            # In linear bandits, next_state is the same action set.
            # next_action_set
            next_batch = {
                'action_set': batch.get('next_action_set', batch['action_set']),  # next_state
                'context_actions': torch.zeros_like(batch['context_actions']),  # [batch_size, 1, num_actions] - 
                'context_rewards': torch.zeros_like(batch['context_rewards'])  # [batch_size, 1, 1] - 
            }
            next_v_values = self.v_network(next_batch)  # [batch_size, 1, 1]  [batch_size, 1]
            if next_v_values.dim() == 3:
                next_v_values = next_v_values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            elif next_v_values.dim() == 2:
                next_v_values = next_v_values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            next_v_values = next_v_values.unsqueeze(-1)  # [batch_size, 1] - target
        
        # TDr + V(s')
        rewards = batch['context_rewards'].squeeze(-1)  # [batch_size, 1] -> [batch_size]
        next_v = next_v_values.squeeze(-1) if next_v_values.dim() > 1 else next_v_values  # [batch_size]
        q_target = (rewards + self.config.get('gamma', 0.99) * next_v).unsqueeze(-1)  # [batch_size, 1]
        
        q_values = self.q_network(batch)  # [batch_size, 1, num_actions]
        q_selected = (q_values * batch['context_actions']).sum(dim=-1)  # [batch_size, 1, num_actions] -> [batch_size, 1]
        if q_selected.dim() == 2:
            q_selected = q_selected.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        if q_target.dim() == 2:
            q_target = q_target.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        q_loss = nn.MSELoss()(q_selected, q_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 3.  (advantage)
        with torch.no_grad():
            q_values = self.q_network(batch)  # [batch_size, 1, num_actions]
            v_values = self.v_network(batch)  # [batch_size, 1, 1]  [batch_size, 1]
            # v_valuessqueeze[batch_size]
            if v_values.dim() == 3:
                v_values = v_values.squeeze(-1).squeeze(-1)  # [batch_size, 1, 1] -> [batch_size, 1] -> [batch_size]
            elif v_values.dim() == 2:
                v_values = v_values.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            # v_valuesq_values: [batch_size] -> [batch_size, 1, num_actions]
            v_values_expanded = v_values.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, q_values.size(-1))  # [batch_size, 1, num_actions]
            advantages = q_values - v_values_expanded  # [batch_size, 1, num_actions]
        
        # advantage
        # IQL-E[exp(A(s,a)/) * log (a|s)]
        policy_logits = self.policy_network(batch)  # [batch_size, 1, num_actions]
        
        # one-hot
        target_actions_onehot = batch['context_actions']  # [batch_size, 1, num_actions]
        
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)  # [batch_size, 1, num_actions]
        
        selected_log_probs = (policy_log_probs * target_actions_onehot).sum(dim=-1)  # [batch_size, 1, num_actions] -> [batch_size, 1]
        if selected_log_probs.dim() == 2:
            selected_log_probs = selected_log_probs.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        
        # advantage
        selected_advantages = (advantages * target_actions_onehot).sum(dim=-1)  # [batch_size, 1, num_actions] -> [batch_size, 1]
        if selected_advantages.dim() == 2:
            selected_advantages = selected_advantages.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        
        # Advantageexp(A(s,a)/)
        weights = torch.exp(selected_advantages / self.temperature)  # [batch_size]
        
        # -E[exp(A/) * log (a|s)]
        policy_loss = -(weights * selected_log_probs).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': (v_loss + q_loss + policy_loss).item()
        }
    
    def select_action(self, x: Dict, greedy: bool = False) -> torch.Tensor:
        """
        
        
        Args:
            x: 
            greedy: 
        """
        self.policy_network.eval()
        with torch.no_grad():
            policy_logits = self.policy_network(x)
            if greedy:
                actions = policy_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(policy_logits, dim=-1)
                actions = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
        self.policy_network.train()
        return actions


class IQLTrainer:
    """IQL"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.model = IQLAdapter(config, device)
    
    def train(self, train_dataloader, num_epochs: int, use_wandb: bool = False):
        """"""
        for epoch in range(num_epochs):
            epoch_losses = {'v_loss': [], 'q_loss': [], 'policy_loss': [], 'total_loss': []}
            
            for batch in train_dataloader:
                losses = self.model.train_step(batch)
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
            print(f"Epoch {epoch+1}/{num_epochs}: {avg_losses}")
            
            if use_wandb:
                import wandb
                wandb.log(avg_losses, step=epoch)
    
    def evaluate(self, test_dataloader):
        """"""
        self.model.policy_network.eval()
        pass

