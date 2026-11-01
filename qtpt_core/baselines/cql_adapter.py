"""
CQL (Conservative Q-Learning) Adapter for QTPT Experiments

CQL is adapted to the QTPT interface for fair comparison.

Reference: https://github.com/aviralkumar2907/CQL
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from model import Transformer


class CQLAdapter:
    """
    Conservative Q-learning wrapper using the same Transformer backbone as QTPT.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Args:
            config: QTPT-compatible configuration.
            device: torch.device.
        """
        self.config = config
        self.device = device
        self.min_q_weight = config.get('min_q_weight', 5.0)
        
        # Use the same Transformer architecture as QTPT, configured as a Q-network.
        q_config = config.copy()
        q_config['pred_q'] = True
        self.q_network = Transformer(q_config, device)
        self.target_q_network = Transformer(q_config, device)
        
        self.q_network.to(device)
        self.target_q_network.to(device)
        
        # Target network.
        self.update_target_network(tau=1.0)
        
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
    
    def update_target_network(self, tau: float = 0.005):
        """Soft-update the target Q-network."""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Run one CQL training step on transition-style offline data.
        
        Args:
            batch: transition batch (s, a, r, s').
            batch['context_actions']: [batch_size, 1, num_actions] - seq_len=1
            batch['context_rewards']: [batch_size, 1, 1] - seq_len=1
            
        Returns:
            Dictionary of scalar loss values.
        """
        # Move tensors to the target device.
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Run one CQL training step on transition-style offline data. (s, a, r, s')
        # seq_len=1
        
        # 1. Q-learning (Bellman: Q(s,a) = r + max Q(s',a'))
        with torch.no_grad():
            # Build the next-state batch.
            next_batch = {
                'action_set': batch.get('next_action_set', batch['action_set']),  # next_state
                'context_actions': torch.zeros_like(batch['context_actions']),  # [batch_size, 1, num_actions] - 
                'context_rewards': torch.zeros_like(batch['context_rewards'])  # [batch_size, 1, 1] - 
            }
            next_q_values = self.target_q_network(next_batch)  # [batch_size, 1, num_actions]
            next_q_max = next_q_values.max(dim=-1, keepdim=True)[0]  # [batch_size, 1, 1]
        
        # TDr + max Q(s',a')
        rewards = batch['context_rewards']  # [batch_size, 1, 1]
        q_target = rewards + self.config.get('gamma', 0.99) * next_q_max  # [batch_size, 1, 1]
        
        q_values = self.q_network(batch)  # [batch_size, 1, num_actions]
        q_selected = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, 1, 1]
        
        # MSE Loss
        # squeeze/unsqueeze
        if q_selected.shape != q_target.shape:
            while q_selected.dim() > q_target.dim():
                q_selected = q_selected.squeeze(-1)
            while q_target.dim() > q_selected.dim():
                q_target = q_target.squeeze(-1)
            # [batch_size][batch_size, 1]
            if q_selected.dim() == 2 and q_selected.size(1) == 1:
                q_selected = q_selected.squeeze(-1)
            if q_target.dim() == 2 and q_target.size(1) == 1:
                q_target = q_target.squeeze(-1)
        
        bellman_loss = nn.MSELoss()(q_selected, q_target)
        
        # 2. CQL (QQ)
        # Q
        q_data = q_selected  # [batch_size, 1, 1]  [batch_size, 1]  [batch_size]
        
        # Q
        q_all = q_values  # [batch_size, 1, num_actions]
        
        # CQLlogsumexp(Q(s,a)) - Q(s,a_data)
        # Q
        logsumexp_q = torch.logsumexp(q_all, dim=-1, keepdim=True)  # [batch_size, 1, 1]
        
        # q_datalogsumexp_q
        if q_data.shape != logsumexp_q.shape:
            while q_data.dim() > logsumexp_q.dim():
                q_data = q_data.squeeze(-1)
            while logsumexp_q.dim() > q_data.dim():
                logsumexp_q = logsumexp_q.squeeze(-1)
            # [batch_size][batch_size, 1]
            if q_data.dim() == 2 and q_data.size(1) == 1:
                q_data = q_data.squeeze(-1)
            if logsumexp_q.dim() == 2 and logsumexp_q.size(1) == 1:
                logsumexp_q = logsumexp_q.squeeze(-1)
        
        cql_loss = (logsumexp_q - q_data).mean()
        
        total_loss = bellman_loss + self.min_q_weight * cql_loss
        
        self.q_optimizer.zero_grad()
        total_loss.backward()
        self.q_optimizer.step()
        
        # Target network.
        self.update_target_network(tau=0.005)
        
        return {
            'bellman_loss': bellman_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def select_action(self, x: Dict, greedy: bool = True) -> torch.Tensor:
        """
        
        
        Args:
            x: 
            greedy: CQL
        """
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(x)
            actions = q_values.argmax(dim=-1)
        self.q_network.train()
        return actions


class CQLTrainer:
    """CQL"""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.model = CQLAdapter(config, device)
    
    def train(self, train_dataloader, num_epochs: int, use_wandb: bool = False):
        """"""
        for epoch in range(num_epochs):
            epoch_losses = {'bellman_loss': [], 'cql_loss': [], 'total_loss': []}
            
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
        self.model.q_network.eval()
        pass

