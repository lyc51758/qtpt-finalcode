"""
Q-SFT (Q-Learning for Language Models via Supervised Fine-Tuning) Adapter for QTPT Experiments

Q-SFT is adapted to the QTPT interface. It first estimates Q-values and then uses them to construct supervised soft targets.

Reference: Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning (Hong et al., 2024).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict
from model import Transformer


class QSFTAdapter:
    """
    Q-SFT wrapper using the same Transformer backbone as QTPT.
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Args:
            config: QTPT-compatible configuration.
            device: torch.device.
        """
        self.config = config
        self.device = device
        self.gamma = config.get('gamma', 0.99)
        self.temperature = config.get('temperature', 1.0)
        
        # Q-network used to produce soft targets.
        q_config = config.copy()
        q_config['pred_q'] = True
        self.q_network = Transformer(q_config, device)
        
        # Target Q-network for TD targets.
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
    
    def compute_td_target(self, batch: Dict) -> torch.Tensor:
        """
        Compute TD targets used by Q-SFT.
        
        Args:
            batch: sequence batch.
            batch['context_actions']: [batch_size, seq_len, num_actions]
            batch['context_rewards']: [batch_size, seq_len, 1]
            
        Returns:
            TD targets with shape [batch_size, seq_len, 1].
        """
        with torch.no_grad():
            # Build shifted context.
            # Shift context by one step.
            # context_actions[:, 1:, :]  t+1  H-1 
            # context_rewards[:, 1:, :]  t+1  H-1 
            
            seq_len = batch['context_actions'].shape[1]
            
            # batch
            #  t t+1 context
            # context
            next_context_actions = torch.cat([
                batch['context_actions'][:, 1:, :],  # [batch_size, seq_len-1, num_actions]
                torch.zeros((batch['context_actions'].shape[0], 1, batch['context_actions'].shape[2]), 
                           device=self.device)  # [batch_size, 1, num_actions]
            ], dim=1)  # [batch_size, seq_len, num_actions]
            
            next_context_rewards = torch.cat([
                batch['context_rewards'][:, 1:, :],  # [batch_size, seq_len-1, 1]
                torch.zeros((batch['context_rewards'].shape[0], 1, 1), device=self.device)  # [batch_size, 1, 1]
            ], dim=1)  # [batch_size, seq_len, 1]
            
            next_batch = {
                'action_set': batch['action_set'],  # action set for linear bandits
                'context_actions': next_context_actions,
                'context_rewards': next_context_rewards
            }
            
            # Q
            next_q_values = self.target_q_network(next_batch)  # [batch_size, seq_len, num_actions]
            
            # Q
            next_q_max = next_q_values.max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len, 1]
            
            # next_q_max0
            
            # TDr +  * max Q(s', a')
            rewards = batch['context_rewards']  # [batch_size, seq_len, 1]
            td_target = rewards + self.gamma * next_q_max
            
            # TDreward
            # next_q_max0
            
        return td_target
    
    def weighted_cross_entropy_loss(self, q_values: torch.Tensor, 
                                     td_target: torch.Tensor,
                                     actions: torch.Tensor) -> torch.Tensor:
        """
        Q-SFT
        
        Q-SFT
        1. Qsoftmax
        2. Qw(a) = exp(Q*(s,a) / )
        3. L = -sum_a w(a) * log(P(a|s))
        
        Q-SFT
        
        Args:
            q_values: Q [batch_size, seq_len, num_actions]
            td_target: TD targets with shape [batch_size, seq_len, 1]. - Q
            actions:  [batch_size, seq_len, num_actions] (one-hot)
            
        Returns:
            
        """
        batch_size, seq_len, num_actions = q_values.shape
        
        # 1: Q
        # softmax
        log_probs = F.log_softmax(q_values / self.temperature, dim=-1)  # [batch_size, seq_len, num_actions]
        
        # 2: Q
        # Q-SFTw(a) = exp(Q*(s,a) / )
        # td_targetQ
        
        # Q
        #  = exp(td_target / temperature)
        # Q
        
        action_indices = actions.argmax(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        
        # exp(Q / temperature)
        q_weights = torch.exp(q_values / self.temperature)  # [batch_size, seq_len, num_actions]
        
        # td_target
        # scatter_
        td_target_expanded = td_target.expand(-1, -1, num_actions)  # [batch_size, seq_len, num_actions]
        # td_targetQ
        action_mask = actions.bool()  # [batch_size, seq_len, num_actions]
        q_weights = torch.where(
            action_mask,
            torch.exp(td_target_expanded / self.temperature),
            q_weights
        )
        
        weights = q_weights / (q_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 3: 
        # Q-SFTL = -sum_a w(a) * log(P(a|s))
        # w(a)P(a|s)
        
        weighted_log_probs = weights * log_probs  # [batch_size, seq_len, num_actions]
        
        loss = -weighted_log_probs.sum(dim=-1)  # [batch_size, seq_len]
        
        return loss.mean()
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Q-SFT
        
        Args:
            batch: sequence batch.
            batch['context_actions']: [batch_size, seq_len, num_actions]
            batch['context_rewards']: [batch_size, seq_len, 1]
            
        Returns:
            
        """
        # batch
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Q
        q_values = self.q_network(batch)  # [batch_size, seq_len, num_actions]
        
        # TD
        td_target = self.compute_td_target(batch)  # [batch_size, seq_len, 1]
        
        loss = self.weighted_cross_entropy_loss(
            q_values, 
            td_target, 
            batch['context_actions']
        )
        
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        
        # Target network.
        self.update_target_network(tau=0.005)
        
        return {
            'total_loss': loss.item(),
            'q_loss': loss.item()  # Q-SFT
        }
    
    def select_action(self, x: Dict, greedy: bool = True) -> torch.Tensor:
        """
        
        
        Args:
            x: 
            greedy: 
            
        Returns:
            
        """
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(x)  # [batch_size, seq_len, num_actions]
            if greedy:
                actions = q_values.argmax(dim=-1)  # [batch_size, seq_len]
            else:
                # softmax
                probs = F.softmax(q_values / self.temperature, dim=-1)
                actions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))
        self.q_network.train()
        return actions

