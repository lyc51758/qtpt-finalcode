"""
QDT (Q-learning Decision Transformer) Adapter for QTPT Experiments

QDT first learns a Q-function and then uses learned values to relabel return-to-go targets for Decision Transformer training.

Reference: Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List
from model import Transformer


class QDTWrapper:
    """
    Wrapper used to evaluate the Decision Transformer component after QDT return-to-go relabeling.
    """
    def __init__(self, qdt_adapter):
        self.qdt_adapter = qdt_adapter
        self.dt_network = qdt_adapter.dt_network
        self.config = qdt_adapter.config
        self.device = qdt_adapter.device
        
        self.test = True
    
    def train(self):
        """Switch both networks to train mode."""
        self.dt_network.train()
        self.qdt_adapter.q_network.train()
    
    def eval(self):
        """Switch both networks to eval mode."""
        self.dt_network.eval()
        self.qdt_adapter.q_network.eval()
    
    def __call__(self, x: Dict) -> torch.Tensor:
        """
        Predict action logits using QDT-style return-to-go conditioning.
        
        Args:
            x: action_set, context_actions, context_rewards
            
        Returns:
            logits [batch_size, num_actions] (test)
        """
        batch_size = x['action_set'].shape[0]
        device = x['action_set'].device
        
        if 'context_actions' in x and x['context_actions'].numel() > 0:
            seq_len = x['context_actions'].shape[1]
            context_actions = x['context_actions']
        else:
            seq_len = 0
            context_actions = torch.empty((batch_size, 0, self.qdt_adapter.dt_network.action_dim), device=device)
        
        # return-to-go
        if seq_len > 0:
            # Q
            hist_batch = {
                'action_set': x['action_set'],
                'context_actions': context_actions,
                'context_rewards': torch.zeros((batch_size, seq_len, 1), device=device)  # 
            }
            
            self.qdt_adapter.q_network.eval()
            with torch.no_grad():
                q_values = self.qdt_adapter.q_network(hist_batch)  # [batch_size, seq_len, num_actions]
                
                # Q
                # q_values: [batch_size, seq_len, num_actions]
                # context_actions: [batch_size, seq_len, num_actions]
                rtg = (q_values * context_actions).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            self.qdt_adapter.q_network.train()
        else:
            # return-to-go
            rtg = torch.zeros((batch_size, 0, 1), device=device)
        
        # Decision Transformer
        dt_input = {
            'action_set': x['action_set'],
            'context_actions': context_actions,
            'context_rewards': rtg  # return-to-go
        }
        
        # Decision Transformer
        action_logits = self.dt_network(dt_input)  # [batch_size, num_actions] (test)
        
        return action_logits


class QDTAdapter:
    """
    QDTQDT
    
    QDT
    1. 1Q-learningreturn-to-go
    2. 2Decision Transformer
    """
    
    def __init__(self, config: Dict, device: torch.device):
        """
        Args:
            config: QTPT
            device: 
        """
        self.config = config
        self.device = device
        self.gamma = config.get('gamma', 0.99)
        
        # 1Q
        q_config = config.copy()
        q_config['pred_q'] = True  # Q
        self.q_network = Transformer(q_config, device)
        self.target_q_network = Transformer(q_config, device)
        self.q_network.to(device)
        self.target_q_network.to(device)
        self.update_target_network(tau=1.0)  # Target network.
        
        # 2Decision Transformer
        dt_config = config.copy()
        dt_config['pred_q'] = False  # Q
        self.dt_network = Transformer(dt_config, device)
        self.dt_network.to(device)
        
        self.q_optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
        
        self.dt_optimizer = optim.Adam(
            self.dt_network.parameters(),
            lr=config.get('lr', 5e-6),
            weight_decay=config.get('weight_decay', 0.001)
        )
        
        self.relabeled_rewards = None
    
    def update_target_network(self, tau: float = 0.005):
        """Soft-update the target network."""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def compute_td_target(self, batch: Dict) -> torch.Tensor:
        """
        TD
        
        Args:
            batch: 
            
        Returns:
            TD [batch_size, seq_len, 1]
        """
        with torch.no_grad():
            seq_len = batch['context_actions'].shape[1]
            
            # batch
            next_context_actions = torch.cat([
                batch['context_actions'][:, 1:, :],
                torch.zeros((batch['context_actions'].shape[0], 1, batch['context_actions'].shape[2]), 
                           device=self.device)
            ], dim=1)
            
            next_context_rewards = torch.cat([
                batch['context_rewards'][:, 1:, :],
                torch.zeros((batch['context_rewards'].shape[0], 1, 1), device=self.device)
            ], dim=1)
            
            next_batch = {
                'action_set': batch['action_set'],
                'context_actions': next_context_actions,
                'context_rewards': next_context_rewards
            }
            
            # Q
            next_q_values = self.target_q_network(next_batch)  # [batch_size, seq_len, num_actions]
            next_q_max = next_q_values.max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len, 1]
            
            # TDr +  * max Q(s', a')
            rewards = batch['context_rewards']  # [batch_size, seq_len, 1]
            td_target = rewards + self.gamma * next_q_max
            
        return td_target
    
    def train_q_network(self, batch: Dict) -> Dict[str, float]:
        """
        1Q
        
        Args:
            batch: 
            
        Returns:
            
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Q
        q_values = self.q_network(batch)  # [batch_size, seq_len, num_actions]
        
        # TD
        td_target = self.compute_td_target(batch)  # [batch_size, seq_len, 1]
        
        # Q
        pred_q = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        
        # Q-learningMSE(pred_q, td_target)
        q_loss = nn.MSELoss()(pred_q, td_target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Target network.
        self.update_target_network(tau=0.005)
        
        return {
            'q_loss': q_loss.item()
        }
    
    def relabel_rewards(self, batch: Dict) -> torch.Tensor:
        """
        Qreturn-to-go
        
        QDT return-to-go target = Q(s, a) - baseline Q.
        
        Args:
            batch: 
            
        Returns:
             [batch_size, seq_len, 1]
        """
        self.q_network.eval()
        with torch.no_grad():
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Q
            q_values = self.q_network(batch)  # [batch_size, seq_len, num_actions]
            
            # Q
            # QDT return-to-go target = Q(s, a).
            relabeled_rewards = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            
        self.q_network.train()
        return relabeled_rewards
    
    def train_dt_network(self, batch: Dict, relabeled_rewards: torch.Tensor) -> Dict[str, float]:
        """
        2Decision Transformer
        
        Args:
            batch: 
            relabeled_rewards:  [batch_size, seq_len, 1]
            
        Returns:
            
        """
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        relabeled_rewards = relabeled_rewards.to(self.device)
        
        dt_batch = {
            'action_set': batch['action_set'],
            'context_actions': batch['context_actions'],
            'context_rewards': relabeled_rewards  # 
        }
        
        # Decision Transformer
        action_logits = self.dt_network(dt_batch)  # [batch_size, seq_len, num_actions]
        
        action_indices = batch['context_actions'].argmax(dim=-1)  # [batch_size, seq_len]
        
        # Decision Transformer
        # logitstargets
        action_logits_flat = action_logits.view(-1, action_logits.size(-1))  # [batch_size * seq_len, num_actions]
        action_indices_flat = action_indices.view(-1)  # [batch_size * seq_len]
        
        dt_loss = nn.CrossEntropyLoss()(action_logits_flat, action_indices_flat)
        
        self.dt_optimizer.zero_grad()
        dt_loss.backward()
        self.dt_optimizer.step()
        
        return {
            'dt_loss': dt_loss.item()
        }
    
    def train_step(self, batch: Dict, train_phase: str = 'both') -> Dict[str, float]:
        """
        
        
        Args:
            batch: 
            train_phase: 
                - 'q_only': Q1
                - 'dt_only': DT2
                - 'both': 
                
        Returns:
            
        """
        losses = {}
        
        if train_phase in ['q_only', 'both']:
            # 1Q
            q_losses = self.train_q_network(batch)
            losses.update(q_losses)
        
        if train_phase in ['dt_only', 'both']:
            # 2DT
            relabeled_rewards = self.relabel_rewards(batch)
            dt_losses = self.train_dt_network(batch, relabeled_rewards)
            losses.update(dt_losses)
        
        losses['total_loss'] = sum(losses.values())
        return losses
    
    def compute_return_to_go(self, x: Dict) -> torch.Tensor:
        """
        return-to-goQ
        
        QDT return-to-go target = Q(s, a).
        -Qreturn-to-go
        
        Args:
            x: action_set, context_actions, context_rewards
            
        Returns:
            return-to-go [batch_size, seq_len, 1]
        """
        self.q_network.eval()
        with torch.no_grad():
            x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            # Q
            q_values = self.q_network(x)  # [batch_size, seq_len, num_actions]  [batch_size, num_actions] (test)
            
            # testq_values[batch_size, num_actions]
            if q_values.dim() == 2:
                # testQ
                rtg = q_values.max(dim=-1, keepdim=True)[0]  # [batch_size, 1]
                rtg = rtg.unsqueeze(1)  # [batch_size, 1, 1] seq_len
            else:
                # q_values[batch_size, seq_len, num_actions]
                # context_actionsQ
                # Q
                if 'context_actions' in x and x['context_actions'].numel() > 0:
                    # Q
                    rtg = (q_values * x['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                else:
                    # Q
                    rtg = q_values.max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len, 1]
            
        self.q_network.train()
        return rtg
    
    def select_action(self, x: Dict, greedy: bool = True) -> torch.Tensor:
        """
        Decision Transformer
        
        Args:
            x: action_set, context_actions, context_rewards
            greedy: 
            
        Returns:
             [batch_size]  [batch_size, seq_len]
        """
        self.dt_network.eval()
        with torch.no_grad():
            x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            
            # return-to-goQ
            rtg = self.compute_return_to_go(x)  # [batch_size, seq_len, 1]
            
            # return-to-go
            dt_input = {
                'action_set': x['action_set'],
                'context_actions': x.get('context_actions', torch.empty((x['action_set'].shape[0], 0, rtg.shape[-1]), device=self.device)),
                'context_rewards': rtg  # return-to-go
            }
            
            # Decision Transformer
            action_logits = self.dt_network(dt_input)  # [batch_size, seq_len, num_actions]  [batch_size, num_actions] (test)
            
            # test [batch_size, num_actions]
            #  [batch_size, seq_len, num_actions]
            if action_logits.dim() == 2:
                # test
                if greedy:
                    actions = action_logits.argmax(dim=-1)  # [batch_size]
                else:
                    probs = F.softmax(action_logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)  # [batch_size]
            else:
                if greedy:
                    actions = action_logits.argmax(dim=-1)  # [batch_size, seq_len]
                else:
                    probs = F.softmax(action_logits, dim=-1)
                    actions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), probs.size(1))
        self.dt_network.train()
        return actions
