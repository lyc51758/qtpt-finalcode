"""Train and evaluate QDT under the QTPT protocol."""
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
import sys
from datetime import datetime

# Add the project root to Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from icrl import Environment
from dataset import LinearBanditDataset
from baselines.qdt_adapter import QDTAdapter
from pretrain_v2 import load_data_new, evaluate_bandits, evaluate_darkroom


def train_and_evaluate_qdt(args, train_dataloader, test_dataloader, device, config):
    """Train and evaluate QDT under the QTPT protocol."""
    
    print("=" * 80)
    print("Training QDT on existing environments")
    print("=" * 80)
    
    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project or "qtpt-baselines",
            name=f"qdt_{args.env}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "method": "QDT",
                "env": args.env,
                "source": args.source,
                **config
            },
            reinit=True
        )
    
    # QDT
    model = QDTAdapter(config, device)
    
    print("\nTraining QDT...")
    
    best_model_state = None
    best_metric = float('inf') if args.env == 'linear_bandit' else float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    save_dir = os.path.join(args.save_dir, f"qdt_{args.env}_{args.source}")
    os.makedirs(save_dir, exist_ok=True)
    
    # QDTQDecision Transformer
    # 1Q
    # QDTQ
    print("\nPhase 1: Training Q-network for reward relabeling...")
    
    # Qepochepoch
    # epoch
    q_network_epochs = getattr(args, 'q_network_epochs', None)
    if q_network_epochs is None:
        q_epochs = args.num_epochs // 2
    else:
        q_epochs = q_network_epochs
    
    # epochQ21epoch
    fixed_q_epoch = getattr(args, 'fixed_q_epoch', 21)
    best_q_model_state = None
    
    print(f"\nTraining Q-Network (will use epoch {fixed_q_epoch} for relabeling)...")
    
    for epoch in range(1, q_epochs + 1):
        epoch_losses = {'q_loss': []}
        model.q_network.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Q-Network Epoch {epoch}")):
            if epoch == 1 and batch_idx == 0:
                print(f"Debug - batch shapes:")
                print(f"  action_set: {batch['action_set'].shape}")
                print(f"  context_actions: {batch['context_actions'].shape}")
                print(f"  context_rewards: {batch['context_rewards'].shape}")
            losses = model.train_step(batch, train_phase='q_only')
            epoch_losses['q_loss'].append(losses['q_loss'])
        
        avg_q_loss = np.mean(epoch_losses['q_loss'])
        
        val_losses = []
        model.q_network.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                q_values = model.q_network(batch)  # [batch_size, seq_len, num_actions]
                td_target = model.compute_td_target(batch)  # [batch_size, seq_len, 1]
                pred_q = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                val_loss = torch.nn.MSELoss()(pred_q, td_target)
                val_losses.append(val_loss.item())
        model.q_network.train()
        
        avg_q_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        print(f"Q-Network Epoch {epoch} | Train Loss {avg_q_loss:.4f} | Val Loss {avg_q_val_loss:.4f}")
        
        # wandb
        if args.use_wandb:
            wandb.log({
                'phase': 'q_network',
                'q_epoch': epoch,
                'q_train_loss': avg_q_loss,
                'q_val_loss': avg_q_val_loss
            })
        
        # epochQ
        if epoch == fixed_q_epoch:
            best_q_model_state = {
                'q_network': model.q_network.state_dict(),
                'target_q_network': model.target_q_network.state_dict()
            }
            print(f" Saved Q-network from epoch {fixed_q_epoch} (train loss: {avg_q_loss:.4f}, val loss: {avg_q_val_loss:.4f})")
            print(f" Stopping Q-network training at epoch {fixed_q_epoch} as requested")
            break  # 
    
    # epochQ
    if best_q_model_state is not None:
        model.q_network.load_state_dict(best_q_model_state['q_network'])
        model.target_q_network.load_state_dict(best_q_model_state['target_q_network'])
        print(f"\n Loaded Q-network from epoch {fixed_q_epoch} for reward relabeling")
    else:
        print(f"\n Warning: Epoch {fixed_q_epoch} not reached. Using final Q-network.")
        best_q_model_state = {
            'q_network': model.q_network.state_dict(),
            'target_q_network': model.target_q_network.state_dict()
        }
    
    print(f"\nQ-Network training completed. Using Q-network from epoch {fixed_q_epoch}")
    
    # 2Decision Transformer
    print("\nPhase 2: Training Decision Transformer with relabeled rewards...")
    # Decision TransformerepochepochQ
    dt_epochs = args.num_epochs - fixed_q_epoch
    
    for epoch in range(1, dt_epochs + 1):
        epoch_losses = {'dt_loss': [], 'total_loss': []}
        model.dt_network.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"DT Epoch {epoch}")):
            losses = model.train_step(batch, train_phase='dt_only')
            for key, value in losses.items():
                epoch_losses[key].append(value)
        
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Decision Transformer
        val_losses = []
        model.dt_network.eval()
        
        with torch.no_grad():
            for batch in test_dataloader:
                relabeled_rewards = model.relabel_rewards(batch)
                
                dt_batch = {
                    'action_set': batch['action_set'],
                    'context_actions': batch['context_actions'],
                    'context_rewards': relabeled_rewards
                }
                
                # Decision Transformer
                action_logits = model.dt_network(dt_batch)  # [batch_size, seq_len, num_actions]
                action_indices = batch['context_actions'].argmax(dim=-1)  # [batch_size, seq_len]
                
                action_logits_flat = action_logits.view(-1, action_logits.size(-1))
                action_indices_flat = action_indices.view(-1)
                val_loss = nn.CrossEntropyLoss()(action_logits_flat, action_indices_flat)
                val_losses.append(val_loss.item())
        
        validation_loss = np.mean(val_losses) if val_losses else 0.0
        
        # Decision Transformer
        if args.env == 'linear_bandit':
            # Decision Transformer
            # return-to-go
            from baselines.qdt_adapter import QDTWrapper
            dt_wrapper = QDTWrapper(model)
            dt_wrapper.dt_network.config['test'] = True
            dt_wrapper.dt_network.test = True
            eval_config = config.copy()
            eval_config['horizon'] = 200
            average_regrets = evaluate_bandits(
                dt_wrapper, eval_config, device, 
                num_trajectories=200, T=200, mode='greedy'
            )
            dt_wrapper.dt_network.config['test'] = False
            dt_wrapper.dt_network.test = False
            avg_regret = np.mean(average_regrets)
            
            print(f"DT Epoch {epoch} | Training Loss {avg_losses['total_loss']:.4f} | "
                  f"Validation Loss {validation_loss:.4f} | "
                  f"Average Cumulative Regrets {avg_regret:.4f}")
            
            # lower regret is better
            improvement = best_metric - avg_regret
            early_stop_min_delta = getattr(args, 'early_stop_min_delta', 0.0)
            if improvement > early_stop_min_delta:
                best_metric = avg_regret
                best_epoch = epoch
                best_model_state = {
                    'q_network': model.q_network.state_dict(),
                    'target_q_network': model.target_q_network.state_dict(),
                    'dt_network': model.dt_network.state_dict()
                }
                epochs_without_improvement = 0
                torch.save(best_model_state, os.path.join(save_dir, f'_best_ckpt_epoch_{best_epoch}.pth'))
                print(f"New best model at epoch {epoch} with regret {best_metric:.4f}")
            else:
                epochs_without_improvement += 1
            
            # checkpoint
            n_epoch_store_model = getattr(args, 'n_epoch_store_model', 10)
            if epoch % n_epoch_store_model == 0:
                checkpoint = {
                    'q_network': model.q_network.state_dict(),
                    'target_q_network': model.target_q_network.state_dict(),
                    'dt_network': model.dt_network.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch + q_epochs,  # epoch
                    'training_loss': avg_losses['total_loss'],
                    'dt_loss': avg_losses.get('dt_loss', 0.0),
                    'validation_loss': validation_loss,
                    'average_cumulative_regrets': avg_regret,
                    'best_regret': best_metric,
                    'best_epoch': best_epoch
                })
            
            if args.early_stop and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement")
                break
        
        elif args.env == 'darkroom':
            # DarkroomDecision Transformer
            from baselines.qdt_adapter import QDTWrapper
            dt_wrapper = QDTWrapper(model)
            dt_wrapper.dt_network.config['test'] = True
            dt_wrapper.dt_network.test = True
            eval_config = config.copy()
            eval_config['horizon'] = 200
            average_rewards = evaluate_darkroom(
                dt_wrapper, eval_config, device,
                num_trajectories=200, T=200, mode='greedy'
            )
            dt_wrapper.dt_network.config['test'] = False
            dt_wrapper.dt_network.test = False
            avg_reward = np.mean(average_rewards)
            
            print(f"Epoch {epoch} | Training Loss {avg_losses['total_loss']:.4f} | "
                  f"Validation Loss {validation_loss:.4f} | "
                  f"Average Rewards {avg_reward:.4f}")
            
            # higher reward is better
            improvement = avg_reward - best_metric
            early_stop_min_delta = getattr(args, 'early_stop_min_delta', 0.0)
            if improvement > early_stop_min_delta:
                best_metric = avg_reward
                best_epoch = epoch
                best_model_state = {
                    'q_network': model.q_network.state_dict(),
                    'target_q_network': model.target_q_network.state_dict()
                }
                epochs_without_improvement = 0
                torch.save(best_model_state, os.path.join(save_dir, f'_best_ckpt_epoch_{best_epoch}.pth'))
                print(f"New best model at epoch {epoch} with reward {best_metric:.4f}")
            else:
                epochs_without_improvement += 1
            
            # checkpoint
            n_epoch_store_model = getattr(args, 'n_epoch_store_model', 10)
            if epoch % n_epoch_store_model == 0:
                checkpoint = {
                    'q_network': model.q_network.state_dict(),
                    'target_q_network': model.target_q_network.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # wandb
            if args.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'training_loss': avg_losses['total_loss'],
                    'q_loss': avg_losses['q_loss'],
                    'validation_loss': validation_loss,
                    'average_rewards': avg_reward,
                    'best_reward': best_metric,
                    'best_epoch': best_epoch
                })
            
            if args.early_stop and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch} after {epochs_without_improvement} epochs without improvement")
                break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    if best_model_state is not None:
        print(f"Best model saved at epoch {best_epoch} with metric {best_metric:.4f}")
        print(f"Model saved to: {os.path.join(save_dir, f'_best_ckpt_epoch_{best_epoch}.pth')}")
    
    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train QDT on existing environments')
    
    parser.add_argument('--env', type=str, default='linear_bandit', choices=['linear_bandit', 'darkroom'])
    parser.add_argument('--source', type=str, default='random', choices=['random', 'medium', 'expert'])
    parser.add_argument('--num_envs', type=int, default=100000)
    parser.add_argument('--num_per_task', type=int, default=1)
    
    # QTPT
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--action_num', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'softmax'])
    
    # QTPT
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--gamma', type=float, default=0.99)
    
    # QDTQepochepoch
    parser.add_argument('--q_network_epochs', type=int, default=None,
                       help='Number of epochs for Q-network training. If None, uses num_epochs // 2')
    parser.add_argument('--fixed_q_epoch', type=int, default=21,
                       help='Fixed epoch to use for Q-network (default: 21). The Q-network from this epoch will be used for reward relabeling.')
    parser.add_argument('--q_early_stop_patience', type=int, default=20,
                       help='Early stopping patience for Q-network training (deprecated, using fixed_q_epoch instead)')
    parser.add_argument('--q_early_stop_min_delta', type=float, default=0.0,
                       help='Minimum delta for Q-network early stopping (deprecated, using fixed_q_epoch instead)')
    
    parser.add_argument('--early_stop', action='store_true',
                       help='Enable early stopping based on validation metric')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Number of epochs to wait before early stopping (default: 10)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                       help='Minimum change to qualify as an improvement (default: 0.0)')
    parser.add_argument('--n_epoch_store_model', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='qtpt-baselines')
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='results/baselines')
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # QTPT
    print("Loading data...")
    args.state_dim = args.dim * args.action_num
    args.H = args.horizon
    args.Q = True
    args.data_path = getattr(args, 'data_path', 'data/')
    
    # load_data_newargs
    traj = load_data_new(args)
    split_idx = int(0.9 * len(traj))
    train_traj = traj[:split_idx]
    test_traj = traj[split_idx:]
    
    print(f"Loaded {len(train_traj)} training trajectories and {len(test_traj)} test trajectories")
    
    # QTPT
    train_dataset = LinearBanditDataset(train_traj, device, args.action_num, args.horizon)
    test_dataset = LinearBanditDataset(test_traj, device, args.action_num, args.horizon)
    
    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: {
            'action_set': torch.stack([item['action_set'] for item in batch]),
            'context_actions': torch.stack([item['context_actions'] for item in batch]),
            'context_rewards': torch.stack([item['context_rewards'] for item in batch])
        }
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'action_set': torch.stack([item['action_set'] for item in batch]),
            'context_actions': torch.stack([item['context_actions'] for item in batch]),
            'context_rewards': torch.stack([item['context_rewards'] for item in batch])
        }
    )
    
    config = {
        'dim': args.dim,
        'action_dim': args.action_num,
        'act_num': args.action_num,
        'state_dim': args.state_dim,
        'horizon': args.horizon,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'activation': args.activation,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gamma': args.gamma,
        'test': False  # False
    }
    
    train_and_evaluate_qdt(args, train_dataloader, test_dataloader, device, config)


if __name__ == '__main__':
    main()

