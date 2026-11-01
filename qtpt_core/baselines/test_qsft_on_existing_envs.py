"""Train and evaluate Q-SFT under the QTPT protocol."""
import argparse
import torch
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
from baselines.q_sft_adapter import QSFTAdapter
from pretrain_v2 import load_data_new, evaluate_bandits, evaluate_darkroom


def train_and_evaluate_qsft(args, train_dataloader, test_dataloader, device, config):
    """Train and evaluate Q-SFT under the QTPT protocol."""
    
    print("=" * 80)
    print("Training Q-SFT on existing environments")
    print("=" * 80)
    
    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project or "qtpt-baselines",
            name=f"qsft_{args.env}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "method": "Q-SFT",
                "env": args.env,
                "source": args.source,
                **config
            },
            reinit=True
        )
    
    # Q-SFT
    model = QSFTAdapter(config, device)
    
    print("\nTraining Q-SFT...")
    
    best_model_state = None
    best_metric = float('inf') if args.env == 'linear_bandit' else float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    save_dir = os.path.join(args.save_dir, f"qsft_{args.env}_{args.source}")
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_losses = {'q_loss': [], 'total_loss': []}
        
        model.q_network.train()
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            # Debug: print shapes on first batch
            if epoch == 1 and batch_idx == 0:
                print(f"Debug - batch shapes:")
                print(f"  action_set: {batch['action_set'].shape}")
                print(f"  context_actions: {batch['context_actions'].shape}")
                print(f"  context_rewards: {batch['context_rewards'].shape}")
            losses = model.train_step(batch)
            for key, value in losses.items():
                epoch_losses[key].append(value)
        
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        val_losses = []
        model.q_network.eval()
        
        with torch.no_grad():
            for batch in test_dataloader:
                q_values = model.q_network(batch)  # [batch_size, seq_len, num_actions]
                td_target = model.compute_td_target(batch)  # [batch_size, seq_len, 1]
                
                val_loss = model.weighted_cross_entropy_loss(
                    q_values, 
                    td_target, 
                    batch['context_actions']
                )
                val_losses.append(val_loss.item())
        
        validation_loss = np.mean(val_losses) if val_losses else 0.0
        
        # QTPT
        if args.env == 'linear_bandit':
            # Q network
            original_test_config = model.q_network.config.get('test', False)
            original_test_attr = model.q_network.test
            model.q_network.config['test'] = True
            model.q_network.test = True
            eval_config = config.copy()
            eval_config['horizon'] = 200
            average_regrets = evaluate_bandits(
                model.q_network, eval_config, device, 
                num_trajectories=200, T=200, mode='greedy'
            )
            model.q_network.config['test'] = original_test_config
            model.q_network.test = original_test_attr
            avg_regret = np.mean(average_regrets)
            
            print(f"Epoch {epoch} | Training Loss {avg_losses['total_loss']:.4f} | "
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
                    'target_q_network': model.target_q_network.state_dict()
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
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'_ckpt_{epoch}.pth'))
                print(f"Checkpoint saved at epoch {epoch}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_losses['total_loss'],
                    "validation_loss": validation_loss,
                    "q_loss": avg_losses['q_loss'],
                    "average_cumulative_regrets": avg_regret
                }, step=epoch)
            
            early_stop = getattr(args, 'early_stop', False)
            early_stop_patience = getattr(args, 'early_stop_patience', 10)
            if early_stop and epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
                print(f"No improvement for {epochs_without_improvement} epochs (patience: {early_stop_patience})")
                print(f"Best model was at epoch {best_epoch} with regret {best_metric:.4f}")
                break
        
        elif args.env == 'darkroom':
            print(f"Epoch {epoch} | Training Loss {avg_losses['total_loss']:.4f} | "
                  f"Validation Loss {validation_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_losses['total_loss'],
                    "validation_loss": validation_loss,
                    "q_loss": avg_losses['q_loss']
                }, step=epoch)
    
    # wandb
    if args.use_wandb:
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Test Q-SFT on existing QTPT environments')
    
    # QTPT
    parser.add_argument('--env', type=str, default='linear_bandit', choices=['linear_bandit', 'darkroom'])
    parser.add_argument('--source', type=str, default='random', choices=['random', 'linucb'])
    parser.add_argument('--num_envs', type=int, default=10000)
    parser.add_argument('--num_per_task', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data/')
    
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
    
    # Q-SFT
    parser.add_argument('--temperature', type=float, default=1.0, help='Q-SFT temperature parameter')
    
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
    
    traj = load_data_new(args)
    split_idx = int(0.9 * len(traj))
    train_traj = traj[:split_idx]
    test_traj = traj[split_idx:]
    
    # Q-SFTLinearBanditDatasetQTPT
    print("Using full sequences for Q-SFT (same as QTPT)...")
    train_dataset = LinearBanditDataset(train_traj, device, args.action_num, args.horizon)
    test_dataset = LinearBanditDataset(test_traj, device, args.action_num, args.horizon)
    
    sample = train_dataset[0]
    print(f"Debug - Single sample shapes:")
    print(f"  action_set: {sample['action_set'].shape}")
    print(f"  context_actions: {sample['context_actions'].shape}")
    print(f"  context_rewards: {sample['context_rewards'].shape}")
    
    # collatebatching
    def collate_fn(batch):
        """collate"""
        # batchdataset
        action_sets = torch.stack([item['action_set'] for item in batch])  # [batch_size, action_num * dim]
        context_actions = torch.stack([item['context_actions'] for item in batch])  # [batch_size, horizon, action_num]
        context_rewards = torch.stack([item['context_rewards'] for item in batch])  # [batch_size, horizon, 1]
        return {
            'action_set': action_sets,
            'context_actions': context_actions,
            'context_rewards': context_rewards
        }
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # QTPT
    config = {
        'env': args.env,
        'horizon': args.horizon,
        'dim': args.dim,
        'act_num': args.action_num,
        'state_dim': args.state_dim,
        'dropout': args.dropout,
        'action_dim': args.action_num,
        'n_layer': args.n_layer,
        'n_embd': args.n_embd,
        'n_head': args.n_head,
        'shuffle': True,
        'activation': args.activation,
        'pred_q': True,
        'test': False,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gamma': args.gamma,
        'batch_size': args.batch_size,
        'temperature': args.temperature,  # Q-SFT
    }
    
    qsft_model = train_and_evaluate_qsft(args, train_dataloader, test_dataloader, device, config)
    
    final_model = {
        'q_network': qsft_model.q_network.state_dict(),
        'target_q_network': qsft_model.target_q_network.state_dict()
    }
    torch.save(final_model, f"{args.save_dir}/qsft_model_{args.env}_{args.source}_final.pth")
    
    print("\n" + "=" * 80)
    print("Training complete! Check wandb for results.")
    print("=" * 80)


if __name__ == '__main__':
    main()

