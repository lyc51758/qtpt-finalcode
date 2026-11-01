"""Train and evaluate IQL/CQL baselines under the QTPT protocol."""
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
from baselines.transition_dataset import TransitionDataset
from baselines.iql_adapter import IQLAdapter
from baselines.cql_adapter import CQLAdapter
from pretrain_v2 import load_data_new, evaluate_bandits, evaluate_darkroom


def train_and_evaluate_iql(args, train_dataloader, test_dataloader, device, config):
    """Train and evaluate IQL under the QTPT protocol."""
    
    print("=" * 80)
    print("Training IQL on existing environments")
    print("=" * 80)
    
    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project or "qtpt-baselines",
            name=f"iql_{args.env}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "method": "IQL",
                "env": args.env,
                "source": args.source,
                **config
            },
            reinit=True
        )
    
    # IQL
    model = IQLAdapter(config, device)
    
    print("\nTraining IQL...")
    
    best_model_state = None
    best_metric = float('inf') if args.env == 'linear_bandit' else float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    save_dir = os.path.join(args.save_dir, f"iql_{args.env}_{args.source}")
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_losses = {'v_loss': [], 'q_loss': [], 'policy_loss': [], 'total_loss': []}
        
        model.policy_network.train()
        model.q_network.train()
        model.v_network.train()
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            losses = model.train_step(batch)
            for key, value in losses.items():
                epoch_losses[key].append(value)
        
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        val_losses = {'v_loss': [], 'q_loss': [], 'policy_loss': []}
        model.policy_network.eval()
        model.q_network.eval()
        model.v_network.eval()
        
        with torch.no_grad():
            for batch in test_dataloader:
                q_values = model.q_network(batch)
                v_values = model.v_network(batch)
                policy_logits = model.policy_network(batch)
                
                # v_values
                if v_values.dim() == 3:
                    v_values = v_values.squeeze(-1)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
                
                # QIQLtransitionseq_len=1
                pred_q = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, 1, 1]
                # next_stateVIQLnext_state
                next_batch = {
                    'action_set': batch.get('next_action_set', batch['action_set']),
                    'context_actions': torch.zeros_like(batch['context_actions']),
                    'context_rewards': torch.zeros_like(batch['context_rewards'])
                }
                next_v = model.v_network(next_batch)  # [batch_size, 1, 1]  [batch_size, 1]
                if next_v.dim() == 3:
                    next_v = next_v.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                elif next_v.dim() == 2:
                    next_v = next_v.squeeze(-1)  # [batch_size, 1] -> [batch_size]
                next_v = next_v.unsqueeze(-1)  # [batch_size, 1]
                q_target = batch['context_rewards'].squeeze(-1) + config.get('gamma', 0.99) * next_v  # [batch_size, 1]
                q_target = q_target.unsqueeze(-1)  # [batch_size, 1, 1]
                q_loss = torch.nn.MSELoss()(pred_q, q_target)
                val_losses['q_loss'].append(q_loss.item())
        
        val_avg_losses = {key: np.mean(values) if values else 0.0 for key, values in val_losses.items()}
        validation_loss = val_avg_losses.get('q_loss', 0.0)
        
        # QTPT
        if args.env == 'linear_bandit':
            # policy network
            # test=Trueevaluate_bandits
            original_test_config = model.policy_network.config.get('test', False)
            original_test_attr = model.policy_network.test
            model.policy_network.config['test'] = True
            model.policy_network.test = True
            # horizon
            eval_config = config.copy()
            eval_config['horizon'] = 200  # horizon
            average_regrets = evaluate_bandits(
                model.policy_network, eval_config, device, 
                num_trajectories=200, T=200, mode='greedy'
            )
            model.policy_network.config['test'] = original_test_config
            model.policy_network.test = original_test_attr
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
                    'policy_network': model.policy_network.state_dict(),
                    'q_network': model.q_network.state_dict(),
                    'v_network': model.v_network.state_dict()
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
                    'policy_network': model.policy_network.state_dict(),
                    'q_network': model.q_network.state_dict(),
                    'v_network': model.v_network.state_dict(),
                    'epoch': epoch
                }
                torch.save(checkpoint, os.path.join(save_dir, f'_ckpt_{epoch}.pth'))
                print(f"Checkpoint saved at epoch {epoch}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_losses['total_loss'],
                    "validation_loss": validation_loss,
                    "v_loss": avg_losses['v_loss'],
                    "q_loss": avg_losses['q_loss'],
                    "policy_loss": avg_losses['policy_loss'],
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
            # eval_trajs
            print(f"Epoch {epoch} | Training Loss {avg_losses['total_loss']:.4f} | "
                  f"Validation Loss {validation_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": avg_losses['total_loss'],
                    "validation_loss": validation_loss,
                    "v_loss": avg_losses['v_loss'],
                    "q_loss": avg_losses['q_loss'],
                    "policy_loss": avg_losses['policy_loss']
                }, step=epoch)
    
    # wandb
    if args.use_wandb:
        wandb.finish()
    
    return model


def train_and_evaluate_cql(args, train_dataloader, test_dataloader, device, config):
    """Train and evaluate CQL under the QTPT protocol."""
    
    print("=" * 80)
    print("Training CQL on existing environments")
    print("=" * 80)
    
    # wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project or "qtpt-baselines",
            name=f"cql_{args.env}_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "method": "CQL",
                "env": args.env,
                "source": args.source,
                **config
            },
            reinit=True
        )
    
    # CQL
    config['min_q_weight'] = args.min_q_weight
    model = CQLAdapter(config, device)
    
    print("\nTraining CQL...")
    
    best_model_state = None
    best_metric = float('inf') if args.env == 'linear_bandit' else float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    save_dir = os.path.join(args.save_dir, f"cql_{args.env}_{args.source}")
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(1, args.num_epochs + 1):
        epoch_losses = {'bellman_loss': [], 'cql_loss': [], 'total_loss': []}
        
        model.q_network.train()
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            losses = model.train_step(batch)
            for key, value in losses.items():
                epoch_losses[key].append(value)
        
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        val_losses = []
        model.q_network.eval()
        
        with torch.no_grad():
            for batch in test_dataloader:
                q_values = model.q_network(batch)  # [batch_size, 1, num_actions]
                pred_q = (q_values * batch['context_actions']).sum(dim=-1, keepdim=True)  # [batch_size, 1, 1]
                
                # CQLtransitionseq_len=1
                # next_stateQCQLnext_state
                next_batch = {
                    'action_set': batch.get('next_action_set', batch['action_set']),
                    'context_actions': torch.zeros_like(batch['context_actions']),
                    'context_rewards': torch.zeros_like(batch['context_rewards'])
                }
                next_q = model.target_q_network(next_batch).max(dim=-1, keepdim=True)[0]  # [batch_size, 1, 1]
                q_target = batch['context_rewards'] + config.get('gamma', 0.99) * next_q  # [batch_size, 1, 1]
                
                # MSE Loss
                if pred_q.shape != q_target.shape:
                    while pred_q.dim() > q_target.dim():
                        pred_q = pred_q.squeeze(-1)
                    while q_target.dim() > pred_q.dim():
                        q_target = q_target.squeeze(-1)
                    # [batch_size][batch_size, 1]
                    if pred_q.dim() == 2 and pred_q.size(1) == 1:
                        pred_q = pred_q.squeeze(-1)
                    if q_target.dim() == 2 and q_target.size(1) == 1:
                        q_target = q_target.squeeze(-1)
                
                val_loss = torch.nn.MSELoss()(pred_q, q_target)
                val_losses.append(val_loss.item())
        
        validation_loss = np.mean(val_losses) if val_losses else 0.0
        
        # QTPT
        if args.env == 'linear_bandit':
            # Q network
            # test=Trueevaluate_bandits
            original_test_config = model.q_network.config.get('test', False)
            original_test_attr = model.q_network.test
            model.q_network.config['test'] = True
            model.q_network.test = True
            # horizon
            eval_config = config.copy()
            eval_config['horizon'] = 200  # horizon
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
                    "bellman_loss": avg_losses['bellman_loss'],
                    "cql_loss": avg_losses['cql_loss'],
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
                    "bellman_loss": avg_losses['bellman_loss'],
                    "cql_loss": avg_losses['cql_loss']
                }, step=epoch)
    
    # wandb
    if args.use_wandb:
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Test IQL and CQL on existing QTPT environments')
    
    # QTPT
    parser.add_argument('--env', type=str, default='linear_bandit', choices=['linear_bandit', 'darkroom'])
    parser.add_argument('--source', type=str, default='random', choices=['random', 'linucb'])
    parser.add_argument('--num_envs', type=int, default=100000)
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
    
    parser.add_argument('--min_q_weight', type=float, default=5.0, help='CQL min Q weight')
    parser.add_argument('--method', type=str, choices=['iql', 'cql', 'both'], default='both',
                       help='Method to run: iql, cql, or both (iql+cql)')
    
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
    
    # IQL/CQLTransitionDatasettransition
    # (s, a, r, s')
    print("Expanding trajectories into individual transitions for standard IQL/CQL...")
    train_dataset = TransitionDataset(train_traj, device, args.action_num)
    test_dataset = TransitionDataset(test_traj, device, args.action_num)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    }
    
    if args.method in ['iql', 'both']:
        iql_model = train_and_evaluate_iql(args, train_dataloader, test_dataloader, device, config)
        final_model = {
            'policy_network': iql_model.policy_network.state_dict(),
            'q_network': iql_model.q_network.state_dict(),
            'v_network': iql_model.v_network.state_dict()
        }
        torch.save(final_model, f"{args.save_dir}/iql_model_{args.env}_{args.source}_final.pth")
    
    if args.method in ['cql', 'both']:
        cql_model = train_and_evaluate_cql(args, train_dataloader, test_dataloader, device, config)
        final_model = {
            'q_network': cql_model.q_network.state_dict(),
            'target_q_network': cql_model.target_q_network.state_dict()
        }
        torch.save(final_model, f"{args.save_dir}/cql_model_{args.env}_{args.source}_final.pth")
    
    print("\n" + "=" * 80)
    print("Training complete! Check wandb for results.")
    print("=" * 80)


if __name__ == '__main__':
    main()

