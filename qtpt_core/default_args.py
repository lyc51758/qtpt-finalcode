"""
Default argument parsers for QTPT experiments
"""
import argparse


def add_data_args(parser):
    """Add data-related arguments"""
    parser.add_argument('--env', type=str, default='linear_bandit', 
                       choices=['linear_bandit', 'darkroom'],
                       help='Environment name')
    parser.add_argument('--envs', type=int, default=100000,
                       help='Number of environments (alternative to --num_envs)')
    parser.add_argument('--num_envs', type=int, default=None,
                       help='Number of environments (alternative to --envs)')
    parser.add_argument('--num_per_task', type=int, default=1,
                       help='Number of trajectories per task')
    parser.add_argument('--dim', type=int, default=5,
                       help='Dimension of the environment')
    parser.add_argument('--H', type=int, default=200,
                       help='Horizon (time steps)')
    parser.add_argument('--hists', type=int, default=1,
                       help='Number of histories per environment')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of samples per history')
    parser.add_argument('--mix', type=float, default=-1,
                       help='Mix ratio for expert data (0.0=random, 1.0=expert, -1=no mix)')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to data directory')


def add_model_args(parser):
    """Add model-related arguments"""
    parser.add_argument('--n_layer', type=int, default=8,
                       help='Number of transformer layers')
    parser.add_argument('--n_embd', type=int, default=256,
                       help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'softmax'],
                       help='Activation function')
    parser.add_argument('--action_num', type=int, default=10,
                       help='Number of actions')
    parser.add_argument('--state_dim', type=int, default=50,
                       help='State dimension')


def add_train_args(parser):
    """Add training-related arguments"""
    parser.add_argument('--lr', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                       help='Minimum learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                       help='Number of training epochs')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                       help='Weight decay')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use wandb for logging')
    parser.add_argument('--Q', action='store_true',
                       help='Use Q-learning')
    parser.add_argument('--double', action='store_true',
                       help='Use double DQN')
    parser.add_argument('--soft_max', action='store_true',
                       help='Use softmax for target Q-value')
    parser.add_argument('--source', type=str, default='random',
                       choices=['random', 'linucb', 'thompson'],
                       help='Data source (random, linucb, or thompson)')
    parser.add_argument('--shuffle', action='store_true', default=True,
                       help='Shuffle data')
    parser.add_argument('--early_stop', action='store_true',
                       help='Enable early stopping based on validation metric')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Number of epochs to wait before early stopping (default: 10)')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                       help='Minimum change to qualify as an improvement (default: 0.0)')

