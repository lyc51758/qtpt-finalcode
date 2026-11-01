import argparse
from icrl import *
from model import *
import pickle
import wandb
import tqdm
import os
import math
import random
import glob
from datetime import datetime
from default_args import *
from dataset import LinearBanditDataset, DarkRoomDataset
from utils import build_darkroom_data_filename, build_darkroom_model_filename

# Conditionally import darkroom environment (only needed for darkroom experiments)
try:
    from envs.darkroom import DarkroomEnv, DarkroomEnvVec
except ImportError:
    DarkroomEnv = None
    DarkroomEnvVec = None
    print("Warning: envs.darkroom not found. Darkroom experiments will not be available.")

# from sklearn.model_selection import train_test_split

# 10epoch
n_epoch_store_model = 10


def load_data(args):
    assert args.source in ['linucb', 'random'] and args.data_size <= 100000, 'Invalid source or data size'
    if args.data_path == 'data/': # there is no explicit data path, load from the default path
        if args.mix == -1: # don't need to mix data
            if args.source == 'random':
                traj_0 = pickle.load(open('data/RandomChoose_trajectories_part0_0804.pkl', 'rb'))
                traj_1 = pickle.load(open('data/RandomChoose_trajectories_part1_0804.pkl', 'rb'))
                traj_2 = pickle.load(open('data/RandomChoose_trajectories_part2_0804.pkl', 'rb'))
                traj_3 = pickle.load(open('data/RandomChoose_trajectories_part3_0804.pkl', 'rb'))
                traj_4 = pickle.load(open('data/RandomChoose_trajectories_part4_0804.pkl', 'rb'))
            else:
                print('use default linucb trajctory')
                traj_0 = pickle.load(open('data/linucb_trajectories_part0_0731.pkl', 'rb'))
                traj_1 = pickle.load(open('data/linucb_trajectories_part1_0731.pkl', 'rb'))
                traj_2 = pickle.load(open('data/linucb_trajectories_part2_0731.pkl', 'rb'))
                traj_3 = pickle.load(open('data/linucb_trajectories_part3_0731.pkl', 'rb'))
                traj_4 = pickle.load(open('data/linucb_trajectories_part4_0731.pkl', 'rb'))
        else: # mix data with mix ratio of linucb and random
            print(f"Mixing data with ratio {args.mix} for linucb and {1 - args.mix} for random")
            linucb_files = [
            'data/linucb_trajectories_part0_0731.pkl',
            'data/linucb_trajectories_part1_0731.pkl',
            'data/linucb_trajectories_part2_0731.pkl',
            'data/linucb_trajectories_part3_0731.pkl',
            'data/linucb_trajectories_part4_0731.pkl'
            ]
            random_files = [
                'data/RandomChoose_trajectories_part0_0804.pkl',
                'data/RandomChoose_trajectories_part1_0804.pkl',
                'data/RandomChoose_trajectories_part2_0804.pkl',
                'data/RandomChoose_trajectories_part3_0804.pkl',
                'data/RandomChoose_trajectories_part4_0804.pkl'
            ]

            traj = []
            for linucb_file, random_file in zip(linucb_files, random_files):
                try:
                    linucb_part = pickle.load(open(linucb_file, 'rb'))
                    random_part = pickle.load(open(random_file, 'rb'))
                    
                    combined_part = random.choices(linucb_part, k=int(args.mix * len(linucb_part))) + \
                                    random.choices(random_part, k=int((1 - args.mix) * len(random_part)))
                    traj.extend(combined_part)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"File not found: {e.filename}")
                except pickle.UnpicklingError as e:
                    raise ValueError(f"Error unpickling file: {e}")

            return traj[:args.data_size + 1]
    else:
        print('using custom data path', args.data_path)
        traj_0 = pickle.load(open(args.data_path + '_trajectories_0.pkl', 'rb'))
        traj_1 = pickle.load(open(args.data_path + '_trajectories_1.pkl', 'rb'))
        traj_2 = pickle.load(open(args.data_path + '_trajectories_2.pkl', 'rb'))
        traj_3 = pickle.load(open(args.data_path + '_trajectories_3.pkl', 'rb'))
        traj_4 = pickle.load(open(args.data_path + '_trajectories_4.pkl', 'rb'))
    # integrate all trajectories
    traj = []
    traj.extend(traj_0)
    traj.extend(traj_1)
    traj.extend(traj_2)
    traj.extend(traj_3)
    traj.extend(traj_4)
    return traj[:args.data_size + 1]

def load_data_new(args):
    # Handle both dict and object args
    if isinstance(args, dict):
        source = args.get('source', 'random')
        num_envs = args.get('num_envs') if args.get('num_envs') is not None else args.get('envs', 100000)
        num_per_task = args.get('num_per_task', 1)
        data_path = args.get('data_path', 'data/')
    else:
        source = args.source
        num_envs = args.num_envs if args.num_envs is not None else (args.envs if hasattr(args, 'envs') else 100000)
        num_per_task = args.num_per_task
        data_path = args.data_path
    
    # Supported offline data sources for the QTPT bandit experiments.
    assert source in ['linucb', 'random', 'thompson']

    traj = []
    # Calculate number of files based on chunk size (50000 per file)
    num_files = math.ceil(num_envs * num_per_task / 50000)
    
    # Try to load files in multiple formats (handle different naming conventions)
    for i in tqdm.tqdm(range(num_files), desc='Loading data'):
        # Format 1: {source}_EnvNum_{num_envs}_NumperTask_{num_per_task}_trajectories_{i}.pkl (current generate_data.py format, MDP)
        file1 = os.path.join(data_path, f'{source}_EnvNum_{num_envs}_NumperTask_{num_per_task}_trajectories_{i}.pkl')
        
        # Format 2: RandomChoose_EnvNum_... (capitalized for random)
        file2 = None
        if source == "random":
            file2 = os.path.join(data_path, f'RandomChoose_EnvNum_{num_envs}_NumperTask_{num_per_task}_trajectories_{i}.pkl')
        
        # Format 3: {source}_Env_num_{num_envs}_num_per_task_{num_per_task}_trajectories_{i}.pkl (lowercase)
        file3 = os.path.join(data_path, f'{source}_Env_num_{num_envs}_num_per_task_{num_per_task}_trajectories_{i}.pkl')
        
        # Format 4: Old format files (for backward compatibility) - search for any date pattern
        file4 = None
        if num_envs == 100000 and num_per_task == 1:
            # Try to find old format files with any date pattern
            if source == "random":
                pattern = os.path.join(data_path, f'RandomChoose_trajectories_part{i}_*.pkl')
            else:
                pattern = os.path.join(data_path, f'linucb_trajectories_part{i}_*.pkl')
            
            matching_files = glob.glob(pattern)
            if matching_files:
                # Use the first matching file (prefer newer ones if sorted)
                file4 = sorted(matching_files)[-1]  # Get the most recent one
        
        # Try loading in order: Format 1 -> Format 2 -> Format 3 -> Format 4
        file_loaded = False
        for file_path in [file1, file2, file3, file4]:
            if file_path and os.path.exists(file_path):
                print(f"Loading {file_path}")
                traj.extend(pickle.load(open(file_path, 'rb')))
                file_loaded = True
                break
        
        if not file_loaded:
            # List all attempted file paths for debugging
            attempted_files = [f for f in [file1, file2, file3, file4] if f]
            raise FileNotFoundError(
                f"Could not find trajectory data file for {source} with num_envs={num_envs}, num_per_task={num_per_task}, file_index={i}.\n"
                f"Attempted files:\n" + "\n".join(f"  - {f}" for f in attempted_files) + "\n"
                f"Please check if data files exist in {data_path} or generate them using:\n"
                f"  python generate_data.py --env linear_bandit --algos {source} --num_envs {num_envs} --num_per_task {num_per_task}"
            )
    return traj



def trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, scheduler, loss_fn, device, save_dir,use_wandb=True, num_epochs=10, eval_trajs=None):
    # DataLoader can handle batching and shuffling
    model.train()  # Set the model to training mode


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # best_model_state = model.state_dict()
    # best_loss = float('inf')
    n_epoch_store_model = 1
    # num_epochs = num_steps // len(train_dataloader)
    for epoch in range(1, 1+num_epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_dataloader):
            # Unpack the data
            # tokens, action_labels = batch
            # print(batch['context_actions'].size())
            pred_actions = model(batch) # dimension: (batch_size, seq_len, action_dim)
            # print(pred_actions.shape)
            true_actions = batch['context_actions'] # dimension: (batch_size, seq_len, action_dim)
            pred_actions_flat =  pred_actions.view(-1, action_dim)
            # optimal_actions_flat = optimal_actions.view(-1)
            # change optimal_actions to class label
            true_actions_flat = true_actions.argmax(dim=-1).view(-1)

            # print(pred_actions_flat.shape, optimal_actions_flat.shape)
            loss = loss_fn(pred_actions_flat, true_actions_flat)
            # loss.backward()
            # print(action_labels)
            # Reset the gradients in the optimizer
            optimizer.zero_grad()

            # # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        # Compute the test loss
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader):
                pred_actions = model(batch)
                # optimal_actions = batch['optimal_actions']
                optimal_actions = batch['context_actions']
                pred_actions_flat =  pred_actions.view(-1, action_dim)
                # optimal_actions_flat = optimal_actions.view(-1)
                optimal_actions_flat = optimal_actions.argmax(dim=-1).view(-1)
                loss = loss_fn(pred_actions_flat, optimal_actions_flat)    
                test_loss += loss.item()
        model.train()
        test_loss /= len(test_dataloader)

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     best_model_state = model.state_dict()
        # online evaluation
        with torch.no_grad():
            if config['env'] == 'linear_bandit':
                cumulative_regrets = evaluate_bandits(model, config, device, num_trajectories=200, T=200, mode = 'sample')
                print(f"Epoch {epoch} | Train Loss {epoch_loss} | validate Loss {test_loss} | Average Cumulative Regrets {np.mean(cumulative_regrets)}")
                if use_wandb:
                    # wandb logging
                    wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_cumulative_regrets": np.mean(cumulative_regrets)})
                    # create a log file
                with open(f'{save_dir}/log.csv', 'a') as f:
                    if epoch == 1:
                        f.write("epoch,train_loss,validation_loss,average_regrets\n")
                    f.write(f"{epoch},{epoch_loss},{test_loss},{np.mean(cumulative_regrets)}\n")

            elif config['env'] == 'darkroom':
                average_rewards = evaluate_darkroom(model, config, device, mode = 'greedy', eval_trajs=eval_trajs)
                print(f"Epoch {epoch} | Train Loss {epoch_loss} | validate Loss {test_loss} | Average Rewards {np.mean(average_rewards)}")
                if use_wandb:
                    wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_rewards": np.mean(average_rewards)})
                
                with open(f'{save_dir}/log.csv', 'a') as f:
                    if epoch == 1:
                        f.write("epoch,train_loss,validation_loss,average_rewards\n")
                    f.write(f"{epoch},{epoch_loss},{test_loss},{np.mean(average_rewards)}\n")
        
        # if use_wandb:
        #     # wandb logging
        #     wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss})
        #     print(f"Epoch {epoch} | Train Loss {epoch_loss} | validate Loss {test_loss}")

        # with open(f'{save_dir}/log.csv', 'a') as f:
        #     if epoch == 1:
        #         f.write("epoch,train_loss,validation_loss\n")
        #     f.write(f"{epoch},{epoch_loss},{test_loss}\n")
        if epoch % n_epoch_store_model == 0:
            torch.save(model.state_dict(), f'{save_dir}/_ckpt_{epoch}.pth')

    # best_model = Transformer(config, device)
    # best_model.load_state_dict(best_model_state)
    return 


# def time_weighted_mse_loss(pred_Qvalues, TD_target, type = 'exponential'):
#     # pred_Qvalues: (batch_size, seq_len, A)
#     # TD_target: (batch_size, seq_len, A)
#     # time_weights: (batch_size, seq_len)
#     """
#     We design the time_weights to be a function of time, so that the model pays more attention to the later time steps.
#     type: 
#         'exponential': e^(1e kt/T)
#         'linear': t/T
#     """
#     seq_len = pred_Qvalues.shape[1]
#     if type == 'exponential':
#         k = 0.1
#         time_weights = torch.exp(1 - torch.exp(-k * torch.arange(seq_len, device=pred_Qvalues.device) / seq_len))
#     elif type == 'linear':
#         time_weights = torch.arange(seq_len, device=pred_Qvalues.device) / seq_len
#     loss = (pred_Qvalues - TD_target) ** 2 * time_weights
#     return loss.mean()

class ExponentialTimeWeightedLoss(nn.Module):
    def __init__(self, k=10):
        super(ExponentialTimeWeightedLoss, self).__init__()
        self.k = k

    def forward(self, pred_Qvalues, TD_target):
        k = self.k
        # np.exp(k*x/T-k)
        # time_weights = torch.exp(1 - torch.exp(-k * torch.arange(pred_Qvalues.shape[1], device=pred_Qvalues.device) / pred_Qvalues.shape[1]))
        time_weights = torch.exp(k * torch.arange(pred_Qvalues.shape[1], device=pred_Qvalues.device) / pred_Qvalues.shape[1] - k)
        loss = (pred_Qvalues - TD_target) ** 2 * time_weights
        return loss.mean()


def Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir, softmax=False, use_wandb=True, num_epochs=16, gamma = 0.99, double=False, eval_trajs=None, early_stop=False, early_stop_patience=10, early_stop_min_delta=0.0):
    # record the loss before training
    train_loss = 0.0
    test_loss = 0.0
    model.eval()

    global n_epoch_store_model
    
    try:
        _ = n_epoch_store_model
    except NameError:
        n_epoch_store_model = 10  # 10epoch

    # constryct the save directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # batchdevice
    def move_batch_to_device(batch, device):
        """batchtensordevice"""
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            # batchdevice
            batch = move_batch_to_device(batch, device)
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            if softmax:
                next_Qvalues = F.softmax(Qvalues, dim=-1) * Qvalues # dimension: (batch_size, seq_len, A)
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
            else:
                next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
            next_Qvalues = torch.cat([next_Qvalues[:, 1:,:], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            next_Qvalues = next_Qvalues.detach()  # Detach next_Qvalues from computation graph
            
            pred_Qvalues = Qvalues * batch['context_actions']
            pred_Qvalues = pred_Qvalues.sum(dim=-1, keepdim=True)
            
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            loss = loss_fn(pred_Qvalues, TD_target)
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        for batch in tqdm.tqdm(test_dataloader):
            # batchdevice
            batch = move_batch_to_device(batch, device)
            Qvalues = model(batch)
            if softmax:
                next_Qvalues = F.softmax(Qvalues, dim=-1) * Qvalues
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
            else:
                next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]

            next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            next_Qvalues = next_Qvalues.detach()  # Detach next_Qvalues from computation graph
            
            pred_Qvalues = Qvalues * batch['context_actions']
            pred_Qvalues = pred_Qvalues.sum(dim=-1, keepdim=True)
            
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            loss = loss_fn(pred_Qvalues, TD_target)
            test_loss += loss.item()
        test_loss /= len(test_dataloader)


        if config['env'] == 'linear_bandit':
            cumulative_regrets = evaluate_bandits(model, config, device, num_trajectories=200, T=200, mode = 'greedy')

            print(f"Training Loss {train_loss} | validation Loss {test_loss} | Average Cumulative Regrets {np.mean(cumulative_regrets)}")
            if use_wandb:
                wandb.log({"training_loss": train_loss, "validation_loss": test_loss, "average_cumulative_regrets": np.mean(cumulative_regrets)})
                    # create a log file
            with open(f'{save_dir}/log.csv', 'w') as f:
                f.write("epoch,train_loss,validation_loss,average_regrets\n")
                f.write(f"0,{train_loss},{test_loss},{np.mean(cumulative_regrets)}\n")


        elif config['env'] == 'darkroom':
            average_rewards = evaluate_darkroom(model, config, device, eval_trajs, mode = 'greedy')

            print(f"Training Loss {train_loss} | validation Loss {test_loss} | Average Rewards {np.mean(average_rewards)}")
            if use_wandb:
                wandb.log({"training_loss": train_loss, "validation_loss": test_loss, "average_rewards": np.mean(average_rewards)})
            
            with open(f'{save_dir}/log.csv', 'w') as f:
                f.write("epoch,train_loss,validation_loss,average_rewards\n")
                f.write(f"0,{train_loss},{test_loss},{np.mean(average_rewards)}\n")
        
    # DataLoader can handle batching and shuffling
    model.train()  # Set the model to training mode

    if double:
        target_model = Transformer(config, device)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        target_model.to(device)

    # Track best model based on validation performance
    best_model_state = None
    # Initialize based on environment: regrets (lower is better) or rewards (higher is better)
    if config['env'] == 'linear_bandit':
        best_metric = float('inf')  # For regrets (lower is better)
    elif config['env'] == 'darkroom':
        best_metric = float('-inf')  # For rewards (higher is better)
    else:  # Other environments use negative validation loss (lower loss is better)
        best_metric = float('-inf')  # Will be set to -test_loss, so higher (less negative) is better
    best_epoch = 0
    
    # Early stopping tracking
    epochs_without_improvement = 0

    # num_epochs = num_steps // len(train_dataloader)
    step = 0
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_dataloader):
            # batchdevice
            batch = move_batch_to_device(batch, device)
            Qvalues = model(batch) # dimension: (batch_size, seq_len, A)
            # print(Qvalues)
            pred_Qvalues = Qvalues * batch['context_actions']
            pred_Qvalues = pred_Qvalues.sum(dim=-1, keepdim=True)
            
            with torch.no_grad():
                if softmax:
                    next_actions = F.softmax(Qvalues, dim=-1) 
                else:
                    next_actions = Qvalues.max(dim=-1, keepdim=True)[1] # dimension: (batch_size, seq_len, A)
                    next_actions = F.one_hot(next_actions, num_classes=config['act_num']).float().squeeze(dim=-2) # dimension: (batch_size, seq_len, A) 
                if double:
                    # use target model to get the next Q values
                    # next_Qvalues = target_model(batch).gather(2, next_actions.unsqueeze(-1).long())
                    # next_Qvalues = target_model(batch).gather(2, next_actions.long())
                    next_Qvalues = target_model(batch) * next_actions
                else:
                    # next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                    next_Qvalues = Qvalues.detach() * next_actions
                
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
                next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
            # Note: batch['context_rewards'] should be in the shape (batch_size, seq_len, 1) to match pred_Qvalues
            TD_target = batch['context_rewards'] + gamma * next_Qvalues
            # Calculate the loss
            # loss = loss_fn(pred_Qvalues, TD_target, reduction='mean')
            loss = loss_fn(pred_Qvalues, TD_target)
            # Reset the gradients in the optimizer
            optimizer.zero_grad()
            # # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            step += 1
            # if double:
            #     for param, target_param in zip(model.parameters(), target_model.parameters()):
            #         target_param.data = target_param.data * (1 - 0.01) + param.data * 0.01
            if double:
                if step % 100 == 0:
                    target_model.load_state_dict(model.state_dict())
            
        epoch_loss /= len(train_dataloader)
        # Compute the test loss
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader):
                # batchdevice
                batch = move_batch_to_device(batch, device)
                # pred_Qvalues = model(batch).gather(2, batch['context_actions'].unsqueeze(-1).long())
                Qvalues = model(batch)
                pred_Qvalues = Qvalues * batch['context_actions']
                pred_Qvalues =  pred_Qvalues.sum(dim=-1, keepdim=True)
                next_actions = model(batch).max(dim=-1, keepdim=True)[1]
                # if double:
    
                #     next_Qvalues = target_model(batch).gather(2, next_actions.long())
                # else:
                #     next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                if softmax:
                    next_actions = F.softmax(Qvalues, dim=-1) 
                else:
                    next_actions = model(batch).max(dim=-1, keepdim=True)[1] # dimension: (batch_size, seq_len, A)
                    next_actions = F.one_hot(next_actions, num_classes=config['act_num']).float().squeeze(dim=-2) # dimension: (batch_size, seq_len, A) 
                
                if double:
                    # use target model to get the next Q values
                    # next_Qvalues = target_model(batch).gather(2, next_actions.unsqueeze(-1).long())
                    # next_Qvalues = target_model(batch).gather(2, next_actions.long())
                    next_Qvalues = target_model(batch) * next_actions
                else:
                    # next_Qvalues = Qvalues.max(dim=-1, keepdim=True)[0]
                    next_Qvalues = Qvalues.detach() * next_actions
                
                next_Qvalues = next_Qvalues.sum(dim=-1, keepdim=True)
                next_Qvalues = torch.cat([next_Qvalues[:, 1:, :], torch.zeros((next_Qvalues.shape[0], 1, 1), device=next_Qvalues.device)], dim=1)
                # shifted_pred_Qvalues = torch.cat([pred_Qvalues[:, 1:, :], torch.zeros((pred_Qvalues.shape[0], 1, 1), device=pred_Qvalues.device)], dim=1)
                
                TD_target = batch['context_rewards'] + gamma * next_Qvalues

                loss = loss_fn(pred_Qvalues, TD_target)
                test_loss += loss.item()

        test_loss /= len(test_dataloader)

        # Save checkpoint at regular intervals (save every epoch by default)
        if n_epoch_store_model > 0 and epoch % n_epoch_store_model == 0:
            torch.save(model.state_dict(), save_dir + f'/_ckpt_{epoch}.pth')
            print(f"Checkpoint saved at epoch {epoch}")
        
        # Always save best model checkpoint when it improves
        save_best_checkpoint = False

        with torch.no_grad():
            if config['env'] == 'linear_bandit':
                average_regrets = evaluate_bandits(model, config, device, num_trajectories=200, T=200, mode = 'greedy')
                current_metric = np.mean(average_regrets)

                # wandb logging
                if use_wandb:
                    wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_cumulative_regrets": current_metric, "epoch": epoch})
                print(f"Epoch {epoch} | Training Loss {epoch_loss} | validation Loss {test_loss} | Average Cumulative Regrets {current_metric}")

                with open(f'{save_dir}/log.csv', 'a') as f:
                    f.write(f"{epoch},{epoch_loss},{test_loss},{current_metric}\n")
                
                # Track best model (lower regret is better)
                improvement = best_metric - current_metric  # Positive means improvement
                if improvement > early_stop_min_delta:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                    save_best_checkpoint = True
                    print(f"New best model at epoch {epoch} with regret {best_metric:.4f}")
                else:
                    epochs_without_improvement += 1

            elif config['env'] == 'darkroom':
                average_rewards = evaluate_darkroom(model, config, device, eval_trajs, mode = 'greedy')
                current_metric = np.mean(average_rewards)

                # wandb logging
                if use_wandb:
                    wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_rewards": current_metric, "epoch": epoch})
                print(f"Epoch {epoch} | Training Loss {epoch_loss} | validation Loss {test_loss} | Average Rewards {current_metric}")

                with open(f'{save_dir}/log.csv', 'a') as f:
                    f.write(f"{epoch},{epoch_loss},{test_loss},{current_metric}\n")
                
                # Track best model (higher reward is better)
                improvement = current_metric - best_metric  # Positive means improvement
                if improvement > early_stop_min_delta:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                    save_best_checkpoint = True
                    print(f"New best model at epoch {epoch} with reward {best_metric:.4f}")
                else:
                    epochs_without_improvement += 1
            
            else:
                # Other environments only track validation loss.
                if use_wandb:
                    wandb.log({
                        "training_loss": epoch_loss, 
                        "validation_loss": test_loss, 
                        "epoch": epoch
                    })
                print(f"Epoch {epoch} | Training Loss {epoch_loss:.4f} | Validation Loss {test_loss:.4f}")
                
                # Track best model based on validation loss
                improvement = best_metric - (-test_loss)  # best_metric is -test_loss, so lower test_loss is better
                if improvement > early_stop_min_delta:
                    best_metric = -test_loss
                    best_epoch = epoch
                    best_model_state = model.state_dict()
                    epochs_without_improvement = 0
                    save_best_checkpoint = True
                    print(f"New best model at epoch {epoch} with validation loss {test_loss:.4f}")
                else:
                    epochs_without_improvement += 1
        
        # Save best model checkpoint immediately when it improves (overwrite previous best)
        if save_best_checkpoint:
            # config
            checkpoint_dict = {
                'model_state_dict': best_model_state,
                'config': config,
                'epoch': epoch,
                'best_metric': best_metric
            }
            torch.save(checkpoint_dict, save_dir + '/best_model.pth')
        
        # Early stopping check
        if early_stop and epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            print(f"No improvement for {epochs_without_improvement} epochs (patience: {early_stop_patience})")
            print(f"Best model was at epoch {best_epoch} with metric {best_metric:.4f}")
            break

        model.train() 

    # Save final model
    print(f"\nSaving final model from epoch {num_epochs}...")
    torch.save(model.state_dict(), save_dir + f'/final_model_epoch_{num_epochs}.pth')
    
    # Save best model if we found one (overwrite previous best)
    if best_model_state is not None:
        print(f"Saving best model from epoch {best_epoch} with metric {best_metric:.4f}...")
        # config
        checkpoint_dict = {
            'model_state_dict': best_model_state,
            'config': config,
            'epoch': best_epoch,
            'best_metric': best_metric
        }
        torch.save(checkpoint_dict, save_dir + '/best_model.pth')
        # Also load best model state into the returned model
        model.load_state_dict(best_model_state)
    
    return model

def linear_bandit_pretrain(args):
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device {device}")
    # parse arguments
    batch_size = args.batch_size
    action_num = args.act_num

    # load data
    # traj = load_data(args)
    traj = load_data_new(args)
    # pretrain
    # train_traj, test_traj = train_test_split(traj, test_size=0.2)
    # define the train and test size
    train_size = int(0.8 * len(traj))
    test_size = len(traj) - train_size
    train_traj, test_traj = torch.utils.data.random_split(traj, [train_size, test_size])
    # create dataset
    print(args.H)
    train_dataset = LinearBanditDataset(train_traj, device, action_num, horizon=args.H)
    test_dataset = LinearBanditDataset(test_traj, device, action_num, horizon=args.H)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # set up model
    # model = TransformerModel(embed_dim=70, num_heads=5)
    
    config = {
            'env': 'linear_bandit',
            'horizon': args.H,
            'dim': args.dim,
            'act_num': args.act_num,
            'state_dim': args.state_dim,
            'dropout': args.dropout,
            'action_dim': args.act_num,
            'n_layer': args.n_layer,
            'n_embd': args.n_embd,
            'n_head': args.n_head,
            'shuffle': True,
            'activation': args.activation,
            'pred_q': args.Q,
            'test': False
        }
    print(config)
    action_dim = config['act_num']
    model = Transformer(config, device)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_steps = len(train_dataloader) * args.num_epochs
    scheduler = CosineAnnealingWarmup(optimizer, learning_rate=args.lr, warmup_steps=2000, lr_decay_steps=num_steps, min_lr=args.min_lr)

    #save_dir models/Qtransformer_linucb_embedding_256_gamma_1_epochs_50_lr_5e-6_batchsize_16
    if hasattr(args, 'save_dir'):
        save_dir = args.save_dir
    else:
        if args.Q:
            save_dir = 'models/V2_Q_' + args.source + '_EnvN' + str(args.num_envs) + '_Taskp' + str(args.num_per_task) + '_emb_' + str(args.n_embd) + '_layer_' + str(args.n_layer) + '_gamma_' + str(args.gamma) + '_E_' + str(args.num_epochs) + '_Lr_' + str(args.lr) + '_BS_' + str(args.batch_size) + '_TW_' + str(args.time_weighted) + '_weight_decay_' + str(args.weight_decay)
        else:
            save_dir = 'models/V2_' + args.source + '_emb_' + str(args.n_embd) + '_layer_' + str(args.n_layer) + '_E_' + str(args.num_epochs) + '_Lr_' + str(args.lr) + '_BS_' + str(args.batch_size) + '_weight_decay_' + str(args.weight_decay)
        # add data and time
        if not args.soft_max:
            save_dir += '_hardmax'
        if args['mix'] != -1:
            save_dir += f'_mix_{args.mix}'
        save_dir += f'_{datetime.now().strftime("%m%d%Y_%H%M%S")}'

    if args.use_wandb:
        # Try to login to wandb if API key is provided via environment variable
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        # If no API key, wandb.init will use existing login or prompt for login
        name = f'{args.source}_EnvN{args.num_envs}_Taskp{args.num_per_task}_G{args.gamma}_E{args.num_epochs}_L{args.n_layer}_Lr{args.lr}_B{args.batch_size}_Em{args.n_embd}_W{args.weight_decay}'
        if args['mix'] != -1:
            name += f'_mix_{args.mix}'
        wandb.init(project="icrl_pretrain_v2", reinit=True, name=name)

    if args.Q:
        loss_fn = F.mse_loss
        # loss_fn = ExponentialTimeWeightedLoss(args.time_weighted)
        Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir=save_dir, softmax=args.soft_max, use_wandb=args.use_wandb, num_epochs=args.num_epochs, gamma=args.gamma, double=args.double, early_stop=getattr(args, 'early_stop', False), early_stop_patience=getattr(args, 'early_stop_patience', 10), early_stop_min_delta=getattr(args, 'early_stop_min_delta', 0.0))
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, scheduler, loss_fn, device, use_wandb=args.use_wandb,save_dir=save_dir, num_epochs=args.num_epochs)
 

def evaluate_bandits(model, config, device, num_trajectories=100, T=200, mode='greedy', eval_batch_size=None):
    model.eval()
    model.test = True
    cumulative_regrets = [0 for _ in range(num_trajectories)]
    
    # 100batch_size
    batchsize = eval_batch_size if eval_batch_size is not None else 100

    num_batches = (num_trajectories + batchsize - 1) // batchsize  # 
    
    for i in tqdm.tqdm(range(num_batches)):
        # batch
        current_batch_size = min(batchsize, num_trajectories - i * batchsize)
        
        envs = [Environment(config['act_num'], config['dim']) for _ in range(current_batch_size)]
        # best_action_indices = [env.get_best_action_index() for env in envs]
        best_action_indices = [env.get_best_action_index() for env in envs]
        best_action_rewards = [np.dot(env.action_set[best_action_index], env.w_star) for env, best_action_index in zip(envs, best_action_indices)]
        action_sets = [torch.tensor(env.get_action_set(), dtype=torch.float32).to(device).reshape(-1) for env in envs]
        # action_sets shape: [current_batch_size, num_actions*context_dim]
        action_sets = torch.stack(action_sets).reshape(current_batch_size, -1) # [current_batch_size, num_actions*context_dim]

        cumulative_regrets_ = [0 for _ in range(current_batch_size)]
        for t in range(1, T+1):

            if t == 1:
                context_actions = torch.empty((current_batch_size, 0, config['act_num']), dtype=torch.float32).to(device)
                context_rewards = torch.empty((current_batch_size, 0, 1), dtype=torch.float32).to(device)
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            else:
                x = {
                    'action_set': action_sets,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards
                }
            last_timestep_outputs = model(x)
            if mode == 'greedy':
                action_indices_tensor = last_timestep_outputs.argmax(dim=-1)  # [current_batch_size]
            elif mode == 'sample':
                action_indices_tensor = torch.distributions.Categorical(logits=last_timestep_outputs).sample()  # [current_batch_size]
            
            # numpyenv.step()
            action_indices_np = action_indices_tensor.cpu().numpy()  # [current_batch_size]
            
            # scattertensor
            action_indices_for_scatter = action_indices_tensor.unsqueeze(1)  # [current_batch_size, 1]

            rewards_ = [env.step(int(action_idx))[0] for env, action_idx in zip(envs, action_indices_np)]
            # actions_ = [env.step(action_index)[1] for env, action_index in zip(envs, action_indices)]
            
            actions_one_hot = torch.zeros(current_batch_size, 1, config['act_num']).to(device)
            actions_one_hot.scatter_(2, action_indices_for_scatter.unsqueeze(-1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(current_batch_size, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)

            expected_rewards = [np.dot(env.action_set[int(action_idx)], env.w_star) for env, action_idx in zip(envs, action_indices_np)]

            cumulative_regrets_ = [cumulative_regrets_[i] + (best_action_rewards[i] - expected_rewards[i]) for i in range(current_batch_size)]

        # batch
        start_idx = i * batchsize
        end_idx = min(start_idx + current_batch_size, num_trajectories)
        cumulative_regrets[start_idx:end_idx] = cumulative_regrets_[:end_idx-start_idx]

    model.test = False
    model.train()
    return cumulative_regrets

def check_optimal_action(eval_trajs, dim, horizon):
    if DarkroomEnv is None:
        raise ImportError("DarkroomEnv is not available. Please ensure the 'envs' module is installed or available.")
    for traj in eval_trajs:
        env = DarkroomEnv(dim, traj['goal'], horizon)
        for t in range(horizon):
            context_action = traj['context_actions'][t]
            context_state = traj['context_states'][t]
            optimal_action = env.opt_action(context_state)
            # if context_action != optimal_action:
            print(f"Optimal action is {optimal_action}, and the context action is {context_action}")
            
    return  


def evaluate_darkroom(model, config, device, eval_trajs, mode='greedy', epsilon_greedy=False):
    # set model to eval
    if DarkroomEnv is None:
        raise ImportError("DarkroomEnv is not available. Please ensure the 'envs' module is installed or available.")

    model.eval()
    model.test = True

    dim = config['dim']
    horizon = config['horizon']
    envs = [DarkroomEnv(dim, traj['goal'], horizon) for traj in eval_trajs]
    # check_optimal_action(eval_trajs, dim, horizon)
    state_dim = config['state_dim']
    cumulative_rewards = [[] for _ in range(len(envs))]

    # for t in tqdm.tqdm(range(1, 1+10*horizon)):
    for t in tqdm.tqdm(range(1, 1+horizon)):
        if t == 1:
            context_states = np.array([env.reset() for env in envs])
            context_states = torch.tensor(context_states, dtype=torch.float32).to(device).reshape(len(envs), 1, state_dim)
            # Convert list to NumPy array first

            context_actions = torch.empty((len(envs), 0, config['act_num']), dtype=torch.float32).to(device)
            context_rewards = torch.empty((len(envs), 0, 1), dtype=torch.float32).to(device)
            # print(context_states.shape)
            # print(context_actions.shape)
            # print(context_rewards.shape)
        elif t >  horizon: # remove the first timestep
            context_states = context_states[:, 1:, :]
            context_actions = context_actions[:, 1:, :]
            context_rewards = context_rewards[:, 1:, :]
            # print(context_states.shape)
        x = {
            'context_states': context_states,
            'context_actions': context_actions,
            'context_rewards': context_rewards
        }
        
        last_timestep_outputs = model(x)
        if epsilon_greedy and random.random() < 1.0/t:
            action_indices = torch.tensor([[random.randint(0, config['act_num']-1) for _ in range(len(envs))]]).to(device).reshape(len(envs), 1, 1)
        else:
            if mode == 'greedy':
                action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1).unsqueeze(1) # [batchsize, 1, 1]
            elif mode == 'sample':
                action_indices = torch.distributions.Categorical(logits=last_timestep_outputs).sample().unsqueeze(1).unsqueeze(1)

        actions= torch.zeros(len(envs), 1, config['act_num']).to(device) # [batchsize, 1, num_actions]
        actions.scatter_(2, action_indices, 1) # one hot encoding

        next_states = []
        rewards = []

        for env, action in zip(envs, actions.cpu().numpy()):
            next_state, reward, done, _ = env.step(action)

            # reset the environment if the episode is done
            if done:
                next_state = env.reset()

            cumulative_rewards[envs.index(env)].append(reward)
            next_states.append(next_state)
            rewards.append(reward)
        
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device).reshape(len(envs), 1, state_dim)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).reshape(len(envs), 1, 1)
        
        context_states = torch.cat([context_states, next_states], dim=1)
        context_actions = torch.cat([context_actions, actions], dim=1)
        context_rewards = torch.cat([context_rewards, rewards], dim=1)


    model.test = False
    model.train()
    
    return [np.sum(cumulative_reward[-100:]) for cumulative_reward in cumulative_rewards]


# def pretrain_with_plot_regret(args):
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#     print(f"Using device {device}")
#     # parse arguments
#     batch_size = args.batch_size
#     action_num = args.action_num

#     # load data
#     traj = load_data(args)
#     dataset = TrajectoryDataset(traj, device, action_num, horizon=args.horizon)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # set up model
#     config = {
#             'horizon': args.horizon,
#             'dim': args.dim,
#             'act_num': args.action_num,
#             'state_dim': args.state_dim,
#             'dropout': args.dropout,
#             'action_dim': args.action_num,
#             'n_layer': args.n_layer,
#             'n_embd': args.n_embd,
#             'n_head': args.n_head,
#             'shuffle': True,
#             'activation': args.activation,
#             'pred_q': args.Q,
#             'test': False
#         }
#     action_dim = config['action_dim']
#     model = Transformer(config, device)

#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

#     scheduler = CosineAnnealingWarmup(optimizer, learning_rate=args.lr, warmup_steps=2000, lr_decay_steps=args.num_steps, min_lr=2e-5)

#     if args.use_wandb:
#         name = f'{args.source}_{args.num_steps}_{args.lr}_{args.batch_size}'
#         wandb.init(project="icrl_pretrain_with_regret", reinit=True, name=name)

#     if args.Q:
#         loss_fn = F.mse_loss
#         model = Q_trainer_new(model, dataloader, config, optimizer, scheduler, loss_fn, device, softmax=args.soft_max, use_wandb=args.use_wandb, num_steps=args.num_steps, gamma=args.gamma, double=args.double)

#     else:
#         loss_fn = nn.CrossEntropyLoss(reduction='sum')
#         # split the dataset into training and testing
#         train_size = int(0.8 * len(traj))
#         test_size = len(traj) - train_size
#         train_traj, test_traj = torch.utils.data.random_split(traj, [train_size, test_size])
#         train_dataset = TrajectoryDataset(train_traj, device, action_num, horizon=args.horizon)
#         test_dataset = TrajectoryDataset(test_traj, device, action_num, horizon=args.horizon)
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         model = trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, loss_fn, device, use_wandb=args.use_wandb, num_steps=args.num_steps)
    
#     # save the model
#     torch.save(model.state_dict(), args.model_path)

def pretrain(args):   
           
    env = args['env']
    seed = args['seed']
    tmp_seed = seed
    # Handle num_envs and envs: if num_envs is provided, use it; otherwise use envs
    n_envs = args.get('num_envs') if args.get('num_envs') is not None else args.get('envs', 100000)
    args['envs'] = n_envs  # Ensure envs is set for consistency
    args['num_envs'] = n_envs  # Ensure num_envs is set for consistency
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    dataset_config = {
        'n_hists': args['hists'],
        'n_samples': args['samples'],
        'horizon': args['horizon'],
        'dim': args['dim'],
        'mix': args['mix'],
    }
    model_config = {
        'shuffle': args['shuffle'],
        'lr': args['lr'],
        'dropout': args['dropout'],
        'n_embd': args['n_embd'],
        'n_layer': args['n_layer'],
        'n_head': args['n_head'],
        'n_envs': args['envs'],
        'n_hists': args['hists'],
        'n_samples': args['samples'],
        'horizon': args['horizon'],
        'dim': args['dim'],
        'seed': seed,
        'Q': args['Q'],
        'soft_max': args['soft_max'],
        'mix': args['mix'],
    }

    if env == 'linear_bandit':
        action_num = 10
        H = 200
        args['act_num'] = action_num
        args['action_dim'] = action_num
        args['state_dim'] = args['dim'] * action_num
        args['H'] = H
        # Convert dict to Namespace for linear_bandit_pretrain
        args_namespace = argparse.Namespace(**args)
        print(args_namespace)
        linear_bandit_pretrain(args_namespace)

        return
    
    elif env == 'darkroom':
        if DarkroomEnv is None:
            raise ImportError("DarkroomEnv is not available. Please ensure the 'envs' module is installed or available.")
        state_dim = 2
        action_dim = 5

        path_train = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=1)
        path_eval = build_darkroom_data_filename(
            env, 100, dataset_config, mode=2)
        save_dir = build_darkroom_model_filename(env, model_config)
        
    config = {
        'env': env,
        'horizon': args['horizon'],
        'state_dim': state_dim,
        'action_dim': action_dim,
        'act_num': action_dim,
        'dim': args['dim'],
        'n_layer': args['n_layer'],
        'n_embd': args['n_embd'],
        'n_head': args['n_head'],
        'shuffle': args['shuffle'],
        'dropout': args['dropout'],
        'activation': args['activation'],
        'pred_q': args['Q'],
        'gpu': args['gpu'],
        'test': False,
    }

    model = Transformer(config, device)
    model.to(device)
    params = {
        'batch_size': args['batch_size'],
        'shuffle': True,
    }
    # log_filename = f'results/loss/{save_dir}_logs.txt'

    # with open(log_filename, 'w') as f:
    #     pass
    
    def printw(string):
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)
 
    if env == 'linear_bandit':
        # load data
        traj = load_data_new(args)
        # pretrain
        # train_traj, test_traj = train_test_split(traj, test_size=0.2)
        # define the train and test size
        train_size = int(0.8 * len(traj))
        test_size = len(traj) - train_size
        train_traj, test_traj = torch.utils.data.random_split(traj, [train_size, test_size])
        # create dataset
        train_dataset = LinearBanditDataset(train_traj, device, action_num, horizon=args.horizon)
        test_dataset = LinearBanditDataset(test_traj, device, action_num, horizon=args.horizon)

    elif env == 'darkroom':
        train_dataset = DarkRoomDataset(path_train, config)
        test_dataset = DarkRoomDataset(path_test, config)
        with open(path_eval, 'rb') as f:
        # with open(path_train, 'rb') as f:
            eval_trajs = pickle.load(f)

    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **params)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    num_steps = len(train_dataloader) * args['num_epochs']
    scheduler = CosineAnnealingWarmup(optimizer, learning_rate=args['lr'], warmup_steps=2000, lr_decay_steps=num_steps, min_lr=args['min_lr'])

    if args['use_wandb']:
        # Try to login to wandb if API key is provided via environment variable
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        # If no API key, wandb.init will use existing login or prompt for login
        name = f"{args['env']}_G{args['gamma']}_E{args['num_epochs']}_L{args['lr']}_B{args['batch_size']}_Em{args['n_embd']}_W{args['weight_decay']}"
        if args['mix'] != -1:
            name += f"_mix_{args['mix']}"
        wandb.init(project="icrl_pretrain_with_regrets_timeWeighted", reinit=True, name=name)

    print('save_dir:', save_dir)

    if args['Q']:
        print('Training Q function')
        # loss_fn = F.mse_loss
        # loss_fn = time_weighted_mse_loss
        loss_fn = ExponentialTimeWeightedLoss(args['time_weighted'])
        Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir=save_dir, softmax=args['soft_max'], use_wandb=args['use_wandb'], num_epochs=args['num_epochs'], gamma=args['gamma'], double=args['double'], eval_trajs=eval_trajs, early_stop=args.get('early_stop', False), early_stop_patience=args.get('early_stop_patience', 10), early_stop_min_delta=args.get('early_stop_min_delta', 0.0))
    else:
        print('Training policy')
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, scheduler, loss_fn, device, use_wandb=args['use_wandb'], save_dir=save_dir, num_epochs=args['num_epochs'], eval_trajs=eval_trajs)

if __name__ == '__main__':
    # if not os.path.exists('results/loss'):
    #     os.makedirs('results/loss', exist_ok=True)
    # if not os.path.exists('models'):
    #     os.makedirs('models', exist_ok=True)
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_model_args(parser)
    add_train_args(parser)

    parser.add_argument('--time_weighted', type=float, default=10)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n_epoch_store_model', type=int, default=1,
                       help='Save checkpoint every N epochs (default: 1, saves every epoch). Set to 0 to disable periodic checkpoints.')
    
    # Note: --early_stop, --early_stop_patience, and --early_stop_min_delta are already defined in add_train_args()

    args = parser.parse_args()
    n_epoch_store_model = args.n_epoch_store_model
    
    args_dict = vars(args)
    print("Args: ", args_dict)
    args_dict['horizon'] = args_dict['H']
    pretrain(args_dict)
