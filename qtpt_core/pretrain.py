import argparse
from icrl import *
from model import *
import pickle
import wandb
import tqdm
import os
import random
from datetime import datetime
from default_args import *
from dataset import LinearBanditDataset, DarkRoomDataset
from utils import build_darkroom_data_filename, build_darkroom_model_filename
from envs.darkroom import DarkroomEnv, DarkroomEnvVec
# from sklearn.model_selection import train_test_split



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


def Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir, softmax=False, use_wandb=True, num_epochs=16, gamma = 0.99, double=False, eval_trajs=None):
    # record the loss before training
    train_loss = 0.0
    test_loss = 0.0
    model.eval()

    global n_epoch_store_model


    # constryct the save directory 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
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

    # best_model_state = model.state_dict()
    # best_loss = float('inf')

    # num_epochs = num_steps // len(train_dataloader)
    step = 0
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        for batch in tqdm.tqdm(train_dataloader):
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

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     best_model_state = model.state_dict()
        if epoch % n_epoch_store_model == 0:
            torch.save(model.state_dict(), save_dir + f'/_ckpt_{epoch}.pth')
        

            with torch.no_grad():
                if config['env'] == 'linear_bandit':
                    average_regrets = evaluate_bandits(model, config, device, num_trajectories=200, T=200, mode = 'greedy')

                    # wandb logging
                    if use_wandb:
                        wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_cumulative_regrets": np.mean(average_regrets)})
                    print(f"Epoch {epoch} | Training Loss {epoch_loss} | validation Loss {test_loss} | Average Cumulative Regrets {np.mean(average_regrets)}")

                    with open(f'{save_dir}/log.csv', 'a') as f:
                        f.write(f"{epoch},{epoch_loss},{test_loss},{np.mean(average_regrets)}\n")

                elif config['env'] == 'darkroom':
                    average_rewards = evaluate_darkroom(model, config, device, eval_trajs, mode = 'greedy')

                    # wandb logging
                    if use_wandb:
                        wandb.log({"training_loss": epoch_loss, "validation_loss": test_loss, "average_rewards": np.mean(average_rewards)})
                    print(f"Epoch {epoch} | Training Loss {epoch_loss} | validation Loss {test_loss} | Average Rewards {np.mean(average_rewards)}")

                    with open(f'{save_dir}/log.csv', 'a') as f:
                        f.write(f"{epoch},{epoch_loss},{test_loss},{np.mean(average_rewards)}\n")
            

        model.train() 

    # best_model = Transformer(config, device=device)
    # best_model.load_state_dict(best_model_state)
    return model

def linear_bandit_pretrain(args):
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device {device}")
    # parse arguments
    batch_size = args.batch_size
    action_num = args.act_num

    # load data
    traj = load_data(args)
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
            save_dir = 'models/Qtransformer_' + args.source + '_emb_' + str(args.n_embd) + '_layer_' + str(args.n_layer) + '_gamma_' + str(args.gamma) + '_E_' + str(args.num_epochs) + '_Lr_' + str(args.lr) + '_BS_' + str(args.batch_size) + '_TW_' + str(args.time_weighted) + '_weight_decay_' + str(args.weight_decay)
        else:
            save_dir = 'models/transformer_' + args.source + '_emb_' + str(args.n_embd) + '_layer_' + str(args.n_layer) + '_E_' + str(args.num_epochs) + '_Lr_' + str(args.lr) + '_BS_' + str(args.batch_size) + '_weight_decay_' + str(args.weight_decay)
        # add data and time
        if not args.soft_max:
            save_dir += '_hardmax'
        if args.mix != -1:
            save_dir += f'_mix_{args.mix}'
        save_dir += f'_{datetime.now().strftime("%m%d%Y_%H%M%S")}'

    if args.use_wandb:
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        name = f'{args.source}_G{args.gamma}_TW{args.time_weighted}_E{args.num_epochs}_L{args.n_layer}_Lr{args.lr}_B{args.batch_size}_Em{args.n_embd}_W{args.weight_decay}'
        if args.mix != -1:
            name += f'_mix_{args.mix}'
        wandb.init(project="icrl_pretrain_timeWeighted", reinit=True, name=name, entity="Your wandb entity")

    if args.Q:
        # loss_fn = F.mse_loss
        loss_fn = ExponentialTimeWeightedLoss(args.time_weighted)
        Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir=save_dir, softmax=args.soft_max, use_wandb=args.use_wandb, num_epochs=args.num_epochs, gamma=args.gamma, double=args.double)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        trainer(model, train_dataloader, test_dataloader, action_dim, config, optimizer, scheduler, loss_fn, device, use_wandb=args.use_wandb,save_dir=save_dir, num_epochs=args.num_epochs)
 

def evaluate_bandits(model, config, device, num_trajectories=100, T=200, mode='greedy'):
    model.eval()
    model.test = True
    cumulative_regrets = [0 for _ in range(num_trajectories)]
    
    batchsize = 100

    for i in tqdm.tqdm(range(num_trajectories//batchsize)):
        envs = [Environment(config['act_num'], config['dim']) for _ in range(batchsize)]
        # best_action_indices = [env.get_best_action_index() for env in envs]
        best_action_indices = [env.get_best_action_index() for env in envs]
        best_action_rewards = [np.dot(env.action_set[best_action_index], env.w_star) for env, best_action_index in zip(envs, best_action_indices)]
        action_sets = [torch.tensor(env.get_action_set(), dtype=torch.float32).to(device).reshape(-1) for env in envs]
        # action_sets shape: [batchsize, num_actions*context_dim]
        action_sets = torch.stack(action_sets).reshape(batchsize, -1) # [batchsize, num_actions*context_dim]

        cumulative_regrets_ = [0 for _ in range(batchsize)]
        for t in range(1, T+1):

            if t == 1:
                context_actions = torch.empty((batchsize, 0, config['act_num']), dtype=torch.float32).to(device)
                context_rewards = torch.empty((batchsize, 0, 1), dtype=torch.float32).to(device)
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
                action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
            elif mode == 'sample':
                # action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
                action_indices = torch.distributions.Categorical(logits=last_timestep_outputs).sample().unsqueeze(1)

            rewards_ = [env.step(action_index)[0] for env, action_index in zip(envs, action_indices)]
            # actions_ = [env.step(action_index)[1] for env, action_index in zip(envs, action_indices)]
            
            actions_one_hot = torch.zeros(batchsize, 1, config['act_num']).to(device)
            actions_one_hot.scatter_(2, action_indices.unsqueeze(1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(batchsize, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)

            expected_rewards = [np.dot(env.action_set[action_index], env.w_star) for env, action_index in zip(envs, action_indices)]

            cumulative_regrets_ = [cumulative_regrets_[i] + (best_action_rewards[i] - expected_rewards[i]) for i in range(batchsize)]

        cumulative_regrets[i*batchsize:(i+1)*batchsize] = cumulative_regrets_

    model.test = False
    model.train()
    return cumulative_regrets

def check_optimal_action(eval_trajs, dim, horizon):
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
    n_envs = args['envs']
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
        args = argparse.Namespace(**args)
        print(args)
        linear_bandit_pretrain(args)

        return
    
    elif env == 'darkroom':
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
        traj = load_data(args)
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
        wandb_api_key = os.environ.get('WANDB_API_KEY')
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        name = f"{args['env']}_G{args['gamma']}_E{args['num_epochs']}_L{args['lr']}_B{args['batch_size']}_Em{args['n_embd']}_W{args['weight_decay']}"
        if args.mix != -1:
            name += f"_mix_{args['mix']}"
        wandb.init(project="icrl_pretrain_with_regrets_timeWeighted", reinit=True, name=name)

    print('save_dir:', save_dir)

    if args['Q']:
        print('Training Q function')
        # loss_fn = F.mse_loss
        # loss_fn = time_weighted_mse_loss
        loss_fn = ExponentialTimeWeightedLoss(args['time_weighted'])
        Q_trainer(model, train_dataloader, test_dataloader, config, optimizer, scheduler, loss_fn, device, save_dir=save_dir, softmax=args['soft_max'], use_wandb=args['use_wandb'], num_epochs=args['num_epochs'], gamma=args['gamma'], double=args['double'], eval_trajs=eval_trajs)
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
    parser.add_argument('--n_epoch_store_model', type=int, default=1)

    global n_epoch_store_model
    n_epoch_store_model = parser.parse_args().n_epoch_store_model

    args = vars(parser.parse_args())
    print("Args: ", args)
    args['horizon'] = args['H']
    pretrain(args)
