from icrl import *
import tqdm
from model import Transformer
import pandas as pd
import argparse
import pickle
import ast

def convert_to_array(string):
    try:
        result = np.array(ast.literal_eval(string))
        if result.size == 0:
            print("Conversion resulted in an empty array:", string)
        return result
    except Exception as e:
        print("Error converting string:", string)
        print("Error message:", str(e))
        return np.array([])  # Return an empty array or np.nan if conversion fails


def expiriment(env, algo, num_trajectories, T=200):
    # algo : linucb/Tomp
    trajectories = [] # Store all trajectories
    # In this setting, s_t = \mathbb{A} = action_set!
    all_regrets = np.zeros((num_trajectories, T))
    
    states_and_best_actions = []
    # use tqdm to show progress bar

    for i in tqdm.tqdm(range(num_trajectories)):

        total_regret = 0
        regrets = []
        best_action_index = env.get_best_action_index()  # Best action doesn't change in this setup
        best_action_reward = np.dot(env.action_set[best_action_index], env.w_star)
        states, actions, rewards, action_indexs = [], [], [], []

        states_and_best_actions.append({"state": env.get_action_set(), "best_action_index": best_action_index, 'w_star': env.w_star})

        for _ in range(T):
            action_index = algo.select_action(env.action_set)
            reward, action = env.step(action_index)
            # find action
            algo.update(reward, action)
            # Calculate regret for this round and add to total
            expected_reward = np.dot(env.action_set[action_index], env.w_star)
            
            round_regret = best_action_reward - expected_reward
            total_regret += round_regret
            # print(round_regret)
            # Store state, action, reward for this round
            states.append(env.get_action_set()) 
            actions.append(action)
            rewards.append(reward)
            action_indexs.append(action_index)
            regrets.append(total_regret)

        all_regrets[i] = regrets # Store regrets for this trajectory
        trajectories.append((states, actions, rewards, action_indexs)) # Store trajectory
        # Reset env and LinUCB for next trajectory
        # env.reset()
        algo.reset()

    return trajectories, all_regrets, states_and_best_actions

def validate(args):
    config = {
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
        'pred_q': args.Q,
        'test': True
    }
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config, device)
    model.load_state_dict(torch.load(args.model_path,  map_location=torch.device(f'cuda:{args.gpu}')))
    model.to(device)
    model.eval()

    num_trajectories = args.num_trajectories
    # batchsize = args.batch_size
    batchsize = args.batch_size if args.batch_size > 0 else num_trajectories
    trajectories = []
    all_regrets = np.zeros((num_trajectories, args.test_horizon))
    initilize_prob = 0 if args.greedy else 1

    for i in tqdm.tqdm(range(num_trajectories//batchsize)):

        envs = [Environment(args.action_num, args.dim, std_variance=args.std_variance) for _ in range(batchsize)]
        total_regret = np.zeros(batchsize)
        # regrets = np.zeros((batchsize, T))
        regrets = [[] for _ in range(batchsize)]

        best_action_indexs = [env.get_best_action_index() for env in envs]  # Best action doesn't change in this setup
        best_action_rewards = [np.dot(env.action_set[best_action_index], env.w_star) for env, best_action_index in zip(envs, best_action_indexs)]
        states, actions, rewards, action_indexs = [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)]
        action_sets = [torch.tensor(env.get_action_set(), dtype=torch.float32).to(device).reshape(-1) for env in envs]
        # action_sets shape: [batchsize, num_actions*context_dim]
        action_sets = torch.stack(action_sets).reshape(batchsize, -1) # [batchsize, num_actions*context_dim]

        for t in range(1, args.test_horizon+1):
            if t == 1:
                context_actions = torch.empty((batchsize, 0, args.action_num), dtype=torch.float32).to(device)
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
            random_number = np.random.rand(batchsize)
            # if random_number < initilize_prob/np.sqrt(t):
            with torch.no_grad():
                last_timestep_outputs = model(x)
                if args.Q:
                    action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
                else:
                    action_indices = torch.multinomial(F.softmax(last_timestep_outputs, dim=-1), 1)
                for k, rand_num in enumerate(random_number):
                    if rand_num < initilize_prob/t:
                        action_indices[k] = torch.randint(0, args.action_num, (1, 1)).to(device)

            # if random_number < initilize_prob/t:
            # # if t <= 30:
            #     action_indices = torch.randint(0, args.action_num, (batchsize, 1)).to(device)
            #     # choose t%num_actions
            #     # action_indices = torch.tensor([t%num_actions]*batchsize).to(device).unsqueeze(1)
            # else:
            #     last_timestep_outputs = model(x) 
            #     # last_timestep_outputs shape: [batchsize, num_actions]
            #     if args.Q:
            #         action_indices = torch.multinomial(F.softmax(last_timestep_outputs, dim=-1), 1)
            #     else:
            #         action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
                # softmax
                # tao = initial_tao/np.sqrt(t)
                # action_indices = torch.multinomial(F.softmax(last_timestep_outputs/tao, dim=-1), 1) #dimension [batchsize, 1]
                
            # last_timestep_outputs = model(x) 
            # [2*t-1].argmax().item()

            rewards_ = [env.step(action_index)[0] for env, action_index in zip(envs, action_indices)]
            actions_ = [env.step(action_index)[1] for env, action_index in zip(envs, action_indices)]
            # print(len(actions_))
            # find action
            actions_one_hot = torch.zeros(batchsize, 1, args.action_num).to(device)
            actions_one_hot.scatter_(2, action_indices.unsqueeze(1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(batchsize, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)
            # print(context_rewards.shape)
            # print(context_actions.shape)
            
            expected_rewards = [np.dot(env.action_set[action_index], env.w_star) for env, action_index in zip(envs, action_indices)]
            
            # round_regret = best_action_reward - expected_reward
            round_regrets = [best_action_reward - expected_reward for best_action_reward, expected_reward in zip(best_action_rewards, expected_rewards)]
            # print(round_regret)
            # total_regret += round_regret
            total_regret += round_regrets
            for j in range(batchsize):
                regrets[j].append(total_regret[j])
                states[j].append(envs[j].get_action_set())
                actions[j].append(actions_[j])
                rewards[j].append(rewards_[j])
                action_indexs[j].append(action_indices[j].item())

        # all_regrets[i] = regrets # Store regrets for this trajectory
        all_regrets[i*batchsize:(i+1)*batchsize] = regrets
        trajectories.append((states, actions, rewards, action_indexs)) # Store trajectory
    
    df_regrets = pd.DataFrame(all_regrets)
    df_regrets.to_csv(args.save_path, index=False)
    # plot the regrets
    plt.plot(all_regrets.mean(axis=0))
    plt.savefig(args.save_path.replace('.csv', '_regret.png'))

def validate_with_training_data(args):
    if args.data_path == '':
        if args.source == 'linucb':
            # df_states_and_best_actions = pd.read_csv('data/linucb_states_and_best_actions.csv') # states, best_action_index, w_star
            df_states_and_best_actions = pd.read_pickle('data/linucb_states_and_best_actions_0731.pkl')
        else:
            # df_states_and_best_actions = pd.read_csv('data/random_states_and_best_actions.csv')
            df_states_and_best_actions = pd.read_pickle('data/random_states_and_best_actions_0804.pkl')
            # df_states_and_best_actions = pd.read_csv('data/random_states_and_best_actions_same.csv')
    else:
        df_states_and_best_actions = pd.read_pickle(args.data_path)
        # print(df_states_and_best_actions.head())
    # state = np.array([
    #         [-0.36390933, 0.78064457, 0.17855494, -0.58462594, 0.37569906],
    #         [0.49662528, 0.86504511, -0.70453884, 0.1949896, 0.66459615],
    #         [-0.05148111, -0.04939751, 0.04231532, 0.49244914, -0.91001613],
    #         [-0.55127472, -0.59447087, -0.72179968, 0.84531335, -0.78662834],
    #         [-0.42916177, 0.77980121, -0.99823992, -0.45904169, -0.69686133],
    #         [0.84416636, -0.1314792, -0.09150188, 0.32386293, 0.73558308],
    #         [0.54236185, 0.27361654, 0.27656773, -0.91723395, 0.80446515],
    #         [0.62921347, -0.79630122, 0.28290665, -0.1104897, 0.31548038],
    #         [0.42354193, 0.46456236, -0.30787742, -0.80987277, 0.91856553],
    #         [-0.91987704, -0.53422356, -0.70974764, 0.80793503, -0.19849932]
    #     ])
    # best_action_index = 1
    # w_star = np.array([0.45543474, 0.52693406, 0.08886499, 0.91319357, 0.33159966])
    
    # df_states_and_best_actions = pd.DataFrame({'states': [state.tolist()]*args.num_trajectories, 'best_action_index': [best_action_index]*args.num_trajectories, 'w_star': [w_star.tolist()]*args.num_trajectories})
    # df_states_and_best_actions['state'] = df_states_and_best_actions['state'].apply(lambda x: convert_to_array(x).reshape(-1, 5))
    # df_states_and_best_actions['w_star'] = df_states_and_best_actions['w_star'].apply(convert_to_array)

    config = {
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
        'pred_q': args.Q,
        'test': True
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = Transformer(config, device)
    model = Transformer(config, device)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    
    
    states = np.array(df_states_and_best_actions['state'].tolist())
    best_action_indices = df_states_and_best_actions['best_action_index'].to_numpy()
    w_star = np.array(df_states_and_best_actions['w_star'].tolist())

    # print(states.shape)

    num_trajectories = args.num_trajectories
    assert num_trajectories <= len(states), 'Number of trajectories should be less than the number of training data'

    all_regrets = np.zeros((num_trajectories, args.test_horizon))
    batchsize = args.batch_size if args.batch_size > 0 else num_trajectories
    trajectories = []
    initialize_prob = 0 if args.greedy else 1

    for i in tqdm.tqdm(range(num_trajectories//batchsize)):
        total_regrets = np.zeros(batchsize)
        regrets = [[] for _ in range(batchsize)]

        best_action_indexs = best_action_indices[i*batchsize:(i+1)*batchsize]
        # best_action_rewards = [np.dot(states[i, best_action_index], w_star[i]) for i, best_action_index in enumerate(best_action_indexs)]
        best_action_rewards = [np.dot(states[i*batchsize+j, best_action_index], w_star[i*batchsize+j]) for j, best_action_index in enumerate(best_action_indexs)]
        # states, actions, rewards, action_indices = [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)]
        actions, rewards, action_indices = [[] for _ in range(batchsize)], [[] for _ in range(batchsize)], [[] for _ in range(batchsize)]
        action_sets = torch.Tensor(states[i*batchsize:(i+1)*batchsize]).reshape(batchsize, -1).to(device)
        
        for t in range(1, args.test_horizon+1):
            if t == 1:
                context_actions = torch.empty((batchsize, 0, args.action_num), dtype=torch.float32).to(device)
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
            random_number = np.random.rand(batchsize)

            with torch.no_grad():
                last_timestep_outputs = model(x)
                if args.Q:
                    action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)
                else:
                    action_indices = torch.multinomial(F.softmax(last_timestep_outputs, dim=-1), 1)
                for k, rand_num in enumerate(random_number):
                    if rand_num < initialize_prob/t:
                        action_indices[k] = torch.randint(0, args.action_num, (1, 1)).to(device)


            # if random_number < initialize_prob/t:
            #     action_indices = torch.randint(0, args.action_num, (batchsize, 1)).to(device)
            # else:
            #     last_timestep_outputs = model(x) 
            #     if args.Q:
            #         action_indices = torch.multinomial(F.softmax(last_timestep_outputs, dim=-1), 1)
            #     else:
            #         action_indices = last_timestep_outputs.argmax(dim=-1).unsqueeze(1)

            # rewards_ = [np.dot(states[i][action_index], w_star[i]) for i, action_index in enumerate(action_indices)]
            # print(action_indices, action_indices.shape)
            # print(states[i, action_indices], w_star[i])
            # for action_index in action_indices:
            #     print(action_index.cpu().item())
            #     # print(w_star, w_star.shape)
            #     # print(states.shape())
            #     print(states[i, action_index.cpu().item()])
                # print(w_star[i])
            rewards_ = [np.dot(states[i*batchsize+j][action_index.cpu().item()], w_star[i*batchsize+j]) for j, action_index in enumerate(action_indices)]
            actions_ = [states[i*batchsize+j][action_index.cpu().item()] for j, action_index in enumerate(action_indices)]
            actions_one_hot = torch.zeros(batchsize, 1, args.action_num).to(device)
            actions_one_hot.scatter_(2, action_indices.unsqueeze(1), 1)

            reward_tensor = torch.tensor(rewards_, dtype=torch.float32).to(device).reshape(batchsize, 1, 1)
            
            context_actions = torch.cat([context_actions, actions_one_hot], dim=1)
            context_rewards = torch.cat([context_rewards, reward_tensor], dim=1)

            expected_rewards = [np.dot(states[i*batchsize+j][action_index.cpu().item()], w_star[i*batchsize+j]) for j, action_index in enumerate(action_indices)]
            round_regrets = [best_action_reward - expected_reward for best_action_reward, expected_reward in zip(best_action_rewards, expected_rewards)]
            total_regrets += round_regrets

            for j in range(batchsize):
                regrets[j].append(total_regrets[j])
                # states[j].append(states[j])
                actions[j].append(actions_[j])
                rewards[j].append(rewards_[j])
                # action_indices[j].append(action_indices[j].item())

        all_regrets[i*batchsize:(i+1)*batchsize] = regrets
        trajectories.append((states, actions, rewards, action_indices)) # Store trajectory

    df_regrets = pd.DataFrame(all_regrets)
    df_regrets.to_csv(args.save_path, index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain the model')
    parser.add_argument('--source', type=str, default='linucb', help='source of the data')
    parser.add_argument('--batch_size', type=int, default=-1, help='batch size')
    parser.add_argument('--state_dim', type=int, default=50, help='state dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--action_num', type=int, default=10, help='number of actions')
    parser.add_argument('--horizon', type=int, default=200, help='horizon')
    parser.add_argument('--dim', type=int, default=5, help='dimension of the action')
    parser.add_argument('--n_layer', type=int, default=8, help='number of layers')
    parser.add_argument('--n_embd', type=int, default=32, help='embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads')
    parser.add_argument('--activation', choices=['relu', 'softmax'], default='relu', help='activation function')
    parser.add_argument('--Q', action='store_true', help='train Q function')
    parser.add_argument('--training_data', action='store_true', help='whether use training data to validate')
    parser.add_argument('--num_trajectories', type=int, default=1000, help='number of trajectories to valide')
    parser.add_argument('--model_path', type=str, default='models/pretrained_transformer_random_softmax_gamma_1_new.pth', help='model to validate')
    parser.add_argument('--save_path', type=str, default='data/_regrets.csv', help='path to save the regrets')
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2, 3], default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--data_path', type=str, default='', help='path to the training data')
    parser.add_argument('--greedy', action='store_true', help='use greedy policy') # if not, epsilon=1/t
    parser.add_argument('--std_variance', type=float, default=1.5, help='standard deviation of the reward')
    parser.add_argument('--test_horizon', type=int, default=200, help='horizon for testing')
    args = parser.parse_args()
    if args.training_data:
        validate_with_training_data(args)
    else:
        validate(args)
    # env = Environment(num_actions=10, context_dim=5, std_variance=1.5)
    # # linucb = LinUCB(num_actions=10, context_dim=5)
    # randomchoose = RandomChoose(num_actions=10, context_dim=5)
    # T = 200
    # num_trajectories = 100000
    # trajectories, all_regrets, states_and_best_actions = expiriment(env, randomchoose, num_trajectories)
    # # trajectories, all_regrets, states_and_best_actions = expiriment(env, linucb, num_trajectories)
    # df_regrets= pd.DataFrame(all_regrets)
    # df_regrets.to_csv('data/randomchoose_regrets_same.csv', index=False)

    # # store state and best action for each trajectory
    # df_states_and_best_actions = pd.DataFrame(states_and_best_actions)
    # df_states_and_best_actions.to_csv('data/random_states_and_best_actions_same.csv', index=False)

    # #  trajectories 
    # chunk_size = 20000  # 
    # for i in range(0, len(trajectories), chunk_size):
    #     with open(f'data/random_trajectories_part{i//chunk_size}_same.pkl', 'wb') as f:
    #         pickle.dump(trajectories[i:i+chunk_size], f)


    # #  trajectories 
    # chunk_size = 20000  # 
    # for i in range(0, len(trajectories), chunk_size):
    #     with open(f'data/random_trajectories_part{i//chunk_size}_same.pkl', 'wb') as f:
    #         pickle.dump(trajectories[i:i+chunk_size], f)

