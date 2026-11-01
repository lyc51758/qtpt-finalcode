import numpy as np
import tqdm
from icrl import Environment, LinUCB, RandomChoose
import pandas as pd
import pickle
import argparse
from default_args import *
from utils import build_darkroom_data_filename
# Conditionally import DarkroomEnv only if needed
try:
    from envs.darkroom import DarkroomEnv
except ImportError:
    DarkroomEnv = None  # Will be checked when darkroom env is used
import os
import random

def gen_data_bandit(env, algo, num_trajectories, T=200):
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
        algo.reset()

    return trajectories, all_regrets, states_and_best_actions

def collect_data(total, num_per_task, algo, num_actions=10, dim=5, T=200):
    all_trajectories = []
    all_regrets = []
    all_states_and_best_actions = []

    assert total % num_per_task == 0 # total must be divisible by num_per_task
    num_tasks = total // num_per_task

    for _ in tqdm.tqdm(range(num_tasks)):
        env = Environment(num_actions=num_actions, context_dim=dim)
        trajectories, regrets, states_and_best_actions = gen_data_bandit(env, algo, num_per_task, T)
        all_trajectories.extend(trajectories)
        all_regrets.extend(regrets)
        all_states_and_best_actions.extend(states_and_best_actions)
    
    return all_trajectories, all_regrets, all_states_and_best_actions


def rollin_mdp(env, rollin_type):
    states = []
    actions = []
    next_states = []
    rewards = []

    state = env.reset()
    for _ in range(env.horizon):
        if rollin_type == 'uniform':
            # state = env.sample_state()
            action = env.sample_action()
        elif rollin_type == 'expert':
            action = env.opt_action(state)
        else:
            raise NotImplementedError
        next_state, reward = env.transit(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards

def get_legal_actions(state, action_num):
    """
    return the legal actions for the given state, so the agent won't crush to the wall.

    Returns:
       a binary list whose dimension is the same as the action space, 1 means legal, 0 means illegal.
    """
    legal_actions = np.ones(action_num)
    if state[0] == 0:
        legal_actions[1] = 0
    if state[0] == 9:
        legal_actions[0] = 0
    if state[1] == 0:
        legal_actions[3] = 0
    if state[1] == 9:
        legal_actions[2] = 0
    return legal_actions


def rollin_dr(env, rollin_type, expert_histories):
    states = []
    actions = []
    next_states = []
    rewards = []
    if rollin_type == 'uniform':
        state = env.reset()
        for _ in range(env.horizon):
            action = env.sample_action()
            next_state, reward, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            state = next_state
    elif rollin_type == 'ppo':
        # load history from learning history
        expert_history = expert_histories[tuple(env.goal)]
        total_steps = expert_history.shape[0]
        start_index = np.random.randint(0, total_steps - env.horizon)
        expert_history = expert_history[start_index:start_index + env.horizon]
        for i in range(env.horizon):
            state, action, reward = expert_history[i]
            # change action to one-hot
            action_one_hot = np.zeros(env.action_space.n)
            action_one_hot[action] = 1
            states.append(state)
            actions.append(action_one_hot)
            next_states.append([-1, -1])
            rewards.append(reward)
    elif rollin_type == 'expert':
        state = env.reset()
        flag = False # represent the agent is not in the goal state
        for _ in range(env.horizon):
            if not flag:
                legal_actions = get_legal_actions(state, env.action_space.n)
                action_index = random.choice(np.where(legal_actions == 1)[0])
                action = np.zeros(env.action_space.n)
                action[action_index] = 1
            else:
                action = np.array([0, 0, 0, 0, 1]) # stay in the goal state
            next_state, reward, _, _ = env.step(action)
            if reward == 1:
                flag = True
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            state = next_state   
        
    else:
        raise NotImplementedError
    
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards



def generate_dr_histories_from_envs(envs, dim, n_hists, n_samples, mix=0):

    trajs = []
    expert_histories = {}
    if mix > 0:
        for i in range(dim):
            for j in range(dim):
                expert_histories[(i, j)] = np.load(f'dr_history/training_transitions_{i}_{j}.npy', allow_pickle=True)

    rewards_list = []
    for env in tqdm.tqdm(envs):
        for j in range(n_hists):
            if np.random.rand() < mix:
                rollin_type = 'expert'
            else:
                rollin_type = 'uniform'
            (
                context_states,
                context_actions,
                context_next_states,
                context_rewards,
            ) = rollin_dr(env, rollin_type=rollin_type, expert_histories=expert_histories)
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)

                rewards_list.append(sum(context_rewards))

                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
                }

                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index

                trajs.append(traj)
    print(f"Average reward: {np.mean(rewards_list)}")
    return trajs


def generate_darkroom_histories(goals, dim, horizon, **kwargs):
    envs = [DarkroomEnv(dim, goal, horizon) for goal in goals]
    trajs = generate_dr_histories_from_envs(envs, dim, **kwargs)
    return trajs




def main():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    parser.add_argument('--algos', nargs='+', default=['linucb', 'random'])
    args = vars(parser.parse_args())
    print("Args: ", args)

    # Create data directory if it doesn't exist
    data_path = args.get('data_path', 'data/')
    os.makedirs(data_path, exist_ok=True)

    env = args['env']
    n_envs = args['envs'] if hasattr(args, 'envs') else 100000

    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']


    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
    }


    if env == 'linear_bandit':
        num_actions = 10
        dim = 5
        T = 200

        # algo_list = ['linucb', 'random']
        # algo_list = ['linucb']
        algo_list = args['algos']
        # num_per_task_list = [100000, 10000, 1000, 100, 10, 1]
        # num_per_task_list = [100, 500]
        # Support --num_per_task argument, default to 1
        num_per_task = args.get('num_per_task', 1)
        # Use user-specified --num_envs or --envs if provided, otherwise use default list
        user_num_envs = args.get('num_envs') or args.get('envs')
        if user_num_envs is not None and user_num_envs != 100000:
            # User specified a value, use it
            num_envs_list = [user_num_envs]
        else:
            # Default: generate for multiple sizes (only if using default)
            num_envs_list = [10000, 500000]
        # total = 100000
        # num_envs = 5000
        
        for algo_name in algo_list:
            if algo_name == 'linucb':
                algo = LinUCB(num_actions=num_actions, context_dim=dim)
            elif algo_name == 'random':
                algo = RandomChoose(num_actions=num_actions,context_dim=dim)
            # for num_per_task in num_per_task_list:
            for num_envs in num_envs_list:
                print(f'Generating data for {algo_name} with {num_per_task} trajectories')
                total = num_envs * num_per_task
                all_trajectories, all_regrets, all_states_and_best_actions = collect_data(total=total, num_per_task=num_per_task, algo=algo, num_actions=num_actions, dim=dim, T=T)

                df_regrets= pd.DataFrame(all_regrets)
                df_regrets.to_csv(f'data/{algo_name}_Env_num_{num_envs}_num_per_task_{num_per_task}_regrets.csv', index=False)
                df_states_and_best_actions = pd.DataFrame(all_states_and_best_actions)
                # save pickle
                df_states_and_best_actions.to_pickle(f'data/{algo_name}_Env_num_{num_envs}_num_per_task_{num_per_task}_states_and_best_actions.pkl')
                
                chunk_size = 50000  # 
                for i in range(0, total, chunk_size):
                    with open(f'data/{algo_name}_EnvNum_{num_envs}_NumperTask_{num_per_task}_trajectories_{i//chunk_size}.pkl', 'wb') as f:
                        pickle.dump(all_trajectories[i:i+chunk_size], f)
                # chunk = all_trajectories[i:i+chunk_size]
                # df_trajectories = pd.DataFrame(chunk)
                # df_trajectories.to_pickle(f'data/{algo_name}_{total//num_per_task}_{num_per_task}_trajectories_{i//chunk_size}.pkl')
    elif env == 'darkroom':
        if DarkroomEnv is None:
            raise ImportError("DarkroomEnv is required for darkroom environment. Please ensure envs/darkroom.py exists.")
        # rollin_type = args['rollin'] if hasattr(args, 'rollin_type') else 'uniform'
        mix_ratio = args['mix']
        # the mix_ratio is the ratio of expert rollin
        config.update({'dim': dim, 'mix': mix_ratio})

        n_train_envs = int(.8 * n_envs)
        n_test_envs = n_envs - n_train_envs

        goals = np.array([[(j, i) for i in range(dim)]
                         for j in range(dim)]).reshape(-1, 2)  # 10*10 grid
        
        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        train_goals = goals[:train_test_split]
        test_goals = goals[train_test_split:]

        eval_goals = np.array(test_goals.tolist() *
                              int(100 // len(test_goals)))
        
        train_goals = np.repeat(train_goals, n_envs // (dim * dim), axis=0)
        test_goals = np.repeat(test_goals, n_envs // (dim * dim), axis=0)

        train_trajs = generate_darkroom_histories(train_goals, **config)
        test_trajs = generate_darkroom_histories(test_goals, **config)
        eval_trajs = generate_darkroom_histories(eval_goals, **config)

        train_filepath = build_darkroom_data_filename(env, n_envs, config, mode=0)
        test_filepath = build_darkroom_data_filename(env, n_envs, config, mode=1)
        eval_filepath = build_darkroom_data_filename(env, 100, config, mode=2)

    else:
        raise ValueError(f'Environment {env} not supported')
    
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")



if __name__ == '__main__':
    main()
