import random

import torch.nn
from scipy import spatial
from PyQt5 import QtCore
from configparser import ConfigParser
from stable_baselines3 import TD3, SAC, PPO, DDPG
import numpy as np
import gym_env
import gym
import math
import os
import sys
import cv2
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
sys.path.append(
    r"C:\Users\QianWu\Documents")

eval_path = r'D:\AirSim\Models\SAC'
config_file = eval_path + '/config/config.ini'
model_file = eval_path + '/models/model_sb3.zip'
savePath = "./DRLResult/drl_result.txt"


def rule_based_policy(obs):
    '''
    custom linear policy
    used for LGMD compare
    '''
    action = 0
    # 将obs从1~-1转换成0~1
    obs = np.squeeze(obs, axis=0)

    for i in range(5):
        obs[i] = obs[i] / 2 + 0.5

    # obs_weight_depth = np.array([1.0, 3.0, 5.0, -3.0, -1.0, 3.0])
    obs_weight = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action = obs * obs_weight

    action_sum = np.sum(action)

    if action_sum > math.radians(40):
        action_sum = math.radians(40)
    elif action_sum < -math.radians(40):
        action_sum = -math.radians(40)

    return np.array([action_sum])


resultPool = set()
allStates = set()
resultNum = list()
delta = 20
innerDelta = 20
kdTree = None


def getDistance(new_obs):
    if kdTree is None:
        return np.inf
    else:
        dist, _ = kdTree.query(np.array(list(new_obs.flatten())))
        return dist


class EvaluateThread(QtCore.QThread):
    # signals
    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env="NH_center", eval_dynamics=None,
                 initial_Position=[0, 0, 0]):
        super(EvaluateThread, self).__init__()
        self.obs = None
        self.results = None
        # self.initialPosition = None
        print("init training thread")

        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)
        self.initial_position = initial_Position

        # change eval_env and eval_dynamics if is not None
        if eval_env is not None:
            self.cfg.set('options', 'env_name', eval_env)

        if eval_env == 'NH_center':
            self.cfg.set('environment', 'accept_radius', str(1))

        if eval_dynamics is not None:
            self.cfg.set('options', 'dynamic_name', eval_dynamics)

        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')

    def terminate(self):
        print('Evaluation terminated')

    def run(self):
        # self.run_rule_policy()
        self.run_drl_model()

    # 相当于test
    def run_drl_model(self):
        print('start evaluation')
        algo = self.cfg.get('options', 'algo')
        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        elif algo == 'SAC':
            model = SAC.load(self.model_file, env=self.env)
        elif algo == 'PPO':
            model = PPO.load(self.model_file, env=self.env)
        elif algo == 'DDPG':
            model = DDPG.load(self.model_file, env=self.env)
        else:
            raise Exception('algo set error {}'.format(algo))
        self.env.model = model

        obs = self.env.reset()
        # self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        episode_successes = []
        episode_crashes = []
        traj_list_all = []
        action_list_all = []
        state_list_all = []
        obs_list_all = []

        traj_list = []
        action_list = []
        state_raw_list = []
        step_num_list = []
        obs_list = []
        cv2.waitKey()

        # tmp = obs
        while episode_num < self.eval_ep_num:
            unscaled_action, _ = model.predict(obs, deterministic=True)
            # print("episode:", episode_num)
            time_step += 1

            new_obs, reward, done, info, = self.env.step(unscaled_action)
            pose = self.env.dynamic_model.get_position()
            traj_list.append(pose)
            action_list.append(unscaled_action)
            state_raw_list.append(self.env.dynamic_model.state_raw)
            obs_list.append(obs)

            obs = new_obs
            reward_sum[-1] += reward

            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                maybe_is_crash = info.get('is_crash')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                episode_successes.append(float(maybe_is_success))
                episode_crashes.append(float(maybe_is_crash))
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()
                if info.get('is_success'):
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get('step_num'))
                elif info.get('is_crash'):
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)
                # traj_list.append(info)
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)
                traj_list = []
                action_list = []
                state_raw_list = []
                obs_list = []

        # save trajectory data in eval folder
        eval_folder = self.eval_path + '/eval_{}_{}_{}'.format(self.eval_ep_num, self.eval_env, self.eval_dynamics)
        os.makedirs(eval_folder, exist_ok=True)
        np.save(eval_folder + '/traj_eval',
                np.array(traj_list_all, dtype=object))
        np.save(eval_folder + '/action_eval',
                np.array(action_list_all, dtype=object))
        np.save(eval_folder + '/state_eval',
                np.array(state_list_all, dtype=object))
        np.save(eval_folder + '/obs_eval',
                np.array(obs_list_all, dtype=object))

        print('Average episode reward: ', reward_sum[:self.eval_ep_num].mean(),
              'Success rate:', np.mean(episode_successes),
              'Crash rate: ', np.mean(episode_crashes),
              'average success step num: ', np.mean(step_num_list))

        results = [reward_sum[:self.eval_ep_num].mean(), np.mean(episode_successes), np.mean(episode_crashes),
                   np.mean(step_num_list)]

        print(results)
        np.save(eval_folder + '/results', np.array(results))

        self.results = results[0]
        self.obs = obs
        return

    def run_rule_policy(self):
        obs = self.env.reset()
        episode_num = 0
        time_step = 0
        reward_sum = np.array([.0])
        while episode_num < self.eval_ep_num:
            unscaled_action = rule_based_policy(obs)
            time_step += 1
            new_obs, reward, done, info, = self.env.step(unscaled_action)
            reward_sum[-1] += reward

            obs = new_obs
            if done:
                episode_num += 1
                maybe_is_success = info.get('is_success')
                print('episode: ', episode_num, ' reward:', reward_sum[-1],
                      'success:', maybe_is_success)
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()


def main():
    eval_path = r'D:\AirSim\Models\SAC'
    config_file = eval_path + '/config/config.ini'
    model_file = eval_path + '/models/model_sb3.zip'

    eval_ep_num = 1
    evaluate_thread = EvaluateThread(eval_path, config_file, model_file,
                                     eval_ep_num)
    evaluate_thread.run()


def run_eval_multi():
    # run evaluation for multi models
    eval_logs_name = 'Maze'
    eval_logs_path = 'logs_eval/' + eval_logs_name
    eval_ep_num = 50
    eval_env_name = 'NH_center'        # 1-Trees 2-SimpleAvoid 3-NH_center
    eval_dynamic_name = 'SimpleMultirotor'  # 1-SimpleMultirotor or Multirotor

    model_list = []
    for train_name in os.listdir(eval_logs_path):
        for repeat_name in os.listdir(eval_logs_path + '/' + train_name):
            model_path = eval_logs_path + '/' + train_name + '/' + repeat_name
            model_list.append(model_path)
            # print(model_path)

    # evaluate model according to model path
    eval_num = len(model_list)
    results_list = []

    for i in tqdm(range(eval_num)):
        eval_path = model_list[i]
        config_file = eval_path + '/config/config.ini'
        model_file = eval_path + '/models/model_sb3.zip'

        print(i, eval_path)
        evaluate_thread = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, eval_env_name,
                                         eval_dynamic_name)
        results = evaluate_thread.run()
        results_list.append(results)

        del evaluate_thread

    # save all results in a numpy file
    print(results_list)
    np.save('logs_eval/results/eval_{}_{}_{}_{}'.format(eval_ep_num, eval_logs_name, eval_env_name, eval_dynamic_name),
            np.array(results_list))


def randFun(coverage):
    x = random.randint(-350, 350)
    y = random.randint(-350, 350)
    z = random.randint(1, 15)
    return [x, y, z]


def mutator(arg, l):
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dddddddddddddddddddddddd", device)

    x = arg[0]
    y = arg[1]
    z = arg[2]
    eval_ep_num = 1
    et = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, "NH_center", None, [x, y, 5])
    obs = et.env.reset()
    model = SAC.load(model_file, env=et.env, device=device)

    # Convert obs to torch tensor without flattening, ensure it matches the expected shape
    obs_tensor = torch.tensor(obs, dtype=torch.float32, requires_grad=True, device=device).reshape(1, 2, 60, 90)

    unscaled_action, _ = model.predict(obs, deterministic=True)
    unscaled_action_tensor = torch.tensor(unscaled_action, dtype=torch.float32, requires_grad=True, device=device).unsqueeze(0)

    new_obs, reward, done, info = et.env.step(unscaled_action)

    # Get next action and new observations
    unscaled_action_1, _ = model.predict(new_obs, deterministic=True)
    new_obs_1, reward_1, done_1, info_1 = et.env.step(unscaled_action_1)

    # Convert new observations to torch tensors without flattening
    new_obs_1_tensor = torch.tensor(new_obs_1, dtype=torch.float32, device=device).reshape(1, 2, 60, 90)
    unscaled_action_1_tensor = torch.tensor(unscaled_action_1, dtype=torch.float32, requires_grad=True, device=device).unsqueeze(0)

    # Ensure shapes are correct
    print(f"new_obs_1_tensor shape: {new_obs_1_tensor.shape}")
    print(f"unscaled_action_1_tensor shape: {unscaled_action_1_tensor.shape}")

    # Get Q-values from the model
    q_1 = model.critic(new_obs_1_tensor, unscaled_action_1_tensor)
    pred = model.critic(obs_tensor, unscaled_action_tensor)

    # Debugging: check if pred and q_1 are tuples
    print(f"Type of q_1: {type(q_1)}")
    print(f"Type of pred: {type(pred)}")

    # Assuming q_1 and pred are tuples, we take the first element if necessary
    if isinstance(q_1, tuple):
        q_1 = q_1[0]
    if isinstance(pred, tuple):
        pred = pred[0]

    #todo 这里a的值还是不对，导致走了兜底
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("pred tensor contains NaN or inf values.")
        return [x, y, z]

    a = torch.argmax(pred).item()
    print(f"Index from argmax: {a}")

    # 检查索引是否合法
    if a < 0 or a >= pred.size(1):
        print(f"Invalid index {a} for tensor of size {pred.size()}")
        return [x, y, z]

    label = pred.clone().detach()
    # 确保 reward 是一个数值类型
    reward = float(reward)
    label[0][a] = reward + 0.9 * torch.max(q_1).item()

    criterion = torch.nn.MSELoss()
    loss = criterion(pred, label)

    # Debugging: check loss value
    print(f"Loss: {loss.item()}")

    # Randomly generate some gradients for demonstration purposes
    grad_x = random.uniform(-1, 1)
    grad_y = random.uniform(-1, 1)

    print(f"Randomly generated gradients: grad_x={grad_x}, grad_y={grad_y}")

    x = x - grad_x * l
    y = y - grad_y * l

    # Apply limits to x and y
    x = max(min(x, 20), 20)
    y = max(min(y, 20), 20)
    z = 1

    return [x, y, z]

def DRLFuzz(num, n, l, alpha, theta, coverage):
    eval_ep_num = 1

    global kdTree
    score = list()
    initialPosition = list()

    for _ in range(num):
        s = randFun(coverage)
        initialPosition.append(s)
        score.append(0)
        allStates.add(tuple(s))

    # 迭代n次
    for k in range(n):
        for i in range(num):
            evaluate_thread = EvaluateThread(eval_path, config_file, model_file,
                                             eval_ep_num, "NH_center", None,
                                             [initialPosition[i][0], initialPosition[i][1], initialPosition[i][2]])
            evaluate_thread.run()
            score[i], obs = evaluate_thread.results, evaluate_thread.obs

            if score[i] < theta:
                resultPool.add(tuple(evaluate_thread.initial_position))

        kdTree = spatial.KDTree(data=np.array(list(allStates)), leafsize=10000)
        print("iteration {} failed cases num:{}".format(k + 1, len(resultPool)))
        resultNum.append(len(resultPool))

        idx = sorted(range(len(score)), key=lambda x: score[x])  # 得到score中元素排序后的对应索引（从小到大）
        for i in range(num):
            if i < int(num):
                # 对reward最小的进行变异
                st = mutator(initialPosition[idx[i]], l)
                if st != initialPosition[idx[i]]:
                    initialPosition[idx[i]] = st
                else:
                    initialPosition[idx[i]] = randFun(coverage)
            else:
                initialPosition[idx[i]] = randFun(coverage)
        print("resultPool len: ", len(resultPool))
        print("result Pool:")
        for i in resultPool:
            print(i)

    return resultPool


if __name__ == "__main__":
    try:
        # main()
        result = DRLFuzz(1, 10, 10, 0.1, 100, True)
        # run_eval_multi()
    except KeyboardInterrupt:
        print('system exit')
