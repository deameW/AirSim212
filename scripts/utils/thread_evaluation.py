import random

import torch.nn
from scipy import spatial
from PyQt5 import QtCore
from configparser import ConfigParser

from torch import nn, autocast
from torch.cuda.amp import GradScaler

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

#
# def main():
#     eval_path = r'D:\AirSim\Models\SAC'
#     config_file = eval_path + '/config/config.ini'
#     model_file = eval_path + '/models/model_sb3.zip'
#
#     eval_ep_num = 1
#     evaluate_thread = EvaluateThread(eval_path, config_file, model_file,
#                                      eval_ep_num)
#     evaluate_thread.run()


def randFun(coverage):
    x = random.randint(-350, 350)
    y = random.randint(-350, 350)
    z = random.randint(1, 15)
    return [x, y, z]


def mutator(arg, l):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    x, y, z = arg[0], arg[1], arg[2]
    print(f"State before mutate: {[x, y, z]}")

    eval_ep_num = 1
    et = EvaluateThread(eval_path, config_file, model_file, eval_ep_num, "NH_center", None, [x, y, z])
    obs = et.env.reset()
    model = SAC.load(model_file, env=et.env, device=device)

    # 统一处理obs转换
    def process_obs(observation):
        return torch.tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(
            0).requires_grad_(True)

    # 使用混合精度训练
    scaler = GradScaler()

    with autocast("cuda"):
        obs_tensor = process_obs(obs)
        with torch.no_grad():
            unscaled_action, _ = model.predict(obs, deterministic=True)
            unscaled_action_tensor = torch.tensor(unscaled_action, dtype=torch.float32, device=device).unsqueeze(0)

        new_obs, reward, done, info = et.env.step(unscaled_action)

        # 第二次动作预测
        with torch.no_grad():
            unscaled_action_1, _ = model.predict(new_obs, deterministic=True)
            new_obs_1, reward_1, done_1, info_1 = et.env.step(unscaled_action_1)

        new_obs_1_tensor = process_obs(new_obs_1)
        unscaled_action_1_tensor = torch.tensor(unscaled_action_1, dtype=torch.float32, device=device).unsqueeze(0)

        # 计算目标Q值
        with torch.no_grad():
            q1_next, q2_next = model.critic(new_obs_1_tensor, unscaled_action_1_tensor)
            q_1 = torch.min(q1_next, q2_next)
            if torch.isnan(q_1).any() or torch.isinf(q_1).any():
                print("NaN/inf detected in q_1")
                noise_range = 0.5
                x = x + random.uniform(-noise_range, noise_range)
                y = y + random.uniform(-noise_range, noise_range)
                z = z + random.uniform(-noise_range, noise_range)

                return [x, y, z]

        # 计算当前Q值
        pred_q1, pred_q2 = model.critic(obs_tensor, unscaled_action_tensor)
        pred = torch.min(pred_q1, pred_q2)

        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("pred contains NaN/inf values")
            # 生成小的随机噪声
            noise_range = 0.5
            x = x + random.uniform(-noise_range, noise_range)
            y = y + random.uniform(-noise_range, noise_range)
            z = z + random.uniform(-noise_range, noise_range)

            return [x, y, z]

        # 创建标签
        label = pred.detach().clone()
        reward = float(reward)
        with torch.no_grad():
            q_target = torch.tensor(reward + 0.9 * q_1, device=device)
            label = label.view_as(q_target).expand_as(q_target)
            label.copy_(q_target)

        # 损失计算与反向传播
        criterion = nn.MSELoss()
        loss = criterion(pred, label)
        model.critic.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(model.critic.optimizer)
        scaler.update()

    # 梯度处理
    if obs_tensor.grad is None:
        print("No gradients computed")
        return [x, y, z]

    # 计算梯度统计量
    grad = obs_tensor.grad.abs().mean(dim=(0, 2, 3))
    grad_x = grad[0].item()
    grad_y = grad[1].item()

    # 参数更新（添加梯度裁剪）
    lr = 0.001
    x = max(min(x - grad_x * lr, 20), -20)
    y = max(min(y - grad_y * lr, 20), -20)
    z = 1  # 固定值更新策略

    # 释放内存
    del obs_tensor, unscaled_action_tensor, new_obs_1_tensor, unscaled_action_1_tensor, pred_q1, pred_q2, pred, label
    torch.cuda.empty_cache()

    return [x, y, z]



def DRLFuzz(num, n, l, alpha, theta, coverage, mu, metric_list):
    """
    覆盖测试函数
    :param num: 初始化数量
    :param n: 迭代次数
    :param l: ？
    :param alpha: ？
    :param theta: 失败得分阈值
    :param coverage: 是否是覆盖测试
    :param mu: 测试用例生成方式
    :param metric_list: 覆盖率指标列表
    :return: CoverageTestResponse 对象
    """
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
                if mu == "genetic":
                    st = nga_mutator(initialPosition[idx[i]], l)
                if mu == "grad":
                    st = mutator(initialPosition[idx[i]], l)
                if st != initialPosition[idx[i]]:
                    initialPosition[idx[i]] = st
                else:
                    initialPosition[idx[i]] = randFun(coverage)
            else:
                initialPosition[idx[i]] = randFun(coverage)
        resultPool.add(tuple(st))
        print("resultPool len: ", len(resultPool))
        print("result Pool:")
        for i in resultPool:
            print(i)

    return resultPool


def nga_mutator(arg, mutation_rate=0.15, niche_radius=2.0):
    """
    小生境遗传算法变异方法
    :param arg: 当前状态参数 [x, y, z]
    :param mutation_rate: 基础变异率
    :param niche_radius: 小生境影响半径
    :return: 变异后的新参数
    """
    x, y, z = arg

    # 使用全局KD树计算小生境密度
    if kdTree is not None and kdTree.data.size > 0:
        # 查询最近邻居距离（考虑x,y坐标）
        # dist, _ = kdTree.query(np.array([x, y]), k=1)
        # density = 1 / (dist + 1e-6)  # 计算密度值
        adaptive_rate = mutation_rate * (1 + niche_radius / (1 + 0.6))  # 自适应变异率
    else:
        adaptive_rate = mutation_rate

    # 高斯变异（主要变异方式）
    x += np.random.normal(0, adaptive_rate)
    y += np.random.normal(0, adaptive_rate)

    # 均匀变异（补充多样性）
    if np.random.rand() < 0.2:
        x += np.random.uniform(-adaptive_rate, adaptive_rate)
        y += np.random.uniform(-adaptive_rate, adaptive_rate)

    # 边界约束（保持与原始代码一致）
    x = np.clip(x, -20, 20)
    y = np.clip(y, -20, 20)
    z = 1  # 固定z值

    return [x, y, z]


if __name__ == "__main__":
    try:
        # main()
        # result = DRLFuzz(1, 10, 10, 0.1, 100, True)
        result = DRLFuzz(1, 1, 1, 0.1, 100, True, "genetic", []) # grad or genetic
        # run_eval_multi()
    except KeyboardInterrupt:
        print('system exit')


class TestRecord:
    def __init__(self, GenerateStates, ResultFilePath, CoverageResult, Extra):
        self.GenerateStates = GenerateStates
        self.ResultFilePath = ResultFilePath
        self.CoverageResult = CoverageResult
        self.Extra = Extra


class CoverageTestResponse:
    def __init__(self, CoverageTestTaskID, Records):
        self.CoverageTestTaskID = CoverageTestTaskID
        self.Records = Records


# 假设的 DRLFuzz 方法
def DRLFuzz(model_name, scene, coverage_metrics, mutation_method, iteration):
    # 这里只是示例返回，实际应实现具体逻辑
    coordinates = [UAVCordinates(1, 2, 3), UAVCordinates(4, 5, 6)]
    coverage_result = {metric: "90.00%" for metric in coverage_metrics}
    result_file_path = f"{model_name}_{mutation_method}_results.txt"
    return coordinates, coverage_result, result_file_path


def coverage_test(CoverageTestTaskID, TargetModels, Scene, CoverageMetrics, MutationMethods, Iteration, extra):
    """
    覆盖测试函数
    :param CoverageTestTaskID: 评测任务唯一标识
    :param TargetModels: 目标模型列表
    :param Scene: 任务场景
    :param CoverageMetrics: 覆盖指标列表
    :param MutationMethods: 测试用例生成方式列表
    :param Iteration: 迭代次数
    :param extra: 其他待拓展信息
    :return: CoverageTestResponse 对象
    """
    records = {}
    for model in TargetModels:
        model_records = {}
        for method in MutationMethods:
            # 调用 DRLFuzz 方法
            coordinates, coverage_result, result_file_path = DRLFuzz(model.ModelName, Scene, CoverageMetrics, method,
                                                                     Iteration)
            generate_states = [GenerateState(cord) for cord in coordinates]
            extra_info = "Some extra info"
            test_record = TestRecord(generate_states, result_file_path, coverage_result, extra_info)
            model_records[method] = test_record
        records[model.ModelName] = model_records

    response = CoverageTestResponse(CoverageTestTaskID, records)
    return response


# 结构定义
class UploadModelInfo:
    def __init__(self, ModelName, ModelPath):
        self.ModelName = ModelName
        self.ModelPath = ModelPath


class GenerateState:
    def __init__(self, UAVState):
        self.UAVState = UAVState


class UAVCordinates:
    def __init__(self, X, Y, Z):
        self.X = X
        self.Y = Y
        self.Z = Z
