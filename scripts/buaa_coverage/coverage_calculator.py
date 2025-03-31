import numpy as np
from typing import List, Set, Tuple, Dict
import math
from enum import Enum

class CoverageType(Enum):
    SB_COV = "state_boundary"  # 状态边界覆盖率
    KS_COV = "k_multisection_state"  # K-多段状态覆盖率
    AB_COV = "action_boundary"  # 动作边界覆盖率
    KA_COV = "k_multisection_action"  # K-多段动作覆盖率
    SACOV = "state_action"  # 状态-动作覆盖率
    SINGLE = "single"  # 单状态覆盖率
    LINK = "link"  # 状态转换覆盖率
    CROSS = "cross"  # 交叉状态覆盖率

class CoverageCalculator:
    def __init__(self, 
                 position_bounds: Tuple[float, float] = (-10, 10),
                 attitude_bounds: Tuple[float, float] = (-math.pi, math.pi),
                 position_grid_size: int = 20,
                 attitude_grid_size: int = 12,
                 k_sections: int = 5):
        """
        初始化覆盖率计算器
        
        Args:
            position_bounds: 位置范围 (min, max)
            attitude_bounds: 姿态范围 (min, max)
            position_grid_size: 位置空间划分的网格数
            attitude_grid_size: 姿态空间划分的网格数
            k_sections: K-多段划分的段数
        """
        self.position_bounds = position_bounds
        self.attitude_bounds = attitude_bounds
        self.position_grid_size = position_grid_size
        self.attitude_grid_size = attitude_grid_size
        self.k_sections = k_sections
        
        # 创建状态空间网格
        self.position_grid = np.linspace(position_bounds[0], position_bounds[1], position_grid_size)
        self.attitude_grid = np.linspace(attitude_bounds[0], attitude_bounds[1], attitude_grid_size)
        
        # 记录已访问的状态和动作
        self.visited_states: Set[Tuple[int, int, int, int, int, int]] = set()
        self.visited_actions: Set[int] = set()
        self.state_action_pairs: Set[Tuple[Tuple[int, int, int, int, int, int], int]] = set()
        
        # 记录边界值
        self.state_boundaries = {
            'high': [float('-inf')] * 6,  # 6个状态维度
            'low': [float('inf')] * 6
        }
        self.action_boundaries = {
            'high': float('-inf'),
            'low': float('inf')
        }
        
        # 训练期间的边界值（示例值，需要根据实际情况设置）
        self.training_state_boundaries = {
            'high': [2.39, 2.39, 2.39, 2.39, 2.39, 2.39],
            'low': [-2.38, -2.38, -2.38, -2.38, -2.38, -2.38]
        }
        self.training_action_boundaries = {
            'high': 1,
            'low': 0
        }
        
        # 初始化各种覆盖率的缓存
        self._coverage_cache: Dict[CoverageType, float] = {}
    
    def discretize_state(self, 
                        position: Tuple[float, float, float],
                        attitude: Tuple[float, float, float]) -> Tuple[int, int, int, int, int, int]:
        """
        将连续状态离散化为网格索引
        
        Args:
            position: (x, y, z) 位置
            attitude: (roll, pitch, yaw) 姿态
            
        Returns:
            离散化后的状态索引
        """
        x_idx = np.digitize(position[0], self.position_grid) - 1
        y_idx = np.digitize(position[1], self.position_grid) - 1
        z_idx = np.digitize(position[2], self.position_grid) - 1
        
        roll_idx = np.digitize(attitude[0], self.attitude_grid) - 1
        pitch_idx = np.digitize(attitude[1], self.attitude_grid) - 1
        yaw_idx = np.digitize(attitude[2], self.attitude_grid) - 1
        
        return (x_idx, y_idx, z_idx, roll_idx, pitch_idx, yaw_idx)
    
    def update_coverage(self, 
                       position: Tuple[float, float, float],
                       attitude: Tuple[float, float, float],
                       action: int) -> Dict[CoverageType, float]:
        """
        更新所有类型的覆盖率
        
        Args:
            position: 位置 (x, y, z)
            attitude: 姿态 (roll, pitch, yaw)
            action: 动作值
            
        Returns:
            包含所有覆盖率类型的字典
        """
        # 更新状态和动作
        state = self.discretize_state(position, attitude)
        self.visited_states.add(state)
        self.visited_actions.add(action)
        self.state_action_pairs.add((state, action))
        
        # 更新边界值
        for i, value in enumerate(position + attitude):
            self.state_boundaries['high'][i] = max(self.state_boundaries['high'][i], value)
            self.state_boundaries['low'][i] = min(self.state_boundaries['low'][i], value)
        
        self.action_boundaries['high'] = max(self.action_boundaries['high'], action)
        self.action_boundaries['low'] = min(self.action_boundaries['low'], action)
        
        # 计算所有类型的覆盖率
        coverages = {}
        for cov_type in CoverageType:
            coverages[cov_type] = self._calculate_coverage(cov_type)
        
        self._coverage_cache = coverages
        return coverages
    
    def _calculate_state_boundary_coverage(self) -> float:
        """
        计算状态边界覆盖率 (SBCov)
        使用平均值的定义
        """
        coverage_sum = 0
        for i in range(6):  # 6个状态维度
            test_range = self.state_boundaries['high'][i] - self.state_boundaries['low'][i]
            train_range = self.training_state_boundaries['high'][i] - self.training_state_boundaries['low'][i]
            if train_range > 0:
                coverage_sum += test_range / train_range
        return coverage_sum / 6
    
    def _calculate_k_multisection_state_coverage(self) -> float:
        """
        计算K-多段状态覆盖率 (KSCov)
        """
        covered_sections = set()
        for state in self.visited_states:
            section_indices = []
            for i, value in enumerate(state):
                # 将每个维度的值映射到对应的区间
                if i < 3:  # 位置维度
                    section = int((value - self.position_bounds[0]) / 
                                ((self.position_bounds[1] - self.position_bounds[0]) / self.k_sections))
                else:  # 姿态维度
                    section = int((value - self.attitude_bounds[0]) / 
                                ((self.attitude_bounds[1] - self.attitude_bounds[0]) / self.k_sections))
                section_indices.append(section)
            covered_sections.add(tuple(section_indices))
        
        total_sections = self.k_sections ** 6  # 6个状态维度
        return len(covered_sections) / total_sections
    
    def _calculate_action_boundary_coverage(self) -> float:
        """
        计算动作边界覆盖率 (ABCov)
        使用平均值的定义
        """
        test_range = self.action_boundaries['high'] - self.action_boundaries['low']
        train_range = self.training_action_boundaries['high'] - self.training_action_boundaries['low']
        return test_range / train_range if train_range > 0 else 0
    
    def _calculate_k_multisection_action_coverage(self) -> float:
        """
        计算K-多段动作覆盖率 (KACov)
        """
        covered_sections = set()
        for action in self.visited_actions:
            section = int((action - self.action_boundaries['low']) / 
                         ((self.action_boundaries['high'] - self.action_boundaries['low']) / self.k_sections))
            covered_sections.add(section)
        
        return len(covered_sections) / self.k_sections
    
    def _calculate_state_action_coverage(self) -> float:
        """
        计算状态-动作覆盖率 (SACov)
        """
        covered_pairs = len(self.state_action_pairs)
        total_pairs = self.k_sections ** 6 * self.k_sections  # 状态空间划分 * 动作空间划分
        return covered_pairs / total_pairs
    
    def _calculate_coverage(self, cov_type: CoverageType) -> float:
        """
        计算指定类型的覆盖率
        """
        if cov_type == CoverageType.SB_COV:
            return self._calculate_state_boundary_coverage()
        elif cov_type == CoverageType.KS_COV:
            return self._calculate_k_multisection_state_coverage()
        elif cov_type == CoverageType.AB_COV:
            return self._calculate_action_boundary_coverage()
        elif cov_type == CoverageType.KA_COV:
            return self._calculate_k_multisection_action_coverage()
        elif cov_type == CoverageType.SACOV:
            return self._calculate_state_action_coverage()
        elif cov_type == CoverageType.SINGLE:
            return len(self.visited_states) / self.total_states
        elif cov_type == CoverageType.LINK:
            return self._calculate_link_coverage()
        elif cov_type == CoverageType.CROSS:
            return self._calculate_cross_coverage()
        else:
            raise ValueError(f"Unknown coverage type: {cov_type}")
    
    def get_coverage(self, cov_type: CoverageType = None) -> float:
        """
        获取指定类型的覆盖率，如果没有指定类型则返回所有覆盖率
        """
        if cov_type is None:
            return self._coverage_cache
        return self._coverage_cache.get(cov_type, 0.0)
    
    def reset(self):
        """重置覆盖率计算器"""
        self.visited_states.clear()
        self.visited_actions.clear()
        self.state_action_pairs.clear()
        self.state_boundaries = {'high': [float('-inf')] * 6, 'low': [float('inf')] * 6}
        self.action_boundaries = {'high': float('-inf'), 'low': float('inf')}
        self._coverage_cache.clear() 