import numpy as np
from coverage_calculator import CoverageCalculator, CoverageType
import airsim

def main():
    # 创建AirSim客户端
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 创建覆盖率计算器实例
    calculator = CoverageCalculator(
        position_bounds=(-50, 50),  # 更大的位置范围，单位：米
        attitude_bounds=(-np.pi, np.pi),  # 姿态范围：[-π, π]
        position_grid_size=20,
        attitude_grid_size=12,
        k_sections=5
    )
    
    # 设置训练期间的边界值（根据实际训练数据调整）
    calculator.training_state_boundaries = {
        'high': [50, 50, 50, np.pi, np.pi, np.pi],  # x, y, z, roll, pitch, yaw
        'low': [-50, -50, -50, -np.pi, -np.pi, -np.pi]
    }
    calculator.training_action_boundaries = {
        'high': 1.0,  # 最大推力/速度比例
        'low': -1.0   # 最小推力/速度比例
    }

    # 模拟一些无人机飞行任务
    test_cases = [
        # 位置(x,y,z), 姿态(roll,pitch,yaw), 动作(vx,vy,vz,yaw_rate)
        ((0, 0, -5), (0, 0, 0), (0.5, 0, 0, 0)),      # 起飞并向前飞
        ((10, 0, -5), (0, 0, 0), (0, 0.5, 0, 0)),     # 向右飞
        ((10, 10, -5), (-0.1, 0.1, np.pi/4), (-0.5, 0, 0, 0)),  # 转向并返回
        ((0, 10, -5), (0, 0, np.pi/2), (0, -0.5, 0, 0)),  # 完成矩形轨迹
        ((0, 0, -5), (0, 0, 0), (0, 0, 0, 0)),        # 返回起点
    ]
    
    # 更新覆盖率并打印结果
    for i, (position, attitude, action) in enumerate(test_cases):
        print(f"\n航点 {i+1}:")
        print(f"位置 (x,y,z): {position}")
        print(f"姿态 (roll,pitch,yaw): {attitude}")
        print(f"控制输入 (vx,vy,vz,yaw_rate): {action}")
        
        # 计算动作值（将四维动作映射到标量）
        action_scalar = np.mean([abs(a) for a in action])
        
        # 更新所有类型的覆盖率
        coverages = calculator.update_coverage(position, attitude, action_scalar)
        
        # 打印各种覆盖率
        print("\n覆盖率计算结果:")
        print("-" * 50)
        for cov_type, value in coverages.items():
            print(f"{cov_type.value}: {value:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 