# 无人机状态空间覆盖率计算器

这个模块用于计算无人机在状态空间中的覆盖率。它通过将连续的状态空间离散化为网格，并记录已访问的状态来计算覆盖率。

## 功能特点

- 支持位置(x,y,z)和姿态(roll,pitch,yaw)的状态空间
- 可配置的状态空间范围和网格大小
- 实时计算和更新覆盖率
- 支持重置覆盖率计算

## 使用方法

1. 创建覆盖率计算器实例：

```python
from coverage_calculator import CoverageCalculator

calculator = CoverageCalculator(
    position_bounds=(-10, 10),  # 位置范围
    attitude_bounds=(-np.pi, np.pi),  # 姿态范围
    position_grid_size=20,  # 位置空间划分的网格数
    attitude_grid_size=12  # 姿态空间划分的网格数
)
```

2. 更新覆盖率：

```python
# 更新状态并获取覆盖率
coverage = calculator.update_coverage(
    position=(x, y, z),
    attitude=(roll, pitch, yaw)
)
```

3. 获取当前覆盖率：

```python
coverage = calculator.get_coverage()
```

4. 重置覆盖率计算：

```python
calculator.reset()
```

## 示例

运行示例脚本：

```bash
python example.py
```

## 注意事项

- 覆盖率计算基于离散化的状态空间，网格大小会影响计算精度
- 状态空间范围应根据实际无人机的工作范围来设置
- 较大的网格数会提高精度但会增加内存使用 