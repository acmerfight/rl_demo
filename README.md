# RL-Demo 强化学习演示

一个简洁明了的强化学习算法演示项目，用于展示和解释几种经典的强化学习算法原理及其Python实现。

## 项目概述

本项目旨在通过简单直观的环境和详细的代码注释，帮助初学者和爱好者理解强化学习的核心概念和算法实现。项目实现了三种经典的强化学习算法：

1. **Q-learning**：基于表格的值迭代方法，适用于离散状态和动作空间
2. **策略梯度（Policy Gradient）**：直接优化策略的方法，适用于离散动作空间
3. **连续策略梯度（Continuous Policy Gradient）**：适用于连续动作空间的策略梯度变体

每个算法都配有详细的中文注释和解释，帮助用户理解算法的工作原理和实现细节。

## 功能特点

- **直观的环境**：
  - 简单迷宫环境（离散状态和动作空间）
  - 二维目标寻找环境（连续状态和动作空间）

- **算法实现**：
  - 表格型 Q-learning 算法
  - REINFORCE 策略梯度算法
  - 连续动作空间的策略梯度算法

- **可视化功能**：
  - 环境状态的实时可视化
  - 智能体行为的直观展示
  - 训练过程的进度和性能统计

- **教学功能**：
  - 详细的算法解释函数
  - 算法更新步骤的分步解释
  - 丰富的中文注释和文档

## 安装指南

### 环境要求

- Python >= 3.13

### 安装步骤

1. 克隆仓库

```bash
git clone <仓库地址>
cd rl-demo
```

2. 创建虚拟环境并安装依赖

```bash
# 使用 venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 使用 uv 安装依赖
pip install uv
uv pip install .
```

## 使用方法

### Q-learning 算法演示

```bash
python q_learning.py
```

### 策略梯度算法演示（离散动作空间）

```bash
python policy_gradient.py
```

### 连续策略梯度算法演示

```bash
python continuous_policy_gradient.py
```

## 项目文件结构

- `q_learning.py` - Q-learning 算法实现与演示
- `policy_gradient.py` - 策略梯度算法实现与演示
- `continuous_policy_gradient.py` - 连续动作空间的策略梯度算法实现与演示
- `main.py` - 项目主入口文件

## 学习资源

通过本项目，您可以学习：

1. 强化学习的基本概念：状态、动作、奖励、策略、值函数等
2. Q-learning 算法的数学原理和实现方法
3. 策略梯度算法的理论基础和实现技巧
4. 连续动作空间问题的处理方法
5. 强化学习算法的调试和可视化技术

## 贡献指南

欢迎对本项目进行贡献！您可以通过以下方式参与：

- 提交 Bug 报告或功能建议
- 改进代码或文档
- 添加新的算法实现或环境

## 许可证

[待定]
