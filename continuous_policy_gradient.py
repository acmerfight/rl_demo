# 当使用权重矩阵[[-0.1, 0], [0, -0.1]]和偏置[0.8, 0.8]时，智能体必然沿直线到达目标(8,8)。这是因为：
# 数学原理
# 动作计算公式：[0.8-0.1x, 0.8-0.1y]
# 对任意位置(x,y)：
# 动作方向永远指向目标点(8,8)
# 这个方向就是从当前点到(8,8)的直线
# 简单证明
# 从(x,y)到(8,8)的直线方向向量应该是:
# [8-x, 8-y]
# 我们计算的动作是:
# [0.8-0.1x, 0.8-0.1y] = [0.8-0.1x, 0.8-0.1y] = [0.1(8-x), 0.1(8-y)]
# 这正是指向目标的直线方向，只是大小变成了原始方向的0.1倍！

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional


# 2D目标寻找环境 - 连续动作空间示例
class TargetFindingEnv:
    """
    一个简单的2D目标寻找环境，智能体需要移动到目标位置

    状态: [x, y] - 智能体的位置
    动作: [dx, dy] - 移动方向和力度（连续值）
    """

    def __init__(self, target_position: Tuple[float, float] = (8, 8), max_steps: int = 100) -> None:
        # 环境界限
        self.x_min: float = 0
        self.x_max: float = 10
        self.y_min: float = 0
        self.y_max: float = 10

        # 目标位置
        self.target_position: np.ndarray = np.array(target_position)

        # 智能体当前位置
        self.agent_position: Optional[np.ndarray] = None
        # 智能体上一步位置
        self.prev_position: Optional[np.ndarray] = None

        # 每个episode的最大步数
        self.max_steps: int = max_steps
        self.current_step: int = 0

        # 动作空间界限
        self.action_bound: float = 1.0  # 动作取值范围 [-1, 1]
        # 上一步的距离
        self.prev_distance: float = -np.inf

    def reset(self) -> np.ndarray:
        """重置环境到随机初始状态"""
        # 随机初始化智能体位置，避免在目标附近
        while True:
            self.agent_position = np.array(
                [
                    np.random.uniform(self.x_min, self.x_max),
                    np.random.uniform(self.y_min, self.y_max),
                ]
            )

            # 确保初始位置与目标有一定距离
            if np.linalg.norm(self.agent_position - self.target_position) > 3.0:
                break

        self.current_step = 0
        self.prev_position = self.agent_position.copy()  # 初始化前一位置为当前位置
        self.prev_distance = float(np.linalg.norm(self.agent_position - self.target_position))
        return self.agent_position.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """
        执行动作，返回新状态、奖励、是否完成和额外信息

        参数:
        - action: [dx, dy] 范围在[-1, 1]之间的连续值
        """
        self.current_step += 1

        # 确保动作是有效的numpy数组
        action = np.asarray(action, dtype=np.float64)
        # 检查是否有NaN值，如果有则替换为0
        if np.isnan(action).any():
            action = np.zeros_like(action)

        # 裁剪动作到有效范围
        action = np.clip(action, -self.action_bound, self.action_bound)

        # 更新智能体位置
        new_position: np.ndarray = self.agent_position + action

        # 确保智能体不超出边界
        new_position[0] = np.clip(new_position[0], self.x_min, self.x_max)
        new_position[1] = np.clip(new_position[1], self.y_min, self.y_max)

        self.agent_position = new_position
        self.prev_position = self.agent_position
        # 计算到目标的距离
        distance_to_target: float = float(np.linalg.norm(self.agent_position - self.target_position))

        done: bool = False
        reward: float = 0.0

        # --- 新的奖励设计开始 ---

        # 初始化奖励为0
        reward = 0.0

        # 1. 到达目标的奖励 - 给予较大正奖励, 并奖励更快到达
        if distance_to_target < 0.5:
            done = True
            # 基础奖励 + 步数奖励（步数越少奖励越高）
            # 这样智能体会尽量用更少的步数到达目标
            reward = 100.0 + 100 * (self.max_steps - self.current_step) / self.max_steps
        else:
            # 最后没有到达目标，根据与目标距离给予惩罚
            if self.current_step >= self.max_steps:
                done = True
                # 基础惩罚 + 距离惩罚（距离越远惩罚越大）
                # -50作为基础失败惩罚，距离作为额外惩罚因子
                # 环境的最大对角线距离为sqrt(10^2+10^2)≈14.14
                # 将距离归一化到[0,1]区间后乘以50作为额外惩罚
                normalized_distance = distance_to_target / 14.14
                reward = -50.0 - 50.0 * normalized_distance

        # 如果尝试超过边界，给以额外惩罚
        if (self.agent_position[0] <= self.x_min or 
            self.agent_position[0] >= self.x_max or 
            self.agent_position[1] <= self.y_min or 
            self.agent_position[1] >= self.y_max):
            reward -= 1.0
        
        # 如果position没有变化，给以额外惩罚 
        if np.allclose(self.agent_position, self.prev_position):
            reward -= 1.0
        # --- 新的奖励设计结束 ---

        self.prev_distance = distance_to_target
        info: Dict[str, float] = {
            "distance": distance_to_target,
            "steps": float(self.current_step),
        }
        return self.agent_position.copy(), reward, done, info

    def render(self, mode: str = "human", ax: Optional[Axes] = None, clear: bool = True) -> Axes:
        """可视化当前环境状态"""
        if ax is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

        if clear:
            ax.clear()

        # 设置边界
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect("equal")

        # 画出目标位置
        target = Circle(
            tuple(self.target_position),
            radius=0.5,
            color="green",
            alpha=0.7,
            label="Target",
        )
        ax.add_patch(target)

        # 画出智能体位置
        if self.agent_position is not None:
            agent = Circle(
                tuple(self.agent_position),
                radius=0.3,
                color="blue",
                alpha=0.7,
                label="Agent",
            )
            ax.add_patch(agent)

        # 添加标签
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Target Finding Environment")
        ax.legend()

        # 如果是交互模式，需要刷新画布
        if mode == "human":
            plt.pause(0.01)

        return ax


class ContinuousPolicyGradientAgent:
    """
    连续动作空间的策略梯度智能体

    使用高斯策略(正态分布)来表示连续动作
    策略参数包括均值(mean)和标准差(std)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.999,
    ) -> None:
        """
        初始化连续策略梯度智能体

        参数:
        - state_dim: 状态向量维度
        - action_dim: 动作向量维度
        - learning_rate: 学习率
        - gamma: 折扣因子
        """
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.learning_rate: float = learning_rate
        self.gamma: float = gamma

        # 初始化策略参数 - 使用简单线性模型
        # 对于连续动作，我们需要学习两组参数:
        # 1. 动作均值的参数 (确定动作的中心)
        # 2. 动作标准差参数 (确定探索的程度)

        # 均值网络参数: state -> action_mean - 使用较小的初始值以防止大输出
        self.mean_weights: np.ndarray = np.random.randn(state_dim, action_dim) * 0.01
        self.mean_bias: np.ndarray = np.zeros(action_dim)

        # 标准差参数 - 使用log_std以防止负值，初始为0对应标准差为1.0
        # 正态分布是一种非常重要的概率分布，它有一个基本特性被称为"68-95-99.7规则"：
        # 约68%的数据点落在[μ-σ, μ+σ]范围内
        # 约95%的数据点落在[μ-2σ, μ+2σ]范围内
        # 约99.7%的数据点落在[μ-3σ, μ+3σ]范围内
        # action_mean是均值μ
        # std是标准差σ，初始值约为0.6065
        # 标准差 = e^(log_std) = e^(-0.5) ≈ 0.6065
        # 环境的动作空间限制在[-1, 1]范围内
        # 标准差0.6意味着大部分动作会限制在策略均值的±0.6范围内
        # 有时会探索到±1.2范围（约占95%），但很少超过这个范围
        # 初始0.6允许足够探索来发现有效策略
        # 随着学习进行，算法可以自动降低标准差，专注于利用
        self.log_std: np.ndarray = np.zeros(action_dim) - 0.5  # 初始标准差大约为0.6065

        # 轨迹记录
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []

        # 训练历史
        self.episode_rewards: List[float] = []

    def compute_action_mean(self, state: np.ndarray) -> np.ndarray:
        """
        根据状态计算动作均值

        在连续策略梯度算法中，动作均值代表智能体在当前状态下最倾向采取的动作。
        它是高斯策略（正态分布）的中心点，表示最优的动作估计。
        当不进行探索时，智能体会直接使用这个均值作为最终动作。

        计算过程：使用线性变换（状态乘以权重加上偏置）得到动作均值。

        参数:
        - state: 当前环境状态，如智能体的位置坐标

        返回:
        - mean: 动作均值，表示最优估计的动作
        """
        # 确保状态是正确的numpy数组格式
        state = np.asarray(state, dtype=np.float64)

        # 计算动作均值：线性变换（状态 × 权重 + 偏置）
        # 这类似于：根据当前位置(state)结合学到的经验(weights)，计算出最佳移动方向
        mean: np.ndarray = np.dot(state, self.mean_weights) + self.mean_bias

        # 裁剪均值防止过大导致数值不稳定
        # 限制动作均值在合理范围内，就像方向盘只能转到一定角度
        mean = np.clip(mean, -3.0, 3.0)

        return mean

    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        根据策略选择动作

        参数:
        - state: 当前状态
        - explore: 是否进行探索

        返回:
        - action: 选择的动作
        """
        # 计算动作均值
        action_mean: np.ndarray = self.compute_action_mean(state)

        if explore:
            # 根据均值和标准差采样动作，同时防止标准差过小或过大
            # 下限 -5.0：e^(-5.0) ≈ 0.0067 这接近于确定性策略（几乎没有探索）
            # 防止标准差变得过小导致：过早收敛到次优策略，数值不稳定（除以接近零的方差）
            # 上限 2：e^(2) ≈ 7.39 这是较大的探索度，防止标准差变得过大导致：完全随机的行为，学习过程不稳定，大的方差导致梯度爆炸
            # 这个范围允许算法：在学习初期保持较高探索度，随着学习进行逐渐降低探索度，在收敛时达到较低的探索度，在需要时重新增加探索（如遇到新情况）
            # 数值稳定性考虑：过小的标准差会导致数值问题（如除以接近零的数），过大的标准差会导致梯度爆炸，该范围提供了良好的数值特性
            # 这种裁剪机制确保了探索-利用的平衡能够在合理范围内自动调整，是强化学习算法中的常见实践。
            std: np.ndarray = np.exp(np.clip(self.log_std, -5.0, 2))  # 限制标准差范围
            # 生成随机噪声
            noise: np.ndarray = np.random.randn(self.action_dim)
            # 将噪声乘以标准差并加到均值上，得到最终的动作
            # 这等同于从N(action_mean, std²)的正态分布中采样
            action: np.ndarray = action_mean + noise * std
        else:
            # 测试模式，直接使用均值作为动作
            action = action_mean

        # 处理潜在的无效值，保留方向但限制幅度
        # 先替换所有NaN和Inf为0
        action = np.nan_to_num(action, nan=0.0, posinf=None, neginf=None)

        return action

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """存储一步轨迹"""
        # 确保存储的值是有效的
        if not np.any(np.isnan(state)) and not np.any(np.isnan(action)) and not np.isnan(reward):
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    def compute_returns(self) -> np.ndarray:
        """计算每一步的回报(折扣累积奖励)"""
        if not self.rewards:  # 检查是否有奖励
            return np.array([])

        returns: List[float] = []
        G: float = 0.0  # 累积回报

        # 从后向前计算回报,回报是从当前时间步到序列结束的所有奖励的累积和（考虑折扣因子）。
        # 这意味着每个时间步的回报是当前奖励加上未来奖励的折扣累积和。
        # 从序列末尾开始向前计算：
        # 最后一步的回报 = 最后一步的奖励
        # 倒数第二步的回报 = 倒数第二步的奖励 + γ×最后一步的回报
        # 依此类推...
        # 这样每一步的回报G反映了该步骤对未来总奖励的贡献。
        for r in reversed(self.rewards):
            # 从后向前计算回报,回报是从当前时间步到序列结束的所有奖励的累积和（考虑折扣因子）。
            # 这意味着每个时间步的回报是当前奖励加上未来奖励的折扣累积和。
            # 从序列末尾开始向前计算：
            # 最后一步的回报 = 最后一步的奖励
            # 倒数第二步的回报 = 倒数第二步的奖励 + γ×最后一步的回报
            # 依此类推...
            # 这样每一步的回报G反映了该步骤对未来总奖励的贡献。
            G = float(r) + self.gamma * G
            returns.insert(0, G)

        # 归一化回报以稳定训练，但只在有足够样本时进行
        returns_array: np.ndarray = np.array(returns, dtype=np.float64)
        if len(returns_array) > 1:
            # 计算均值和标准差
            returns_mean: float = float(np.mean(returns_array))
            returns_std: float = float(np.std(returns_array))

            # 只有当标准差足够大时才归一化，防止除以很小的数
            if returns_std > 1e-6:
                returns_array = (returns_array - returns_mean) / returns_std

        return returns_array

    def update_policy(self) -> None:
        """
        ===== 策略梯度算法学习过程 =====
        
        整体流程图示:
        状态s → 预测动作均值μ → 添加随机性执行实际动作a → 获得回报G
                    ↑                                     |
                    |_____________________________________|
                              使用回报G更新策略
        
        这个函数实现了REINFORCE算法的核心：根据收集的经验调整策略参数
        通俗理解：就像分析过去的投篮经验，改进自己的投篮方式
        """
        # 如果没有收集到经验，直接返回
        if len(self.states) == 0:
            return

        # 计算回报 (未来累积奖励)
        # 图示: R1→R2→R3→...→Rn (单步奖励)
        #       ↓  ↓  ↓      ↓
        #      G1→G2→G3→...→Gn (累积回报)
        # 其中: G1 = R1 + γR2 + γ²R3 + ... (当前奖励加未来折扣奖励)
        returns: np.ndarray = self.compute_returns()
        if len(returns) == 0:
            return

        # 对于每一步经验，计算并应用梯度
        for t in range(len(self.states)):
            state: np.ndarray = self.states[t]  # 当时的状态 (如位置坐标)
            action: np.ndarray = self.actions[t]  # 实际执行的动作
            G: float = float(returns[t])  # 这步动作获得的总回报(好坏程度)

            # 根据当前策略重新计算"应该采取的动作"
            # 类比: "如果现在再面对同样情况，我会怎么做?"
            action_mean: np.ndarray = self.compute_action_mean(state)

            # 限制log_std的范围，防止探索过多或过少
            # 图示:  极小 ←——— 适中 ———→ 极大
            #      [-5.0]     [0]      [2.0]
            #       确定      适中     不确定  
            clipped_log_std: np.ndarray = np.clip(self.log_std, -5.0, 2)
            std: np.ndarray = np.exp(clipped_log_std)

            # 防止计算出现数值问题
            std = np.maximum(std, 1e-6)

            # 计算实际动作与预测动作的差异
            # 例: 预测向右2步，实际向右3步，差异为+1
            action_diff: np.ndarray = action - action_mean

            # 限制差异大小，避免过度调整
            action_diff = np.clip(action_diff, -5.0, 2)

            # 1. 更新均值网络参数
            variance: np.ndarray = std**2
            # 计算梯度方向：告诉我们应该如何调整策略
            # 简单理解：
            # - 如果实际动作比预测大，且效果好(G>0)→下次预测值变大
            # - 如果实际动作比预测小，且效果好(G>0)→下次预测值变小
            # - 如果效果不好(G<0)→则朝反方向调整
            mean_grad: np.ndarray = action_diff / variance

            # 根据回报大小自动调整学习率
            # 回报越大→学习率越小→避免过度调整
            lr_scaled: float = self.learning_rate * min(1.0, 1.0 / (np.abs(G) + 1.0))

            # 更新权重：负责将状态映射到动作
            # 权重决定"看到什么特征，执行什么动作"
            for i in range(self.state_dim):
                for j in range(self.action_dim):
                    weight_update: float = lr_scaled * G * state[i] * mean_grad[j]
                    weight_update = np.clip(weight_update, -0.1, 0.1)
                    self.mean_weights[i, j] += weight_update

            # 更新偏置：代表基本行为倾向
            # 偏置决定"不考虑状态时的默认行为"
            bias_update: np.ndarray = lr_scaled * G * mean_grad
            bias_update = np.clip(bias_update, -0.1, 0.1)
            self.mean_bias += bias_update

            # 2. 更新探索程度(标准差)
            # 简单理解：
            # - 如果探索带来好处(G>0)且超出预期→增加探索
            # - 如果探索带来好处(G>0)但在预期内→减少探索
            # - 如果探索带来坏处(G<0)→反向调整
            log_std_grad: np.ndarray = action_diff**2 / variance - 1.0

            # 限制梯度大小
            log_std_grad = np.clip(log_std_grad, -1.0, 1.0)

            # 更新标准差(用较小学习率)
            # 先学好"做什么"(均值)，再调整"变化多少"(方差)
            log_std_update: np.ndarray = lr_scaled * 0.5 * G * log_std_grad
            log_std_update = np.clip(log_std_update, -0.1, 0.1)
            self.log_std += log_std_update

            # 确保探索度在合理范围内
            self.log_std = np.clip(self.log_std, -5.0, 2)

        # 记录这次经历的总奖励
        episode_reward: float = float(sum(self.rewards))
        self.episode_rewards.append(episode_reward)

        # 清空历史记录，准备下一轮学习
        self.states = []
        self.actions = []
        self.rewards = []

    def show_policy(self, ax: Optional[Axes] = None) -> Axes:
        """可视化当前策略"""
        if ax is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

        # 创建网格来显示策略
        x: np.ndarray = np.linspace(0, 10, 11)
        y: np.ndarray = np.linspace(0, 10, 11)
        X, Y = np.meshgrid(x, y)

        # 计算每个点的动作均值
        U: np.ndarray = np.zeros_like(X)
        V: np.ndarray = np.zeros_like(Y)

        for i in range(len(x)):
            for j in range(len(y)):
                state: np.ndarray = np.array([X[j, i], Y[j, i]])
                action_mean: np.ndarray = self.compute_action_mean(state)
                if not np.isnan(action_mean).any():
                    U[j, i] = action_mean[0]
                    V[j, i] = action_mean[1]

        # 归一化
        magnitude: np.ndarray = np.sqrt(U**2 + V**2)
        max_magnitude: float = float(np.max(magnitude) if np.max(magnitude) > 0 else 1.0)
        if max_magnitude > 0:
            U = U / max_magnitude
            V = V / max_magnitude

        # 绘制动作向量场
        ax.quiver(X, Y, U, V, scale=30, color="red", alpha=0.7)

        # 添加标准差信息
        std_display: np.ndarray = np.exp(np.clip(self.log_std, -5.0, 2))
        ax.text(
            0.1,
            0.9,
            f"action std: {std_display}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        return ax


def explain_continuous_policy_gradient() -> None:
    """解释连续动作空间的策略梯度算法"""
    print("\n===== 连续动作空间策略梯度算法解释 =====")
    print("连续动作空间的策略梯度算法使用参数化的概率分布来表示策略。")
    print("常用高斯分布(正态分布)来表示连续动作。")

    print("\n基本概念:")
    print("1. 状态 s: 环境的当前状态，如智能体的位置")
    print("2. 动作 a: 连续值，如方向和力度")
    print("3. 策略 π(a|s;θ): 在状态s下选择动作a的概率密度函数")
    print("4. 回报 G: 从当前时刻到序列结束的累积折扣奖励")

    print("\n高斯策略表示:")
    print("π(a|s;θ) = N(μ(s;θ), σ)")
    print("- μ(s;θ): 动作均值，由状态和参数θ确定")
    print("- σ: 动作标准差，控制探索的程度")

    print("\n参数更新:")
    print("1. 均值网络参数: θ ← θ + α·G·∇θ log π(a|s;θ)")
    print("2. 标准差参数: σ ← σ + α·G·∇σ log π(a|s;θ)")

    print("\n算法特点:")
    print("1. 适用于连续动作空间")
    print("2. 通过调整探索度(标准差)自动控制探索-利用平衡")
    print("3. 以概率分布方式表示策略，更自然地处理不确定性")

    print("\n连续与离散动作空间的区别:")
    print("1. 离散: 使用softmax函数输出动作概率")
    print("2. 连续: 使用高斯分布(或其他连续分布)输出动作概率密度")
    print("3. 连续动作需要参数化均值和方差，离散动作只需要参数化各动作偏好值")
    print("=====================================")


def train(
    env: TargetFindingEnv,
    agent: ContinuousPolicyGradientAgent,
    episodes: int = 10000,
    render_freq: int = 50,
    render_delay: float = 0.01,
) -> List[float]:
    """训练智能体"""
    print("===== Start training continuous policy gradient agent =====")

    # 创建交互式绘图
    fig: Figure = plt.figure(figsize=(15, 10))

    # 环境可视化
    ax1: Axes = fig.add_subplot(2, 2, 1)

    # 训练进度可视化
    ax2: Axes = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Training Progress")

    # 探索度变化可视化
    ax3: Axes = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Action Standard Deviation")
    ax3.set_title("Exploration Rate")
    
    # 成功率可视化
    ax4: Axes = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Success Rate")
    ax4.set_ylim(0, 100)

    # 记录每个episode的标准差
    std_history = []

    # 记录每个episode是否成功到达目标
    success_history = []

    # 在更新策略后记录当前的标准差
    std_display = np.exp(np.clip(agent.log_std, -5.0, 2))
    std_history.append(std_display.copy())

    # 在训练循环中更新标准差图
    ax3.clear()
    ax3.plot(range(len(std_history)), [std[0] for std in std_history], "g-", label="dx")
    ax3.plot(range(len(std_history)), [std[1] for std in std_history], "m-", label="dy")
    ax3.legend()
    ax3.set_title("Exploration Rate")  # 重新设置标题

    episode_rewards: List[float] = []

    for episode in range(episodes):
        # 重置环境
        state: np.ndarray = env.reset()
        total_reward: float = 0.0
        done: bool = False
        episode_success: bool = False

        # 如果是可视化轮次，则绘制初始状态
        should_render: bool = episode % render_freq == 0

        while not done:
            # 选择动作
            action: np.ndarray = agent.get_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 检查是否成功到达目标
            if done and info["distance"] < 0.5:
                episode_success = True

            # 存储轨迹
            agent.store_transition(state, action, reward)

            # 累计奖励
            total_reward += reward

            # 更新状态
            state = next_state

            # 如果需要渲染
            if should_render:
                ax1 = env.render(ax=ax1)
                plt.pause(render_delay)

        # 更新策略
        agent.update_policy()

        # 记录这一轮的总奖励
        episode_rewards.append(total_reward)
        
        # 记录这一轮是否成功
        success_history.append(int(episode_success))

        # 记录当前的标准差
        std_display = np.exp(np.clip(agent.log_std, -5.0, 2))
        std_history.append(std_display.copy())

        # 更新训练进度图和标准差图
        if len(episode_rewards) > 1:
            ax2.clear()
            ax2.plot(episode_rewards, "b-", alpha=0.3, label="Raw")

            # 添加移动平均线
            window = min(50, len(episode_rewards))
            if window > 1:
                moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
                ax2.plot(range(window - 1, len(episode_rewards)), moving_avg, "r-", linewidth=2, label=f"{window}-ep Average")

            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Total Reward")
            ax2.set_title("Training Progress")
            ax2.legend(loc="upper left")

            # 更新标准差图
            ax3.clear()
            ax3.plot(range(len(std_history)), [std[0] for std in std_history], "g-", label="dx")
            ax3.plot(range(len(std_history)), [std[1] for std in std_history], "m-", label="dy")
            ax3.set_xlabel("Episode")
            ax3.set_ylabel("Action Standard Deviation")
            ax3.set_title("Exploration Rate")
            ax3.legend()
            
            # 更新成功率图
            ax4.clear()
            if len(success_history) >= render_freq:
                # 计算最近render_freq个episode的成功率
                success_rates = []
                for i in range(render_freq, len(success_history) + 1):
                    recent_success = success_history[i - render_freq:i]
                    success_rate = 100 * sum(recent_success) / render_freq
                    success_rates.append(success_rate)
                
                ax4.plot(range(render_freq - 1, len(success_history)), success_rates, "g-", linewidth=2)
                ax4.set_xlabel("Episode")
                ax4.set_ylabel("Success Rate (%)")
                ax4.set_title(f"Success Rate (last {render_freq} episodes)")
                ax4.set_ylim(0, 100)
                ax4.grid(True)

            plt.tight_layout()
            plt.pause(0.01)

        # 打印训练信息
        if should_render or episode == episodes - 1:
            # 计算最近 render_freq 个 episode 的平均奖励
            start_index = max(0, episode - render_freq + 1)
            recent_rewards = episode_rewards[start_index : episode + 1]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            # 计算最近 render_freq 个 episode 的成功率
            # 用与图表相同的计算方式，确保一致性
            if episode >= render_freq - 1:
                # 有足够的历史数据时，使用完整的render_freq个episodes
                recent_success = success_history[episode - render_freq + 1:episode + 1]
                success_rate = 100 * sum(recent_success) / render_freq
            else:
                # 训练初期，使用所有可用数据
                recent_success = success_history[:episode + 1]
                success_rate = 100 * sum(recent_success) / len(recent_success) if recent_success else 0.0
            
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.1f}%")

            # 在环境中显示策略
            env.render(ax=ax1, clear=False)
            ax1 = agent.show_policy(ax=ax1)
            plt.pause(0.5)  # 暂停一下，让用户看清楚

    print("===== 训练完成 =====")
    return episode_rewards


def test_agent(
    env: TargetFindingEnv,
    agent: ContinuousPolicyGradientAgent,
    episodes: int = 10000000,
    delay: float = 0.2,
) -> None:
    """测试已训练的智能体"""
    print("\n===== 测试已训练的智能体 =====")

    # 创建交互式绘图
    plt.figure(figsize=(8, 8))
    ax: Axes = plt.gca()

    total_rewards: List[float] = []
    success_count: int = 0

    for episode in range(episodes):
        state: np.ndarray = env.reset()
        total_reward: float = 0.0
        done: bool = False
        steps: int = 0
        episode_info: Optional[Dict[str, float]] = None  # 用于存储最后一步的info

        print(f"\nEpisode {episode + 1} start")
        ax = env.render(ax=ax)
        plt.pause(delay)

        while not done:
            # 测试时使用确定性策略
            action: np.ndarray = agent.get_action(state, explore=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_info = info  # 保存最新的info

            # 累计奖励
            total_reward += reward

            # 打印当前步骤信息
            print(f"Step {steps + 1}: Action={action}, Reward={reward:.2f}, " + f"Distance={info['distance']:.2f}")

            # 更新状态
            state = next_state
            steps += 1

            # 渲染
            ax = env.render(ax=ax)

            # 绘制当前的动作箭头
            arrow_len: float = float(np.linalg.norm(action))
            if arrow_len > 0 and env.agent_position is not None:
                arrow = Arrow(
                    env.agent_position[0],
                    env.agent_position[1],
                    action[0] * 0.8,
                    action[1] * 0.8,
                    width=0.1,
                    color="red",
                )
                ax.add_patch(arrow)

            plt.pause(delay)

        total_rewards.append(total_reward)

        # 使用保存的episode_info
        if episode_info is not None and episode_info.get("distance", float("inf")) < 0.5:
            success_count += 1
            print(f"成功! 总奖励: {total_reward:.2f}, 步数: {steps}")
        else:
            print(f"失败. 总奖励: {total_reward:.2f}, 步数: {steps}")

    success_rate: float = success_count / episodes * 100
    print(f"\n测试结果: 成功率 {success_rate:.1f}%, 平均奖励: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    # 解释连续策略梯度算法
    explain_continuous_policy_gradient()

    # 创建环境
    env: TargetFindingEnv = TargetFindingEnv()

    # 创建智能体
    agent: ContinuousPolicyGradientAgent = ContinuousPolicyGradientAgent(
        state_dim=2,  # [x, y]
        action_dim=2,  # [dx, dy]
        learning_rate=0.001,  # 降低学习率以配合新的奖励系统
        gamma=0.99,
    )

    # 训练智能体
    rewards: List[float] = train(env, agent, episodes=20000, render_freq=50)

    # 测试智能体
    test_agent(env, agent)

    # 显示最终学习到的策略
    plt.figure(figsize=(8, 8))
    ax: Axes = plt.gca()
    env.reset()  # 重置环境状态
    env.render(ax=ax)
    agent.show_policy(ax=ax)
    plt.title("final policy")
    plt.show()

    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(agent.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.show()
