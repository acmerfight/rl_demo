import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Tuple, Optional, Any, List, NamedTuple, Callable
import os
import gc
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn  # ADDED: PyTorch neural network module
import random
from datetime import datetime
import math  # ADDED: math for cosine schedule

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    专为五子棋设计的自定义卷积神经网络 (CNN) 特征提取器。
    适用于较小棋盘尺寸 (例如 10x10 或 15x15)，NatureCNN 在这种尺寸下可能会因过度下采样而出错。
    这个CNN结构比NatureCNN更温和，使用较小的卷积核、步幅和池化层，以避免维度过早减小到无效值。

    网络结构示例 (棋盘尺寸会影响中间及最终展平前的维度):
    - 输入: (N, n_input_channels, H, W)
    - Conv1 (32 filters, 3x3 kernel, stride 1, padding 1) -> ReLU -> MaxPool1 (2x2 kernel, stride 2)
      (例: 10x10 -> 5x5; 15x15 -> 7x7 after pool)
    - Conv2 (64 filters, 3x3 kernel, stride 1, padding 1) -> ReLU -> MaxPool2 (2x2 kernel, stride 2)
      (例: 5x5 -> 2x2; 7x7 -> 3x3 after pool)
    - Conv3 (64 filters, 3x3 kernel, stride 1, padding 1) -> ReLU
      (例: 2x2 stays 2x2; 3x3 stays 3x3)
    - Flatten
    - Linear: (展平后的特征数) -> features_dim
    - ReLU

    :param observation_space: 观察空间 (Gymnasium Space), 用于确定输入通道数和动态计算展平特征数。
    :param features_dim: 提取特征的数量 (例如, PPO策略通常使用 256 或 512)。
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # observation_space.shape 是 (channels, height, width)
        n_input_channels = observation_space.shape[0]

        self.cnn_layers = nn.Sequential(
            # 第一组卷积和池化
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 例: 10x10 -> 5x5; 15x15 -> 7x7
            # 第二组卷积和池化
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 例: 5x5 -> 2x2; 7x7 -> 3x3
            # 第三组卷积（通常不池化，以保留更多信息给全连接层）
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 对于10x10输入，此时特征图是 (64, 2, 2)
            # 对于15x15输入，此时特征图是 (64, 3, 3)
            nn.Flatten(),  # 将多维特征图展平为一维向量
        )

        # 为了使网络结构更具通用性，我们动态计算CNN部分输出的展平后的特征数量。
        # 这需要一个虚拟输入通过CNN层来确定其输出形状。
        with th.no_grad():  # 禁用梯度计算，因为这只是为了获取形状
            # observation_space.sample() 返回 numpy 数组, [None] 增加批次维度
            # th.as_tensor 转换为 PyTorch 张量，并确保类型为 float
            dummy_input_np = observation_space.sample()[None]
            dummy_input_th = th.as_tensor(dummy_input_np).float()
            n_flattened_features = self.cnn_layers(dummy_input_th).shape[1]

        # 全连接层 (Linear Layer)，将CNN提取的特征映射到最终的 features_dim
        self.linear_layers = nn.Sequential(
            nn.Linear(n_flattened_features, features_dim),  # 输入为展平后的特征数
            nn.ReLU(),  # 通常在特征提取的最后也加一个激活函数
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        定义数据通过网络的前向传播路径。

        :param observations: 批量的观察数据, 形状为 (N, C, H, W)，N是批量大小。
        :return: 提取的特征向量, 形状为 (N, features_dim)。
        """
        # 首先通过卷积层和池化层提取空间特征
        cnn_output = self.cnn_layers(observations)
        # 然后通过全连接层得到最终的特征向量
        return self.linear_layers(cnn_output)


class GomokuEnv:
    """
    五子棋环境

    状态(State): 棋盘状态
    动作(Action): 离散值，表示放置棋子的位置索引 (0 到 board_size*board_size-1)
    """

    def __init__(self, board_size: int = 15, agent_perspective: int = 1) -> None:
        """
        初始化五子棋环境

        参数:
        - board_size: 棋盘大小
        - agent_perspective: 从哪方视角计算奖励 (1 代表黑棋视角, -1 代表白棋视角)
        """
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 代表黑棋, -1 代表白棋
        self.done = False
        self.winner = 0
        self.last_move = None
        self.agent_perspective = agent_perspective  # 从哪方视角计算奖励

        # 用于可视化
        self.history = []

    def reset(self, agent_perspective: int = None) -> np.ndarray:
        """
        重置游戏环境

        参数:
        - agent_perspective: 可选，重置智能体视角
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # 黑棋先手
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.last_move = None
        self.history = []

        # 如果提供了新的视角，则更新
        if agent_perspective is not None:
            self.agent_perspective = agent_perspective

        return self._get_state()

    def set_agent_perspective(self, agent_perspective: int) -> None:
        """设置智能体视角 (1 或 -1)"""
        assert agent_perspective in [1, -1], "agent_perspective must be either 1 (black) or -1 (white)"
        self.agent_perspective = agent_perspective

    def _get_state(self) -> np.ndarray:
        """
        获取当前状态表示

        返回3通道状态:
        通道0: 黑棋位置 (黑棋为1, 空位为0)
        通道1: 白棋位置 (白棋为1, 空位为0)
        通道2: 当前玩家 (黑棋回合全为1, 白棋回合全为0)
        """
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == 1).astype(np.float32)
        state[1] = (self.board == -1).astype(np.float32)
        state[2] = (
            np.ones((self.board_size, self.board_size))
            if self.current_player == 1
            else np.zeros((self.board_size, self.board_size))
        )
        return state

    def get_valid_moves(self) -> np.ndarray:
        """获取所有有效的落子位置（一维索引）"""
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    valid_moves.append(i * self.board_size + j)
        return np.array(valid_moves)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步

        参数:
        - action: 整数，表示棋盘位置的一维索引 (0 到 board_size*board_size-1)

        返回:
        - state: 新状态
        - reward: 奖励（从agent_perspective视角计算）
        - done: 游戏是否结束
        - info: 额外信息

        异常:
        - ValueError: 如果动作无效
        - RuntimeError: 如果游戏状态不一致（例如没有有效动作但游戏未结束）
        """
        if self.done:
            return self._get_state(), 0.0, True, {"winner": self.winner}

        # 获取有效的动作列表
        valid_moves = self.get_valid_moves()

        # 检查游戏状态一致性
        if len(valid_moves) == 0:
            raise RuntimeError(
                "No valid moves available but game is not marked as done. This indicates a bug in the game logic."
            )

        # 验证动作有效性
        if action not in valid_moves:
            raise ValueError(f"Invalid action {action}. Valid actions are {valid_moves}.")

        # 转换为二维坐标
        x = action // self.board_size
        y = action % self.board_size

        # 执行动作
        self.board[x, y] = self.current_player
        self.last_move = (x, y)
        self.history.append((x, y, self.current_player))

        # 检查游戏是否结束
        win = self._check_win(x, y)
        draw = np.count_nonzero(self.board) == self.board_size**2

        # 从智能体视角计算奖励
        reward = 0.0
        info = {"winner": 0, "position": (x, y)}

        if win:
            self.done = True
            self.winner = self.current_player
            info["winner"] = self.winner

            # --- 计算胜负奖励 (鼓励尽快获胜) ---
            num_steps = len(self.history)
            min_win_steps = 9  # 理论上最快获胜所需的步数
            max_steps = self.board_size * self.board_size  # 最大可能步数

            # 定义奖励参数
            fastest_win_reward = 2.0  # 最快获胜时的奖励
            slowest_win_reward = 1.0  # 最慢获胜时的奖励
            # 失败惩罚与最慢获胜奖励的绝对值一致
            loss_penalty = -slowest_win_reward

            # 计算获胜步数相对于最快和最慢情况的归一化进度 [0, 1]
            # 0 表示接近最快获胜, 1 表示接近最慢获胜
            step_range = max_steps - min_win_steps
            assert num_steps >= min_win_steps, (
                "when win, num_steps must be greater than or equal to min_win_steps = %d" % min_win_steps
            )
            progress_to_slowest = (num_steps - min_win_steps) / step_range

            # 线性插值计算获胜奖励：从 fastest_win_reward 向 slowest_win_reward 递减
            win_reward_magnitude = fastest_win_reward - progress_to_slowest * (fastest_win_reward - slowest_win_reward)

            # 根据智能体视角分配最终奖励
            reward = win_reward_magnitude if self.current_player == self.agent_perspective else loss_penalty

        elif draw:
            self.done = True
            # 平局奖励（稍微负一点，鼓励智能体追求胜利）
            reward = -0.1

        else:
            # 游戏继续, 切换玩家
            self.current_player *= -1

        return self._get_state(), reward, self.done, info

    def _check_win(self, x: int, y: int) -> bool:
        """
        检查最后一步是否导致获胜

        检查四个方向：水平、垂直、主对角线、副对角线
        如果任意方向有连续五个相同颜色的棋子，则获胜
        """
        player = self.board[x, y]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、主对角线、副对角线

        for dx, dy in directions:
            count = 1  # 当前位置已计入

            # 检查正方向
            for i in range(1, 5):
                nx, ny = x + dx * i, y + dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    if count >= 5:  # 一旦找到5个连珠就提前退出
                        return True
                else:
                    break

            # 检查负方向
            for i in range(1, 5):
                nx, ny = x - dx * i, y - dy * i
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                    if count >= 5:  # 一旦找到5个连珠就提前退出
                        return True
                else:
                    break

        return False

    def render(self, ax: Optional[Axes] = None, clear: bool = True) -> Axes:
        """
        可视化棋盘状态

        参数:
        - ax: matplotlib的轴对象，如果为None则创建新的
        - clear: 是否清除现有绘图

        返回:
        - ax: 更新后的matplotlib轴对象
        """
        if ax is None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()

        if clear:
            ax.clear()

        # 绘制棋盘
        ax.set_xlim(-1, self.board_size)
        ax.set_ylim(-1, self.board_size)
        ax.set_aspect("equal")

        # 绘制网格线
        for i in range(self.board_size):
            ax.axhline(i, color="black", linewidth=1)
            ax.axvline(i, color="black", linewidth=1)

        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:  # 黑棋
                    ax.plot(i, j, "o", markersize=20, markerfacecolor="black", markeredgecolor="black")
                elif self.board[i, j] == -1:  # 白棋
                    ax.plot(i, j, "o", markersize=20, markerfacecolor="white", markeredgecolor="black")

        # 标记最后一步
        if self.last_move:
            x, y = self.last_move
            ax.plot(x, y, "x", markersize=8, markeredgewidth=3, color="red" if self.board[x, y] == 1 else "blue")

        # 设置网格标签
        ax.set_xticks(range(self.board_size))
        ax.set_yticks(range(self.board_size))
        ax.set_xticklabels([chr(65 + i) for i in range(self.board_size)])  # A, B, C, ...
        ax.set_yticklabels(range(1, self.board_size + 1))

        ax.set_title("Gomoku")
        plt.pause(0.01)

        return ax


# 以下为基于SB3、Gymnasium和Maskable PPO的自我博弈训练代码

# 假设模型类型，如果知道具体类型可以替换 Any
StableBaselinesModel = Any


def cosine_decay_lr(initial_lr: float, min_lr: float = 1e-7) -> Callable[[float], float]:
    """
    创建一个余弦衰减学习率调度函数。

    参数:
    - initial_lr: 初始最大学习率。
    - min_lr: 最小学习率。

    返回:
    - 一个函数，该函数接受 'progress_remaining' (从1.0到0.0)作为输入，
      并返回当前的学习率。
    """

    def schedule(progress_remaining: float) -> float:
        current_progress = 1.0 - progress_remaining
        # 使用正确的余弦衰减公式
        return initial_lr * (math.cos(math.pi / 2 * current_progress)) + min_lr * (1 - math.cos(math.pi / 2 * current_progress))

    return schedule


def cosine_decay_clip_range(initial_clip: float, final_clip: float) -> Callable[[float], float]:
    """
    创建一个余弦衰减调度函数，用于PPO的clip_range。

    参数:
    - initial_clip: 初始(最大)clip_range。
    - final_clip: 最终(最小)clip_range。

    返回:
    - 一个函数，接受'progress_remaining'(从1.0到0.0)作为输入，
      并返回当前的clip_range。
    """
    if final_clip > initial_clip:
        raise ValueError("final_clip应小于或等于initial_clip")

    def schedule(progress_remaining: float) -> float:
        current_progress = 1.0 - progress_remaining
        # 使用正确的余弦衰减公式
        return initial_clip * (math.cos(math.pi / 2 * current_progress)) + final_clip * (
            1 - math.cos(math.pi / 2 * current_progress)
        )

    return schedule


class ModelInfo(NamedTuple):
    """存储模型路径及其元数据的结构体"""

    model: str  # 存储模型的文件路径而不是模型对象
    name: str
    iteration: Any
    win_rate: Optional[float]


class ModelPoolManager:
    """
    模型池管理器，用于管理历史模型。
    在内存中保存模型快照，根据胜率移除和采样。
    """

    def __init__(self, max_models: int = 100):
        """
        初始化模型池管理器

        参数:
        - max_models: 池中最大模型数量 (必须是正整数)
        """
        self.max_models = max_models
        # 使用列表存储 ModelInfo 对象
        self.models: List[ModelInfo] = []

    def add_model(self, model: StableBaselinesModel, iteration: Any, win_rate: float) -> str:
        """
        添加新模型到模型池。如果池已满，则移除胜率最低的模型。

        参数:
        - model: 要添加的模型对象 (将使用save/load代替深拷贝)
        - iteration: 当前迭代标识
        - win_rate: 模型的胜率

        返回:
        - 生成的模型名称
        """
        # 生成唯一的模型名称和临时文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_iter_{iteration}_{timestamp}"
        temp_path = os.path.join("temp_models", f"{model_name}.zip")

        # 确保临时目录存在
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        # 保存模型到临时文件
        model.save(temp_path)

        # 创建模型信息对象，存储路径而不是模型对象
        new_model_info = ModelInfo(model=temp_path, name=model_name, iteration=iteration, win_rate=win_rate)

        # 添加到内存池
        self.models.append(new_model_info)
        print(f"已将模型添加到内存模型池: {new_model_info.name}, Win Rate: {new_model_info.win_rate}")

        # 如果模型数量超过上限，移除胜率最低的模型
        self._evict_lowest_win_rate_model_if_needed()

        return model_name

    def _evict_lowest_win_rate_model_if_needed(self):
        """如果模型池已满，则查找并移除胜率最低的模型。"""
        if len(self.models) > self.max_models:
            # 使用 min 和 lambda 函数简洁地查找胜率最低的模型及其索引
            min_index, _ = min(enumerate(self.models), key=lambda item: item[1].win_rate)
            # 移除模型信息
            removed_info = self.models.pop(min_index)

            # 删除对应的模型文件
            try:
                if os.path.exists(removed_info.model):
                    os.remove(removed_info.model)
            except Exception as e:
                print(f"删除模型文件时发生错误: {e}")

            print(f"模型池已满，已移除胜率最低的模型: {removed_info.name} (Win Rate: {removed_info.win_rate})")

    def sample_opponent_model(self) -> str:
        """
        从模型池中根据胜率加权随机选择一个对手模型的路径。
        胜率越高的模型被选中的概率越大。

        返回:
        - 选中的模型文件路径，如果池为空则返回 None
        """
        if not self.models:
            print("模型池为空，无法采样对手。")
            return None

        # 计算权重：胜率 + 基础权重 (epsilon)
        # 基础权重确保所有模型（包括胜率为0或None的模型）都有机会被选中
        base_weight = 0.2
        weights = [info.win_rate + base_weight for info in self.models]
        # 执行加权随机选择
        chosen_index = random.choices(range(len(self.models)), weights=weights, k=1)[0]
        # 获取选中的模型及其信息用于日志记录
        chosen_model_info = self.models[chosen_index]
        model_path = chosen_model_info.model  # Path to the model
        model_name = chosen_model_info.name
        chosen_win_rate = chosen_model_info.win_rate  # 用于日志的原始胜率

        print(
            f"从内存池中根据胜率加权选择对手模型路径: {model_name} (Path: {model_path}, Win Rate: {chosen_win_rate}, Weight: {weights[chosen_index]:.3f})"
        )
        return model_path  # MODIFIED: Return the path

    def get_latest_model(self) -> Optional[StableBaselinesModel]:
        """获取最新添加的模型对象"""
        if not self.models:
            return None
        # 从最新模型的路径加载模型
        latest_model_path = self.models[-1].model
        return MaskablePPO.load(latest_model_path)

    def get_model_count(self) -> int:
        """获取模型池中的模型数量"""
        return len(self.models)


class GomokuGymEnv(gym.Env):
    """
    五子棋环境的Gymnasium包装器，支持真正的自我博弈
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, board_size: int = 15, opponent_model=None, opponent_deterministic: bool = False):
        """
        初始化Gymnasium环境

        参数:
        - board_size: 棋盘大小
        - opponent_model: 对手模型对象
        - opponent_deterministic: 对手模型是否以确定性方式行动
        """
        super().__init__()

        # 创建原始环境
        self.env = GomokuEnv(board_size=board_size)
        self.board_size = board_size
        self.action_space = spaces.Discrete(board_size * board_size)

        # 设置观察空间为3通道的二维状态 [黑棋位置, 白棋位置, 当前玩家]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3, board_size, board_size), dtype=np.float32)

        # 对手模型
        self.opponent_model = opponent_model
        self.use_random_opponent = True if opponent_model is None else False
        self.opponent_deterministic = opponent_deterministic

        if self.opponent_model is not None:
            print(f"使用提供的对手模型 (Deterministic: {self.opponent_deterministic})")
        else:
            print("使用随机策略作为对手")

        # 用于保存当前游戏信息
        self.current_state = None
        self.render_ax = None
        self.agent_player_id = 1  # 智能体扮演的角色 (1 黑棋, -1 白棋), 会在reset时随机化

    def set_opponent_model(self, model_path: str):
        """
        设置对手模型

        参数:
        - model_path: 模型文件路径 (字符串)
        """
        if model_path and os.path.exists(model_path):
            try:
                # 释放旧模型内存
                if self.opponent_model is not None:
                    del self.opponent_model
                    gc.collect()

                self.opponent_model = MaskablePPO.load(model_path)
                self.use_random_opponent = False
                print(f"环境 {os.getpid()} 已从路径 {model_path} 更新对手模型")
            except Exception as e:
                print(f"环境 {os.getpid()} 从路径 {model_path} 加载模型失败: {e}. 将使用随机对手。")
                self.opponent_model = None
                self.use_random_opponent = True
        else:
            print(f"环境 {os.getpid()} 收到无效模型路径 '{model_path}' 或文件不存在。将使用随机对手。")
            self.opponent_model = None
            self.use_random_opponent = True

    def _random_opponent_action(self):
        """随机对手策略"""
        valid_moves = self.env.get_valid_moves()
        if len(valid_moves) > 0:
            return np.random.choice(valid_moves)
        return 0  # 理论上不会发生，因为如果没有有效动作游戏应该已经结束

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)  # 用于决定智能体角色

        # 随机决定智能体是扮演黑棋 (1) 还是白棋 (-1)
        self.agent_player_id = np.random.choice([1, -1])

        # 重置内部环境 (GomokuEnv.reset() 会将 current_player 设为 1)
        _ = self.env.reset()
        # 设置GomokuEnv从哪个视角计算奖励
        self.env.set_agent_perspective(self.agent_player_id)

        # 信息字典 (不直接传递 agent_player_id，回调会通过 get_attr 获取)
        returned_info = {}

        # 如果智能体扮演白棋 (-1)，则对手 (黑棋) 先手
        if self.agent_player_id == -1:
            # 此时 self.env.current_player 应该是 1 (黑棋)
            assert self.env.current_player == 1, "If agent is White, opponent (Black) should be current player."

            opponent_action_mask = self.action_mask()  # 对手 (黑棋) 的动作掩码
            opponent_action = None

            # 获取对手 (黑棋) 的初始观察状态
            initial_board_state_for_opponent = self.env._get_state()

            if not self.use_random_opponent and self.opponent_model is not None:
                opponent_action, _ = self.opponent_model.predict(
                    initial_board_state_for_opponent,
                    action_masks=opponent_action_mask,
                    deterministic=self.opponent_deterministic,
                )
            else:
                opponent_action = self._random_opponent_action()

            # 执行对手 (黑棋) 的第一步
            # next_state_after_opponent 是智能体 (白棋) 将观察到的第一个状态
            # 奖励和结束状态与此步相关，但主要用于设置棋盘
            next_state_after_opponent, _, done_after_opponent_move, _ = self.env.step(opponent_action)
            self.current_state = next_state_after_opponent

            # 五子棋第一步不可能结束游戏
            assert not done_after_opponent_move, "Game should not end after opponent's first move in reset."

        else:  # 智能体扮演黑棋 (1)
            # 智能体先手，观察空棋盘
            self.current_state = self.env._get_state()

        return self.current_state, returned_info

    def step(self, action):
        """
        执行一步动作

        在真正的自我博弈中，智能体只扮演一方，对手是历史版本模型
        """
        # 获取有效动作掩码
        action_mask = self.action_mask()

        # 验证动作是否有效
        if not action_mask[action]:
            # 如果动作无效，则随机选择一个有效动作
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                # 理论上不应该出现这种情况
                raise ValueError("无有效动作可选")

        # 执行动作
        next_state, reward, done, info = self.env.step(action)
        self.current_state = next_state

        # 如果游戏未结束，则由对手走下一步
        if not done:
            opponent_action = None

            # 使用模型对手或随机对手
            if not self.use_random_opponent and self.opponent_model is not None:
                # 获取当前状态的动作掩码
                opponent_mask = self.action_mask()

                # 使用对手模型选择动作
                opponent_action, _ = self.opponent_model.predict(
                    self.current_state, action_masks=opponent_mask, deterministic=self.opponent_deterministic
                )
            else:
                # 使用随机策略
                opponent_action = self._random_opponent_action()

            # 执行对手动作
            next_state, opponent_reward, done, info = self.env.step(opponent_action)
            self.current_state = next_state

            # 从智能体角度计算奖励（无需取反，opponent_reward已是智能体视角）
            reward = opponent_reward

        # 指定截断状态（gymnasium API要求）
        truncated = False

        return self.current_state, reward, done, truncated, info

    def action_mask(self) -> np.ndarray:
        """
        获取动作掩码，指示哪些动作是有效的

        返回:
        - 长度为board_size*board_size的布尔数组，True表示该位置可落子
        """
        valid_moves = self.env.get_valid_moves()
        mask = np.zeros(self.board_size * self.board_size, dtype=bool)
        mask[valid_moves] = True
        return mask

    def render(self, mode="human"):
        """渲染环境"""
        if mode == "human":
            self.render_ax = self.env.render(ax=self.render_ax)
            return None
        else:
            # 保存图像并返回RGB数组
            self.render_ax = self.env.render(ax=self.render_ax)
            return np.zeros((self.board_size * 30, self.board_size * 30, 3), dtype=np.uint8)

    def close(self):
        """关闭环境"""
        if self.render_ax is not None:
            plt.close(self.render_ax.figure)
            self.render_ax = None


def gomoku_mask_fn(env: GomokuGymEnv) -> np.ndarray:
    """
    获取动作掩码的函数，用于 MaskablePPO

    参数:
    - env: GomokuGymEnv实例

    返回:
    - 动作掩码数组
    """
    return env.action_mask()


class ModelPoolUpdateCallback(BaseCallback):
    """
    回调函数，用于定期更新模型池并记录胜率
    """

    def __init__(self, model_pool: ModelPoolManager, update_freq: int = 10000, verbose: int = 0):
        """
        初始化回调函数

        参数:
        - model_pool: 模型池管理器
        - update_freq: 更新频率（步数）
        - verbose: 详细程度
        """
        super().__init__(verbose)
        self.model_pool = model_pool
        self.update_freq = update_freq
        self.iteration = 0
        # 初始化胜率统计
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def _on_step(self) -> bool:
        """每步调用"""
        # 检查是否有环境完成
        for i, done in enumerate(self.locals["dones"]):
            if done:
                # 获取该环境的信息字典
                info = self.locals["infos"][i]
                if "winner" in info:
                    winner = info["winner"]
                    # 获取当前环境的智能体角色
                    # ModelPoolUpdateCallback.training_env 是 VecEnv 实例
                    agent_plays_as = self.training_env.get_attr("agent_player_id", indices=[i])[0]

                    # 根据智能体角色判断胜负
                    if winner == agent_plays_as:
                        self.wins += 1
                    elif winner == -agent_plays_as:  # 对手获胜
                        self.losses += 1
                    elif winner == 0:  # 平局
                        self.draws += 1
                    # else: winner is something unexpected or not set properly (e.g. if game ended due to error)

        # 检查是否达到更新频率
        if self.n_calls > 0 and self.n_calls % self.update_freq == 0:
            # 计算胜率
            total_games = self.wins + self.losses + self.draws
            win_rate = self.wins / total_games if total_games > 0 else 0.0

            if self.verbose > 0:
                print(f"\nSteps: {self.n_calls}, Updating model pool.")
                print(
                    f"Stats since last update: Wins={self.wins}, Losses={self.losses}, Draws={self.draws}, Total={total_games}"
                )
                print(f"Calculated Win Rate: {win_rate:.3f}")

            # 添加当前模型到模型池，并传入胜率
            self.iteration += 1
            self.model_pool.add_model(self.model, self.iteration, win_rate=win_rate)

            # 输出当前状态
            print(f"当前步数: {self.n_calls}, 模型池大小: {self.model_pool.get_model_count()}")

            # 重置胜率统计
            self.wins = 0
            self.losses = 0
            self.draws = 0

        return True


class BenchmarkCallback(BaseCallback):
    """
    Callback for periodically evaluating the agent against a fixed random opponent.
    """

    def __init__(
        self, eval_freq: int, n_eval_episodes: int, board_size: int, verbose: int = 1, opponent_model_path: Optional[str] = None
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.board_size = board_size
        self.eval_env = None
        self.opponent_model_path = opponent_model_path
        self.benchmark_opponent_type = "Random" if opponent_model_path is None else "Fixed Model"

    def _init_callback(self) -> None:
        """Initialize the evaluation environment."""
        opponent_model_for_eval = None
        opponent_deterministic_for_eval = False

        if self.opponent_model_path:
            if os.path.exists(self.opponent_model_path):
                try:
                    opponent_model_for_eval = MaskablePPO.load(self.opponent_model_path)
                    opponent_deterministic_for_eval = True
                    self.benchmark_opponent_type = f"Fixed Model ({os.path.basename(self.opponent_model_path)})"
                    if self.verbose > 0:
                        print(f"BenchmarkCallback: Loaded opponent model from {self.opponent_model_path} for evaluation.")
                except Exception as e:
                    print(
                        f"BenchmarkCallback: Failed to load opponent model from {self.opponent_model_path}: {e}. Defaulting to random opponent."
                    )
                    self.benchmark_opponent_type = "Random (Fallback)"
            else:
                print(
                    f"BenchmarkCallback: Opponent model path {self.opponent_model_path} not found. Defaulting to random opponent."
                )
                self.benchmark_opponent_type = "Random (Path Not Found)"
        else:
            if self.verbose > 0:
                print(f"BenchmarkCallback: No opponent model path provided. Using random opponent for benchmark.")

        env_fn = make_env(
            board_size=self.board_size,
            opponent_model=opponent_model_for_eval,
            seed=np.random.randint(0, 100000),
            opponent_deterministic=opponent_deterministic_for_eval,
        )
        self.eval_env = DummyVecEnv([env_fn])
        if self.verbose > 0:
            print(
                f"BenchmarkCallback: Evaluation environment initialized (Board: {self.board_size}x{self.board_size}, Opponent: {self.benchmark_opponent_type})."
            )

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            wins = 0
            losses = 0
            draws = 0

            if self.verbose > 0:
                print(
                    f"\nRunning benchmark evaluation against {self.benchmark_opponent_type} opponent for {self.n_eval_episodes} episodes..."
                )

            for episode in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                # agent_player_id is set within GomokuGymEnv's reset. Fetch it for the current episode.
                current_agent_player_id = self.eval_env.get_attr("agent_player_id")[0]
                done_episode = False

                while not done_episode:
                    # Get action mask from the wrapped environment
                    action_masks = self.eval_env.env_method("action_mask")[0]
                    action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)
                    obs, _, dones_vec, infos_vec = self.eval_env.step(action)  # dones_vec is a list/array

                    done_episode = dones_vec[0]

                    if done_episode:
                        winner = infos_vec[0]["winner"]
                        if winner == current_agent_player_id:
                            wins += 1
                        elif winner == -current_agent_player_id:  # Opponent won
                            losses += 1
                        else:  # Draw
                            draws += 1

            total_eval_games = wins + losses + draws
            win_rate_vs_opponent = wins / total_eval_games if total_eval_games > 0 else 0.0

            self.logger.record(
                f"eval/win_rate_vs_{self.benchmark_opponent_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                win_rate_vs_opponent,
            )
            self.logger.record(
                f"eval/wins_vs_{self.benchmark_opponent_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}", wins
            )
            self.logger.record(
                f"eval/losses_vs_{self.benchmark_opponent_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                losses,
            )
            self.logger.record(
                f"eval/draws_vs_{self.benchmark_opponent_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                draws,
            )
            self.logger.record(
                f"eval/total_eval_games_vs_{self.benchmark_opponent_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                total_eval_games,
            )

            if self.verbose > 0:
                print(
                    f"Benchmark evaluation (vs {self.benchmark_opponent_type}) complete: Win rate = {win_rate_vs_opponent:.3f} ({wins}W/{losses}L/{draws}D)"
                )

        return True

    def _on_training_end(self) -> None:
        """Close the evaluation environment when training ends."""
        if self.eval_env is not None:
            self.eval_env.close()
            if self.verbose > 0:
                print(f"BenchmarkCallback: Evaluation environment (vs {self.benchmark_opponent_type}) closed.")
            self.eval_env = None


def make_env(board_size=15, opponent_model=None, seed=0, opponent_deterministic: bool = False):
    """创建环境的工厂函数，用于多进程向量化环境"""

    def _init():
        env = GomokuGymEnv(board_size=board_size, opponent_model=opponent_model, opponent_deterministic=opponent_deterministic)
        # GomokuGymEnv.reset() will be called with its own seed logic for player choice
        # The seed passed here is primarily for SB3's VecEnv wrapper consistency if needed elsewhere
        env.reset(seed=seed + random.randint(0, 10000))  # Add randomness to seed if sub-envs use it for more than player choice
        env = ActionMasker(env, gomoku_mask_fn)  # 应用动作掩码
        return env

    return _init


def update_opponent_models(vec_env, model_pool, update_prob=0.5):
    """
    更新向量环境中的对手模型

    参数:
    - vec_env: 向量化环境
    - model_pool: 模型池管理器
    - update_prob: 更新概率
    """
    # 检查模型池是否有模型
    if model_pool.get_model_count() == 0:
        return

    # 为每个环境决定是否更新及使用哪个模型
    for i in range(vec_env.num_envs):
        # 随机决定是否更新
        if random.random() < update_prob:
            # 从模型池中采样对手模型的路径
            opponent_model_path = model_pool.sample_opponent_model()
            if opponent_model_path:  # 确保路径有效
                # 使用env_method调用子进程中的set_opponent_model方法，传递路径
                vec_env.env_method("set_opponent_model", opponent_model_path, indices=[i])


class OpponentUpdateCallback(BaseCallback):
    def __init__(self, vec_env: SubprocVecEnv | DummyVecEnv, model_pool: ModelPoolManager, update_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.model_pool = model_pool
        # update_freq 是指智能体与环境交互的总步数 (agent steps)
        # SB3 回调中的 self.n_calls 就是这个步数
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        # self.n_calls 是 SB3 提供的自训练开始以来的总步数
        if self.n_calls > 0 and self.n_calls % self.update_freq == 0:
            if self.verbose > 0:
                print(f"\nSteps: {self.n_calls}. Updating opponent models in environments.")

            # 调用您现有的更新函数
            update_opponent_models(self.vec_env, self.model_pool)

            if self.verbose > 0:
                print("Opponent models in environments have been updated.")
        return True


def train_self_play_gomoku(
    board_size,
    total_timesteps,  # 这是总的训练步数
    n_envs,
    save_path,
    model_pool_size,
    model_update_freq,  # ModelPoolUpdateCallback 使用的频率
    opponent_update_freq,  # OpponentUpdateCallback 使用的频率
    save_freq,
    learning_rate_schedule: Callable[[float], float],
    clip_range_schedule: Callable[[float], float],
    gamma,
    n_steps,
    batch_size,
    n_epochs,
    seed,
    initial_exploration_steps,  # 这个参数的直接作用会减弱，但逻辑可以通过回调组合实现
    eval_freq_benchmark,
    n_eval_episodes_benchmark,
):
    """
    使用真正的自我博弈和Maskable PPO训练五子棋智能体
    模型池直接在内存中维护，避免频繁磁盘I/O

    参数:
    - board_size: 棋盘大小
    - total_timesteps: 总训练步数
    - n_envs: 并行环境数
    - save_path: 最终模型保存路径
    - model_pool_size: 内存中保存的历史模型数量
    - model_update_freq: 更新模型池的频率（步数）
    - opponent_update_freq: 更新对手的频率（步数）
    - save_freq: 模型保存频率
    - learning_rate_schedule: 学习率调度函数
    - clip_range_schedule: PPO裁剪范围调度函数
    - gamma: 折扣因子
    - n_steps: 每次更新的步数
    - batch_size: 批次大小
    - n_epochs: 每次更新的epoch数
    - seed: 随机种子
    - initial_exploration_steps: 初始探索步数，使用随机对手
    - eval_freq_benchmark: 新参数，用于benchmark
    - n_eval_episodes_benchmark: 新参数，用于benchmark
    """
    set_random_seed(seed)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_pool = ModelPoolManager(max_models=model_pool_size)
    print(f"初始化内存模型池，最大容量: {model_pool_size}")

    env_fns = [make_env(board_size=board_size, opponent_model=None, seed=seed + i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    print(f"Initialized {n_envs} parallel environments.")

    # ... (设备选择, policy_kwargs 不变) ...
    if th.backends.mps.is_available():
        device = "mps"
        print("检测到MPS设备，将使用MPS进行训练。")
    else:
        device = "auto"
        print("未检测到MPS设备，将使用自动选择的设备 (CUDA 或 CPU)。")

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[64], vf=[64]),
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        learning_rate=learning_rate_schedule,
        clip_range=clip_range_schedule,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=2,
        tensorboard_log="./logs/gomoku_tensorboard/",
        policy_kwargs=policy_kwargs,
        device=device,
    )

    # --- 设置回调函数 ---
    callbacks = []

    # 1. 模型池更新回调 (负责将当前模型添加到池中)
    model_pool_callback = ModelPoolUpdateCallback(
        model_pool=model_pool,
        update_freq=model_update_freq,  # 按指定频率添加模型到池
        verbose=2,
    )
    callbacks.append(model_pool_callback)

    # 2. 对手更新回调 (负责从池中更新环境的对手)
    opponent_update_callback = OpponentUpdateCallback(
        vec_env=vec_env,
        model_pool=model_pool,
        update_freq=opponent_update_freq,  # 按指定频率更新环境中的对手
        verbose=1,  # 可以设为1或2
    )
    callbacks.append(opponent_update_callback)

    # 3. 模型保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.dirname(save_path),
        name_prefix=os.path.basename(save_path),
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    # # 4. 基准评估回调
    benchmark_callback = BenchmarkCallback(
        eval_freq=eval_freq_benchmark,
        n_eval_episodes=n_eval_episodes_benchmark,
        board_size=board_size,
        verbose=2,
        opponent_model_path="models/gomoku_self_play_final.zip",
    )
    callbacks.append(benchmark_callback)

    # --- 关于 initial_exploration_steps ---
    # 在新的结构中，模型池的第一个模型将在 model_update_freq 步后由 ModelPoolUpdateCallback 添加。
    # 在此之前，update_opponent_models 会因为模型池为空而使得环境使用随机对手（这是期望的行为）。
    # 如果您希望在 initial_exploration_steps 后强制添加一个模型（即使它还没达到 model_update_freq），
    # 可以考虑调整 ModelPoolUpdateCallback 的逻辑或添加一个专门的一次性回调。
    # 但通常，让 ModelPoolUpdateCallback 自然地在第一个 model_update_freq 后添加模型是可接受的。
    print(f"模型将在第一个 {model_update_freq} 步之后开始被添加到模型池。")
    print(f"环境中的对手将在第一个 {opponent_update_freq} 步之后开始从模型池更新。")

    # --- 开始训练 ---
    print(f"开始总共 {total_timesteps} 步的训练...")
    model.learn(
        total_timesteps=total_timesteps,  # 使用总的训练步数
        callback=callbacks,
        tb_log_name="gomoku_self_play",
        # 对于一个全新的训练，reset_num_timesteps=True 是合适的，
        # 它会确保 model._total_timesteps 正确设置为这里的 total_timesteps。
        # 如果您是加载模型并继续训练，则通常设为 False。
        reset_num_timesteps=True,
    )

    model.save(save_path + "_final")
    vec_env.close()

    print(f"训练完成！模型已保存到: {save_path}_final")
    return model


def play_against_model(model_path, board_size=15, render=True):
    """
    Let a human player play against the model by clicking on the board

    Parameters:
    - model_path: Model path
    - board_size: Board size
    - render: Whether to render the game
    """
    # Load model
    model = MaskablePPO.load(model_path)

    # Create environment - 不传入opponent_model以避免自动AI对战
    env = GomokuGymEnv(board_size=board_size)
    env = ActionMasker(env, gomoku_mask_fn)

    state, _ = env.reset()
    done = False

    # Create figure and axes for interactive plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    env.render_ax = ax

    # For storing user click position
    click_coords = [None]

    def on_click(event):
        """Handle mouse click events"""
        if event.xdata is not None and event.ydata is not None:
            # Convert float coordinates to integer board coordinates
            x = int(round(event.xdata))
            y = int(round(event.ydata))

            # Ensure coordinates are within board range
            if 0 <= x < board_size and 0 <= y < board_size:
                action = x * board_size + y
                valid_moves = env.env.env.get_valid_moves()

                if action in valid_moves:
                    # Store valid click
                    click_coords[0] = action
                    plt.close(fig)  # Close current figure to continue game flow
                else:
                    plt.title("Invalid move, please try again", color="red")
                    fig.canvas.draw_idle()

    print("Game started! You play as Black. Click on the board to place your stone.")

    while not done:
        if env.env.env.current_player == 1:  # Human player's turn (Black)
            # Display board
            if render:
                env.render_ax = env.env.env.render(ax=ax)
                ax.set_title("Your turn (Black)", fontsize=14)

                # Connect click event
                cid = fig.canvas.mpl_connect("button_press_event", on_click)

                # Show figure and wait for click
                plt.show()

                # Disconnect event
                fig.canvas.mpl_disconnect(cid)

                # Get user action
                action = click_coords[0]
                click_coords[0] = None  # Reset click coordinates

                # Create new figure for display
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                env.render_ax = ax

        else:  # Model's turn (White)
            # Get action mask
            action_mask = env.env.action_mask()

            # Model selects action
            action, _ = model.predict(state, action_masks=action_mask, deterministic=True)

            # Display model action
            x, y = action // board_size, action % board_size
            print(f"AI placed stone at: ({x}, {y})")

        # Execute action
        state, reward, done, _, info = env.step(action)

        # Display latest board
        if render and not (env.env.env.current_player == 1 and not done):
            env.render_ax = env.env.env.render(ax=ax)
            plt.pause(0.5)  # Give player a moment to see AI's move

        # Display game result
        if done:
            env.render_ax = env.env.env.render(ax=ax)
            if "winner" in info and info["winner"] != 0:
                winner = "Black (Player)" if info["winner"] == 1 else "White (AI)"
                ax.set_title(f"Game over, {winner} wins!", fontsize=14, color="blue")
            else:
                ax.set_title("Game over, it's a draw!", fontsize=14, color="green")
            plt.show()

    env.close()


if __name__ == "__main__":
    # 训练模型
    TOTAL_TIMESTEPS = 1 * 10000 * 10000
    MODEL_UPDATE_FREQ = 10 * 10000

    # 定义学习率调度
    lr_schedule = cosine_decay_lr(initial_lr=3e-4, min_lr=1e-6)

    # 定义PPO clip_range调度
    clip_schedule = cosine_decay_clip_range(initial_clip=0.3, final_clip=0.05)

    # trained_model = train_self_play_gomoku(
    #     board_size=10,  # 使用较小棋盘加速训练
    #     total_timesteps=TOTAL_TIMESTEPS,
    #     n_envs=os.cpu_count(),  # 使用 CPU 核心数作为环境数量
    #     save_path="models/gomoku_self_play",
    #     model_pool_size=50,  # 保存 50 个历史模型，用来更新对手
    #     model_update_freq=MODEL_UPDATE_FREQ,
    #     opponent_update_freq=MODEL_UPDATE_FREQ * 2,  # 对手更新频率
    #     save_freq=100000,  # 保存模型频率
    #     learning_rate_schedule=lr_schedule,
    #     clip_range_schedule=clip_schedule,
    #     gamma=0.999,
    #     n_steps=512,
    #     batch_size=256,
    #     n_epochs=5,
    #     seed=0,
    #     initial_exploration_steps=TOTAL_TIMESTEPS * 0.01,
    #     eval_freq_benchmark=20000, # 基准评估频率, 这个值需要乘以 envs 的个数才是实际评估频率
    #     n_eval_episodes_benchmark=5, # 基准评估局数
    # )

    # 可选：与训练好的模型对战
    play_against_model("models/gomoku_self_play_final.zip", board_size=10)