import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Dict, Tuple, Optional, Any, List, NamedTuple
import os
import gymnasium as gym
from gymnasium import spaces
import torch as th
import random
from datetime import datetime
from copy import deepcopy

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.torch_layers import NatureCNN


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
        state[2] = np.ones((self.board_size, self.board_size)) if self.current_player == 1 else np.zeros((self.board_size, self.board_size))
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
            return self._get_state(), 0.0, True, {'winner': self.winner}
        
        # 获取有效的动作列表
        valid_moves = self.get_valid_moves()
        
        # 检查游戏状态一致性
        if len(valid_moves) == 0:
            raise RuntimeError("No valid moves available but game is not marked as done. This indicates a bug in the game logic.")
        
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
        draw = np.count_nonzero(self.board) == self.board_size ** 2
        
        # 从智能体视角计算奖励
        reward = 0.0
        info = {'winner': 0, 'position': (x, y)}
        
        if win:
            self.done = True
            self.winner = self.current_player
            info['winner'] = self.winner
            
            # --- 计算胜负奖励 (鼓励尽快获胜) ---
            num_steps = len(self.history)
            min_win_steps = 9  # 理论上最快获胜所需的步数
            max_steps = self.board_size * self.board_size # 最大可能步数

            # 定义奖励参数
            fastest_win_reward = 2.0  # 最快获胜时的奖励
            slowest_win_reward = 1.0  # 最慢获胜时的奖励
            # 失败惩罚与最慢获胜奖励的绝对值一致
            loss_penalty = -slowest_win_reward 

            # 计算获胜步数相对于最快和最慢情况的归一化进度 [0, 1]
            # 0 表示接近最快获胜, 1 表示接近最慢获胜
            step_range =  max_steps - min_win_steps
            assert num_steps >= min_win_steps, "num_steps must be greater than or equal to min_win_steps = %d" % min_win_steps
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
        ax.set_aspect('equal')
        
        # 绘制网格线
        for i in range(self.board_size):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:  # 黑棋
                    ax.plot(i, j, 'o', markersize=20, markerfacecolor='black', markeredgecolor='black')
                elif self.board[i, j] == -1:  # 白棋
                    ax.plot(i, j, 'o', markersize=20, markerfacecolor='white', markeredgecolor='black')
        
        # 标记最后一步
        if self.last_move:
            x, y = self.last_move
            ax.plot(x, y, 'x', markersize=8, markeredgewidth=3, 
                   color='red' if self.board[x, y] == 1 else 'blue')
        
        # 设置网格标签
        ax.set_xticks(range(self.board_size))
        ax.set_yticks(range(self.board_size))
        ax.set_xticklabels([chr(65 + i) for i in range(self.board_size)])  # A, B, C, ...
        ax.set_yticklabels(range(1, self.board_size + 1))
        
        ax.set_title('Gomoku')
        plt.pause(0.01)
        
        return ax

# 以下为基于SB3、Gymnasium和Maskable PPO的自我博弈训练代码

# 假设模型类型，如果知道具体类型可以替换 Any
StableBaselinesModel = Any

class ModelInfo(NamedTuple):
    """存储模型及其元数据的结构体"""
    model: StableBaselinesModel
    name: str
    iteration: Any # 迭代可以是数字或字符串，如 "initial"
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
        - model: 要添加的模型对象 (将进行深拷贝)
        - iteration: 当前迭代标识
        - win_rate: 模型的胜率

        返回:
        - 生成的模型名称
        """
        # 生成唯一的模型名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"model_iter_{iteration}_{timestamp}"

        # 深拷贝模型以存储其状态快照
        model_copy = deepcopy(model)

        # 创建模型信息对象
        new_model_info = ModelInfo(model=model_copy, name=model_name, iteration=iteration, win_rate=win_rate)

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
            # 将 None 胜率视为负无穷大，以便优先移除
            min_index, _ = min(
                enumerate(self.models),
                key=lambda item: item[1].win_rate 
            )
            # 移除模型
            removed_info = self.models.pop(min_index)
            print(f"模型池已满，已移除胜率最低的模型: {removed_info.name} (Win Rate: {removed_info.win_rate})")

    def sample_opponent_model(self) -> StableBaselinesModel:
        """
        从模型池中根据胜率加权随机选择一个对手模型。
        胜率越高的模型被选中的概率越大。

        返回:
        - 选中的模型对象，如果池为空则返回 None
        """
        if not self.models:
            print("模型池为空，无法采样对手。")
            return None

        # 计算权重：胜率 + 基础权重 (epsilon)
        # 基础权重确保所有模型（包括胜率为0或None的模型）都有机会被选中
        base_weight = 0.01
        weights = [info.win_rate + base_weight for info in self.models]
        # 执行加权随机选择
        chosen_index = random.choices(range(len(self.models)), weights=weights, k=1)[0]
        # 获取选中的模型及其信息用于日志记录
        chosen_model_info = self.models[chosen_index]
        sampled_model = chosen_model_info.model
        model_name = chosen_model_info.name
        chosen_win_rate = chosen_model_info.win_rate # 用于日志的原始胜率

        print(f"从内存池中根据胜率加权选择对手模型: {model_name} (Win Rate: {chosen_win_rate}, Weight: {weights[chosen_index]:.3f})")
        return sampled_model

    def get_latest_model(self) -> Optional[StableBaselinesModel]:
        """获取最新添加的模型对象"""
        if not self.models:
            return None
        # 最新的模型总是在列表的末尾
        return self.models[-1].model

    def get_model_count(self) -> int:
        """获取模型池中的模型数量"""
        return len(self.models)


class GomokuGymEnv(gym.Env):
    """
    五子棋环境的Gymnasium包装器，支持真正的自我博弈
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, board_size: int = 15, opponent_model=None):
        """
        初始化Gymnasium环境
        
        参数:
        - board_size: 棋盘大小
        - opponent_model: 对手模型对象
        """
        super().__init__()
        
        # 创建原始环境
        self.env = GomokuEnv(board_size=board_size)
        self.board_size = board_size
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # 设置观察空间为3通道的二维状态 [黑棋位置, 白棋位置, 当前玩家]
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(3, board_size, board_size),
            dtype=np.float32
        )
        
        # 对手模型
        self.opponent_model = opponent_model
        self.use_random_opponent = True if opponent_model is None else False
        
        if self.opponent_model is not None:
            print("使用提供的对手模型")
        else:
            print("使用随机策略作为对手")
        
        # 用于保存当前游戏信息
        self.current_state = None
        self.render_ax = None
    
    def set_opponent_model(self, model):
        """
        设置对手模型
        
        参数:
        - model: 模型对象
        """
        self.opponent_model = model
        self.use_random_opponent = False if model is not None else True
        if model is not None:
            print(f"已更新对手模型")
    
    def _random_opponent_action(self):
        """随机对手策略"""
        valid_moves = self.env.get_valid_moves()
        if len(valid_moves) > 0:
            return np.random.choice(valid_moves)
        return 0  # 理论上不会发生，因为如果没有有效动作游戏应该已经结束
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 重置内部环境
        self.current_state = self.env.reset()
        
        # 信息字典
        info = {}
        
        return self.current_state, info
    
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
                    self.current_state, 
                    action_masks=opponent_mask,
                    deterministic=False
                )
            else:
                # 使用随机策略
                opponent_action = self._random_opponent_action()
            
            # 执行对手动作
            next_state, opponent_reward, done, info = self.env.step(opponent_action)
            self.current_state = next_state
            
            # 从智能体角度计算奖励（取反）
            reward = -opponent_reward
        
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
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
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
        for i, done in enumerate(self.locals['dones']):
            if done:
                # 获取该环境的信息字典
                info = self.locals['infos'][i]
                if 'winner' in info:
                    winner = info['winner']
                    # 假设训练的智能体视角总是 1 (黑棋)
                    # 如果获胜者是 1，则记录为胜利
                    if winner == 1:
                        self.wins += 1
                    # 如果获胜者是 -1，则记录为失败
                    elif winner == -1:
                        self.losses += 1
                    # 如果获胜者是 0 (或无胜者，视为平局)
                    else:
                        self.draws += 1

        # 检查是否达到更新频率
        if self.n_calls > 0 and self.n_calls % self.update_freq == 0:
            # 计算胜率
            total_games = self.wins + self.losses + self.draws
            win_rate = self.wins / total_games if total_games > 0 else 0.0

            if self.verbose > 0:
                print(f"\nSteps: {self.n_calls}, Updating model pool.")
                print(f"Stats since last update: Wins={self.wins}, Losses={self.losses}, Draws={self.draws}, Total={total_games}")
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


def make_env(board_size=15, opponent_model=None, seed=0):
    """创建环境的工厂函数，用于多进程向量化环境"""
    def _init():
        env = GomokuGymEnv(board_size=board_size, opponent_model=opponent_model)
        env = ActionMasker(env, gomoku_mask_fn)  # 应用动作掩码
        env.reset(seed=seed)
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
            # 从模型池中采样对手
            opponent_model = model_pool.sample_opponent_model()
            # 使用env_method调用子进程中的set_opponent_model方法
            vec_env.env_method("set_opponent_model", opponent_model, indices=[i])


def train_self_play_gomoku(
    board_size=10,  # 使用10x10的棋盘训练更快
    total_timesteps=1_000_000,
    n_envs=max(1, os.cpu_count() // 2 if os.cpu_count() else 1), # Default n_envs based on CPU count
    save_path="models/gomoku_self_play",
    model_pool_size=100,  # 内存中保存的历史模型数量
    model_update_freq=10000,  # 更新模型池的频率
    opponent_update_freq=5000,  # 更新对手的频率
    save_freq=10000,
    learning_rate=3e-4,
    gamma=0.99,  # 折扣因子
    n_steps=256,
    batch_size=128,
    n_epochs=10,  # PPO epochs
    seed=0,
    initial_exploration_steps=50000  # 初始探索步数，使用随机对手
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
    - learning_rate: 学习率
    - gamma: 折扣因子
    - n_steps: 每次更新的步数
    - batch_size: 批次大小
    - n_epochs: 每次更新的epoch数
    - seed: 随机种子
    - initial_exploration_steps: 初始探索步数，使用随机对手
    """
    # 设置随机种子
    set_random_seed(seed)
    
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建模型池管理器
    model_pool = ModelPoolManager(max_models=model_pool_size)
    print(f"初始化内存模型池，最大容量: {model_pool_size}")
    
    # 创建多进程环境 - 初始时使用随机对手
    env_fns = [make_env(board_size=board_size, opponent_model=None, seed=seed+i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)
    print(f"Initialized {n_envs} parallel environments.") # Added log for n_envs
    
    # 确定设备
    if th.backends.mps.is_available():
        device = "mps"
        print("检测到MPS设备，将使用MPS进行训练。")
    else:
        device = "auto" # 自动选择 CUDA 或 CPU
        print("未检测到MPS设备，将使用自动选择的设备 (CUDA 或 CPU)。")
        
    # 设置模型参数 - 使用 CNN
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,           # 使用 NatureCNN 作为特征提取器
        features_extractor_kwargs=dict(features_dim=256), # CNN 输出 256 维特征
        activation_fn=th.nn.ReLU,                    # 激活函数
        net_arch=dict(pi=[64], vf=[64])              # 特征提取后，策略和价值网络各有一个64单元的隐藏层
    )
    
    # 创建模型
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=2,  # 增加日志详细程度
        tensorboard_log="./logs/gomoku_tensorboard/",
        policy_kwargs=policy_kwargs,
        device=device  # 传递设备参数
    )
    
    # 设置回调函数
    model_pool_callback = ModelPoolUpdateCallback(
        model_pool=model_pool,
        update_freq=model_update_freq,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # 因为是多进程，所以除以环境数
        save_path=os.path.dirname(save_path),
        name_prefix=os.path.basename(save_path),
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # 训练循环
    remaining_timesteps = total_timesteps
    while remaining_timesteps > 0:
        # 每次训练少量步数，方便更新对手
        steps_to_train = min(opponent_update_freq, remaining_timesteps)
        
        model.learn(
            total_timesteps=steps_to_train,
            callback=[model_pool_callback, checkpoint_callback],
            tb_log_name="gomoku_self_play",
            reset_num_timesteps=False
        )
        
        # 如果完成了初始探索阶段，确保添加一个模型到模型池
        current_steps = total_timesteps - remaining_timesteps + steps_to_train
        if current_steps >= initial_exploration_steps and model_pool.get_model_count() == 0:
            print("完成初始探索阶段，添加第一个模型到内存模型池")
            model_pool.add_model(model, iteration="initial", win_rate=0.0)
        
        # 更新对手模型
        update_opponent_models(vec_env, model_pool)
        print("已更新对手模型")
        
        # 更新剩余步数
        remaining_timesteps -= steps_to_train
        print(f"已训练步数: {total_timesteps - remaining_timesteps}, 剩余步数: {remaining_timesteps}")
    
    # 保存最终模型
    model.save(save_path + "_final")

    # 关闭环境
    vec_env.close()

    print(f"训练完成！模型已保存到: {save_path}_final")
    return model


def play_against_model(model_path, board_size=15, render=True):
    """
    让人类玩家与模型对战，通过点击棋盘落子
    
    参数:
    - model_path: 模型路径
    - board_size: 棋盘大小
    - render: 是否渲染游戏
    """
    # 加载模型
    model = MaskablePPO.load(model_path)
    
    # 创建环境
    env = GomokuGymEnv(board_size=board_size)
    env = ActionMasker(env, gomoku_mask_fn)
    
    state, _ = env.reset()
    done = False
    
    # 创建图形和轴以进行交互式绘图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    env.render_ax = ax
    
    # 用于存储用户点击的位置
    click_coords = [None]
    
    def on_click(event):
        """处理鼠标点击事件"""
        if event.xdata is not None and event.ydata is not None:
            # 将浮点坐标转换为整数棋盘坐标
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            
            # 确保坐标在棋盘范围内
            if 0 <= x < board_size and 0 <= y < board_size:
                action = x * board_size + y
                valid_moves = env.env.get_valid_moves()
                
                if action in valid_moves:
                    # 存储有效点击
                    click_coords[0] = action
                    plt.close(fig)  # 关闭当前图形以继续游戏流程
                else:
                    plt.title("无效落子位置，请重试", color='red')
                    fig.canvas.draw_idle()
    
    print("开始游戏！您执黑先行。点击棋盘落子。")
    
    while not done:
        if env.env.current_player == 1:  # 人类玩家回合 (黑棋)
            # 显示棋盘
            if render:
                env.render_ax = env.env.render(ax=ax)
                ax.set_title("轮到您落子 (黑棋)", fontsize=14)
                
                # 连接点击事件
                cid = fig.canvas.mpl_connect('button_press_event', on_click)
                
                # 显示图形并等待点击
                plt.show()
                
                # 断开事件连接
                fig.canvas.mpl_disconnect(cid)
                
                # 获取用户动作
                action = click_coords[0]
                click_coords[0] = None  # 重置点击坐标
                
                # 创建新图形用于显示
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                env.render_ax = ax
        
        else:  # 模型回合 (白棋)
            # 获取动作掩码
            action_mask = env.action_mask()
            
            # 模型选择动作
            action, _ = model.predict(state, action_masks=action_mask, deterministic=True)
            
            # 显示模型动作
            x, y = action // board_size, action % board_size
            print(f"AI落子位置: ({x}, {y})")
        
        # 执行动作
        state, reward, done, _, info = env.step(action)
        
        # 显示最新棋盘
        if render and not (env.env.current_player == 1 and not done):
            env.render_ax = env.env.render(ax=ax)
            plt.pause(0.5)  # 给玩家一点时间查看AI的落子
        
        # 显示游戏结果
        if done:
            env.render_ax = env.env.render(ax=ax)
            if 'winner' in info and info['winner'] != 0:
                winner = "黑棋(玩家)" if info['winner'] == 1 else "白棋(AI)"
                ax.set_title(f"游戏结束，{winner}胜利！", fontsize=14, color='blue')
            else:
                ax.set_title("游戏结束，平局！", fontsize=14, color='green')
            plt.show()
    
    env.close()


if __name__ == "__main__":
    # 训练模型
    trained_model = train_self_play_gomoku(
        board_size=10,  # 使用较小棋盘加速训练
        total_timesteps=5000 * 10000,  # 可根据需要调整
        save_path="models/gomoku_self_play",
        model_pool_size=100,  # 在内存中保存100个历史模型
        model_update_freq=20000,  # 模型池更新频率
    )
    
    # 可选：与训练好的模型对战
    # play_against_model("models/gomoku_self_play_final", board_size=10)