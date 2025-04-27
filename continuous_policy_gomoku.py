import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Tuple, Optional, Union, Any


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
        
        # 验证动作值
        action = int(action)  # 确保是整数
        
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
            
            # 计算智能体视角的奖励
            reward = 1.0 if self.current_player == self.agent_perspective else -1.0
            
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
        
        ax.set_title('五子棋')
        plt.pause(0.01)
        
        return ax


class DiscretePolicyGomokuAgent:
    """
    五子棋离散策略梯度智能体
    
    使用softmax策略表示离散动作空间
    """
    
    def __init__(
        self,
        board_size: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
    ) -> None:
        """
        初始化五子棋智能体
        
        参数:
        - board_size: 棋盘大小
        - hidden_dim: 隐藏层维度
        - learning_rate: 学习率
        - gamma: 折扣因子
        """
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # 状态维度: 3个通道(黑棋位置, 白棋位置, 当前玩家)
        # 通道0：表示黑棋位置 - 形状为(15,15)的矩阵，黑棋位置为1，其他为0, 黑棋在哪里
        # 通道1：表示白棋位置 - 形状为(15,15)的矩阵，白棋位置为1，其他为0, 白棋在哪里
        # 通道2：表示当前玩家 - 形状为(15,15)的矩阵，黑棋回合全为1，白棋回合全为0, 现在轮到谁下
        # 因为每个位置都有可能落子，所以每个通道需要棋盘大小*棋盘大小个参数
        self.state_dim = 3 * board_size * board_size
        
        # 动作维度: board_size * board_size (所有可能的落子位置)
        self.action_dim = board_size * board_size
        
        # 隐藏层维度，中间神经元个数
        self.hidden_dim = hidden_dim
        
        # 策略网络参数 (3层神经网络), 输入层[675] state_dim → 隐藏层1[128] hidden_dim → 隐藏层2[128] hidden_dim → 输出层[225] action_dim
        # 用了 He 初始化，在 ReLU 激活函数下表现更好
        # 输入层 -> 隐藏层
        self.w1 = np.random.randn(self.state_dim, hidden_dim) * np.sqrt(2 / self.state_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # 隐藏层 -> 隐藏层
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # 隐藏层 -> 输出层 (动作概率logits)
        self.w3 = np.random.randn(hidden_dim, self.action_dim) * np.sqrt(2 / hidden_dim)
        self.b3 = np.zeros(self.action_dim)
        
        # 轨迹记录
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 训练历史
        self.episode_rewards = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax激活函数，数值稳定版本"""
        x = x - np.max(x)  # 数值稳定性
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def compute_action_probs(self, state: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
        """
        计算动作概率分布
        
        参数:
        - state: 当前状态
        - valid_moves: 有效动作列表（必须提供）
        
        返回:
        - probs: 动作概率分布
        """
        # 确保状态格式正确
        state = state.flatten()
        
        # 前向传播
        h1 = self._relu(np.dot(state, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        logits = np.dot(h2, self.w3) + self.b3
        
        # 计算所有动作的概率
        probs = self._softmax(logits)
        
        # 应用动作掩码，只保留有效动作的概率
        # 创建掩码
        mask = np.zeros(self.action_dim)
        mask[valid_moves] = 1
        
        # 应用掩码
        masked_probs = probs * mask
        
        # 重新归一化
        if np.sum(masked_probs) > 0:
            probs = masked_probs / np.sum(masked_probs)
        else:
            # 如果没有有效动作的概率，均匀分配到所有有效动作
            probs = np.zeros(self.action_dim)
            probs[valid_moves] = 1.0 / len(valid_moves)
        
        return probs
    
    def get_action(self, state: np.ndarray, valid_moves: np.ndarray, explore: bool = True) -> int:
        """
        根据策略选择动作
        
        参数:
        - state: 当前状态
        - valid_moves: 有效动作列表（必须提供）
        - explore: 是否启用探索
        
        返回:
        - action: 选择的动作索引
        
        异常:
        - ValueError: 如果选择了无效动作（这表明存在代码逻辑错误）
        """
        # 验证valid_moves非空
        if len(valid_moves) == 0:
            raise ValueError("Empty valid_moves provided. This indicates a game logic error.")
        
        # 计算动作概率
        probs = self.compute_action_probs(state, valid_moves)
        
        if explore:
            # 基于概率分布采样动作
            action = np.random.choice(self.action_dim, p=probs)
            
            # 验证动作有效性（如果无效则是代码逻辑错误）
            if action not in valid_moves:
                raise ValueError(f"Selected action {action} is not in valid_moves despite probability masking. This indicates a bug in the implementation.")
        else:
            # 测试模式，选择最高概率的动作
            action = np.argmax(probs)
            
            # 验证动作有效性（如果无效则是代码逻辑错误）
            if action not in valid_moves:
                raise ValueError(f"Selected action {action} is not in valid_moves despite probability masking. This indicates a bug in the implementation.")
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """存储一步轨迹"""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
    
    def compute_returns(self) -> np.ndarray:
        """计算每一步的回报(折扣累积奖励)"""
        returns = []
        G = 0.0
        
        # 从后往前计算回报
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # 归一化回报
        returns_array = np.array(returns)
        if len(returns_array) > 1:
            returns_mean = np.mean(returns_array)
            returns_std = np.std(returns_array)
            if returns_std > 1e-6:
                returns_array = (returns_array - returns_mean) / returns_std
        
        return returns_array
    
    def update_policy(self) -> None:
        """更新策略参数"""
        if len(self.states) == 0:
            return
        
        # 计算回报
        returns = self.compute_returns()
        
        # 初始化梯度
        dw1 = np.zeros_like(self.w1)
        db1 = np.zeros_like(self.b1)
        dw2 = np.zeros_like(self.w2)
        db2 = np.zeros_like(self.b2)
        dw3 = np.zeros_like(self.w3)
        db3 = np.zeros_like(self.b3)
        
        # 为每个经验计算梯度
        for t in range(len(self.states)):
            state = self.states[t].flatten()
            action = self.actions[t]
            G = returns[t]
            
            # 前向传播
            h1 = self._relu(np.dot(state, self.w1) + self.b1)
            h2 = self._relu(np.dot(h1, self.w2) + self.b2)
            logits = np.dot(h2, self.w3) + self.b3
            probs = self._softmax(logits)
            
            # 计算策略梯度
            # 对于离散动作空间, ∇_θ log π(a|s) = ∇_θ log P(a) = ∇_θ (log_softmax(logits))_a
            # 这等价于 one_hot(a) - probs
            dlogits = -probs.copy()
            dlogits[action] += 1  # 对选择的动作加1
            
            # 反向传播 - 输出层梯度
            dw3 += np.outer(h2, dlogits) * G
            db3 += dlogits * G
            
            # 反向传播 - 隐藏层梯度
            dh2 = np.dot(dlogits, self.w3.T)
            dz2 = dh2 * (h2 > 0)  # ReLU导数
            dw2 += np.outer(h1, dz2) * G
            db2 += dz2 * G
            
            # 反向传播 - 输入层梯度
            dh1 = np.dot(dz2, self.w2.T)
            dz1 = dh1 * (h1 > 0)  # ReLU导数
            dw1 += np.outer(state, dz1) * G
            db1 += dz1 * G
        
        # 按批次大小归一化梯度
        batch_size = len(self.states)
        dw1 /= batch_size
        db1 /= batch_size
        dw2 /= batch_size
        db2 /= batch_size
        dw3 /= batch_size
        db3 /= batch_size
        
        # 应用梯度(带裁剪)
        lr = self.learning_rate
        
        # 裁剪并应用梯度
        self.w1 += lr * np.clip(dw1, -1.0, 1.0)
        self.b1 += lr * np.clip(db1, -1.0, 1.0)
        self.w2 += lr * np.clip(dw2, -1.0, 1.0)
        self.b2 += lr * np.clip(db2, -1.0, 1.0)
        self.w3 += lr * np.clip(dw3, -1.0, 1.0)
        self.b3 += lr * np.clip(db3, -1.0, 1.0)
        
        # 记录本回合的总奖励
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        
        # 清空历史记录
        self.states = []
        self.actions = []
        self.rewards = []


def test_agent_vs_random(
    env: GomokuEnv,
    agent: DiscretePolicyGomokuAgent,
    games: int = 100,
    render_delay: float = 0.5,
) -> None:
    """测试智能体对抗随机玩家"""
    print("\n===== Testing agent against random player =====")
    
    # 创建绘图
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    for game in range(games):
        # 决定智能体是先手还是后手
        agent_player = 1 if np.random.random() < 0.5 else -1
        
        # 重置环境并设置智能体视角
        state = env.reset(agent_perspective=agent_player)
        done = False
        
        print(f"\nGame {game+1}, Agent plays {'black' if agent_player == 1 else 'white'}")
        ax = env.render(ax=ax)
        plt.pause(render_delay)
        
        while not done:
            # 获取有效动作
            valid_moves = env.get_valid_moves()
            
            if env.current_player == agent_player:
                # 智能体回合
                action = agent.get_action(state, valid_moves, explore=False)
                next_state, _, done, info = env.step(action)
                
                print(f"Agent plays at: {chr(65 + info['position'][0])}{info['position'][1] + 1}")
            else:
                # 随机玩家回合 - 直接从valid_moves中随机选择
                if valid_moves.size > 0:
                    random_action = np.random.choice(valid_moves)
                    next_state, _, done, info = env.step(random_action)
                    
                    print(f"Random player plays at: {chr(65 + info['position'][0])}{info['position'][1] + 1}")
                else:
                    # 棋盘已满，平局
                    break
            
            # 更新状态
            state = next_state
            
            # 渲染
            ax = env.render(ax=ax)
            plt.pause(render_delay)
        
        # 检查比赛结果
        if env.winner == agent_player:
            win_count += 1
            print("Agent wins!")
        elif env.winner == -agent_player:
            loss_count += 1
            print("Random player wins.")
        else:
            draw_count += 1
            print("Draw.")
    
    win_rate = win_count / games * 100
    print(f"\nTest results: {games} games")
    print(f"Agent wins: {win_count} games ({win_rate:.1f}%)")
    print(f"Random player wins: {loss_count} games ({loss_count/games*100:.1f}%)")
    print(f"Draws: {draw_count} games ({draw_count/games*100:.1f}%)")


def test_agent_vs_human(
    env: GomokuEnv,
    agent: DiscretePolicyGomokuAgent,
    render_delay: float = 0.5,
) -> None:
    """测试智能体对抗人类玩家，通过交互式点击"""
    print("\n===== Human vs Agent =====")
    
    # 创建交互式绘图
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # 决定谁先手
    print("Do you want to play black (first) or white (second)?")
    user_color = input("Enter 'b' for black, 'w' for white: ").lower()
    
    agent_player = -1 if user_color == 'b' else 1
    human_player = 1 if user_color == 'b' else -1
    
    # 重置环境并设置智能体视角
    state = env.reset(agent_perspective=agent_player)
    done = False
    
    print(f"\nYou play {'black' if human_player == 1 else 'white'}")
    print(f"Agent plays {'black' if agent_player == 1 else 'white'}")
    print("Click on the board to place your stone")
    
    # 用于存储人类走子的变量
    human_move = None
    
    # 处理鼠标点击的函数
    def on_click(event):
        nonlocal human_move
        
        # 只在人类回合且点击在绘图区域内时处理点击
        if env.current_player == human_player and event.inaxes == ax and not done:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            
            # 检查位置是否有效
            if 0 <= x < env.board_size and 0 <= y < env.board_size and env.board[x, y] == 0:
                human_move = (x, y)
                plt.title(f"You placed at: {chr(65 + x)}{y + 1}", fontsize=14)
                plt.draw()
    
    # 连接点击事件
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    
    # 初始渲染
    ax = env.render(ax=ax)
    
    if env.current_player == human_player:
        plt.title("Your turn - click to place a stone", fontsize=14)
    else:
        plt.title("Agent's turn...", fontsize=14)
        
    plt.draw()
    plt.pause(0.1)
    
    while not done:
        if env.current_player == agent_player:
            # 智能体回合
            plt.title("Agent's turn...", fontsize=14)
            plt.draw()
            plt.pause(0.5)  # 暂停以显示"智能体回合"消息
            
            # 获取有效动作
            valid_moves = env.get_valid_moves()
            
            action = agent.get_action(state, valid_moves, explore=False)
            next_state, _, done, info = env.step(action)
            
            # 显示智能体的走子
            move_str = f"{chr(65 + info['position'][0])}{info['position'][1] + 1}"
            print(f"Agent plays at: {move_str}")
            plt.title(f"Agent placed at: {move_str}", fontsize=14)
            
            # 渲染
            ax = env.render(ax=ax)
            plt.draw()
            plt.pause(render_delay)
            
            if not done:
                plt.title("Your turn - click to place a stone", fontsize=14)
                plt.draw()
        else:
            # 人类玩家回合 - 等待点击
            human_move = None
            
            while human_move is None and not done:
                plt.pause(0.1)  # 每0.1秒检查一次点击
            
            if not done:  # 确保游戏在等待时没有结束
                # 转换为动作索引
                x, y = human_move
                human_action = x * env.board_size + y
                
                next_state, _, done, _ = env.step(human_action)
                
                # 渲染
                ax = env.render(ax=ax)
                plt.draw()
                plt.pause(render_delay)
        
        # 更新状态
        state = next_state
    
    # 游戏结束
    if env.winner == human_player:
        result = "You win!"
    elif env.winner == agent_player:
        result = "Agent wins."
    else:
        result = "Draw."
        
    print(result)
    plt.title(f"Game over - {result}", fontsize=16, fontweight='bold')
    plt.draw()
    
    # 保持窗口打开直到手动关闭
    plt.show()


def create_opponent_copy(agent, env):
    """创建当前智能体的副本作为对手"""
    opponent = DiscretePolicyGomokuAgent(
        board_size=env.board_size,
        hidden_dim=agent.hidden_dim,
        learning_rate=agent.learning_rate,
        gamma=agent.gamma
    )
    # 复制参数
    opponent.w1 = agent.w1.copy()
    opponent.b1 = agent.b1.copy()
    opponent.w2 = agent.w2.copy()
    opponent.b2 = agent.b2.copy()
    opponent.w3 = agent.w3.copy()
    opponent.b3 = agent.b3.copy()
    return opponent


def select_opponent(historical_opponents):
    """根据胜率从历史智能体中选择对手"""
    if len(historical_opponents) > 1:
        # 简化的对手选择逻辑：优先选择胜率高的对手，同时保留一定探索性
        exploration_factor = 0.2  # 探索因子：值越大，随机性越强
        
        # 获取所有对手胜率
        win_rates = np.array([opponent['win_rate'] for opponent in historical_opponents])
        
        # 结合探索因子计算选择概率
        selection_probs = win_rates + exploration_factor/len(historical_opponents)
        selection_probs = selection_probs / np.sum(selection_probs)  # 归一化
        
        # 按概率选择对手
        opponent_idx = np.random.choice(len(historical_opponents), p=selection_probs)
    else:
        # 只有一个可用对手（初始版本）
        opponent_idx = 0
    
    opponent_info = historical_opponents[opponent_idx]
    return opponent_idx, opponent_info


def update_opponent_stats(winner, agent_plays_black, opponent_info, win_history):
    """根据游戏结果更新对手统计信息"""
    if winner == 1:  # 黑棋获胜
        win_history.append(1 if agent_plays_black else -1)
        # 更新对手胜率信息
        if not agent_plays_black:  # 对手获胜
            opponent_info['wins'] += 1
    elif winner == -1:  # 白棋获胜
        win_history.append(1 if not agent_plays_black else -1)
        # 更新对手胜率信息
        if agent_plays_black:  # 对手获胜
            opponent_info['wins'] += 1
    else:  # 平局
        win_history.append(0)
    
    # 更新对手的游戏总数和胜率
    opponent_info['games'] += 1
    opponent_info['win_rate'] = opponent_info['wins'] / opponent_info['games']


def manage_historical_opponents(historical_opponents, agent, env, episode, save_opponent_freq, should_render):
    """管理历史对手集合"""
    if (episode + 1) % save_opponent_freq == 0:
        max_opponents = 50  # 限制历史对手最大数量
        if len(historical_opponents) >= max_opponents:
            # 移除表现最差的对手，而不是最旧的
            # 我们定义"表现差"为胜率低且游戏次数足够的对手
            min_games_threshold = 5  # 最少游戏次数阈值，避免移除新对手
            candidates = [i for i, opp in enumerate(historical_opponents) 
                         if opp['games'] >= min_games_threshold]
            
            if candidates:
                # 如果有足够多对战的对手，移除胜率最低的
                win_rates = [historical_opponents[i]['win_rate'] for i in candidates]
                worst_idx = candidates[np.argmin(win_rates)]
                historical_opponents.pop(worst_idx)
            else:
                # 如果没有，则移除最旧的
                historical_opponents.pop(0)
        
        # 添加新对手
        historical_opponents.append({
            'model': create_opponent_copy(agent, env),
            'wins': 0,
            'games': 0,
            'win_rate': 0.5  # 初始胜率设为0.5（中性评价）
        })
        
        if should_render:
            print(f"Saved current agent as opponent version {len(historical_opponents) - 1}")


def update_visualization(ax2, episode_rewards, win_history):
    """更新训练进度可视化"""
    ax2.clear()
    ax2.plot(episode_rewards, 'b-', alpha=0.3, label='Raw Reward')
    
    # 添加移动平均线
    window = min(50, len(episode_rewards))
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        ax2.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-ep Average')
    
    # 添加胜率
    if len(win_history) > 10:
        win_rate = [np.mean([1 if w == 1 else 0 for w in win_history[max(0, i-50):i+1]]) * 100 
                   for i in range(len(win_history))]
        ax2.plot(win_rate, 'g-', linewidth=2, label='Win Rate (%)')
    
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Total Reward / Win Rate (%)")
    ax2.set_title("Training Progress")
    ax2.legend(loc='upper left')
    plt.pause(0.01)


def play_single_game(env, agent, opponent, agent_plays_black, should_render, ax1):
    """运行智能体和对手之间的单场训练游戏"""
    # 重置环境并设置智能体视角
    agent_perspective = 1 if agent_plays_black else -1
    state = env.reset(agent_perspective=agent_perspective)
    
    # 存储智能体的轨迹
    agent_states = []
    agent_actions = []
    agent_rewards = []
    total_reward = 0.0
    done = False
    
    if should_render:
        ax1 = env.render(ax=ax1)
    
    while not done:
        current_player = env.current_player
        is_agent_turn = (agent_plays_black and current_player == 1) or (not agent_plays_black and current_player == -1)
        
        # 获取有效动作
        valid_moves = env.get_valid_moves()
        
        if is_agent_turn:
            # 智能体回合
            action = agent.get_action(state, valid_moves, explore=True)
            next_state, reward, done, info = env.step(action)
            
            # 存储智能体的轨迹
            agent_states.append(state.copy())
            agent_actions.append(action)
            agent_rewards.append(reward)
            
            # 累积奖励用于绘图
            total_reward += reward
        else:
            # 对手回合（历史智能体）
            action = opponent.get_action(state, valid_moves, explore=False)  # 对手不进行探索
            next_state, _, done, _ = env.step(action)
            # 对手的奖励不需要记录
        
        # 更新状态
        state = next_state
        
        # 渲染
        if should_render:
            ax1 = env.render(ax=ax1)
            plt.pause(0.1)
    
    return total_reward, agent_states, agent_actions, agent_rewards


def print_episode_info(episode, episodes, total_reward, winner, agent_plays_black, win_history):
    """打印回合信息"""
    print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")
    if winner == 1:
        print(f"{'Agent' if agent_plays_black else 'Opponent'} (Black) wins!")
    elif winner == -1:
        print(f"{'Agent' if not agent_plays_black else 'Opponent'} (White) wins!")
    else:
        print("Draw!")
    
    # 打印胜率
    if len(win_history) > 10:
        recent_win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-min(50, len(win_history)):]]) * 100
        print(f"Recent win rate: {recent_win_rate:.1f}%")


def self_play_training(
    env: GomokuEnv,
    agent: DiscretePolicyGomokuAgent,
    episodes: int = 500,
    render_freq: int = 50,
    save_opponent_freq: int = 20,  # 每多少回合保存一次对手模型
) -> List[float]:
    """使用历史版本智能体进行自我对弈训练"""
    print("===== Starting self-play training with historical opponents =====")
    
    # 创建交互式绘图
    fig: Figure = plt.figure(figsize=(15, 8))

    # 游戏可视化
    ax1: Axes = fig.add_subplot(1, 2, 1)

    # 训练进度可视化
    ax2: Axes = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Training Progress")
    
    episode_rewards: List[float] = []
    win_history: List[int] = []
    
    # 存储具有胜率信息的历史对手
    # 每个项目是一个字典，包含: 'model': 对手模型, 'wins': 0, 'games': 0, 'win_rate': 0.0
    historical_opponents: List[Dict[str, Any]] = []
    
    # 将初始版本保存为第一个对手
    initial_opponent_model: DiscretePolicyGomokuAgent = create_opponent_copy(agent, env)
    historical_opponents.append({
        'model': initial_opponent_model,
        'wins': 0,
        'games': 0,
        'win_rate': 0.5
    })
    
    for episode in range(episodes):
        # 决定智能体是黑方还是白方, 1代表黑方, -1代表白方
        agent_plays_black: bool = episode % 2 == 0
        
        # 从历史智能体中选择对手
        opponent_idx: int
        opponent_info: Dict[str, Any]
        opponent_idx, opponent_info = select_opponent(historical_opponents)
        opponent: DiscretePolicyGomokuAgent = opponent_info['model']
        
        # 决定是否渲染本回合
        should_render: bool = episode % render_freq == 0
        
        if should_render:
            print(f"Episode {episode+1}, Agent plays {'black' if agent_plays_black else 'white'}")
            win_rate_str: str = f", Win rate: {opponent_info['win_rate']*100:.1f}%" if opponent_info['games'] > 0 else ""
            print(f"Playing against opponent version {opponent_idx}{win_rate_str}")
        
        # 运行智能体和对手之间的单场游戏
        total_reward: float
        agent_states: List[np.ndarray]
        agent_actions: List[int]
        agent_rewards: List[float]
        total_reward, agent_states, agent_actions, agent_rewards = play_single_game(
            env, agent, opponent, agent_plays_black, should_render, ax1
        )
        
        # 更新对手统计信息
        update_opponent_stats(env.winner, agent_plays_black, opponent_info, win_history)
        
        # 将轨迹存储在智能体中用于学习
        state: np.ndarray
        action: int
        reward: float
        for state, action, reward in zip(agent_states, agent_actions, agent_rewards):
            agent.store_transition(s, a, r)
        
        # 更新策略
        agent.update_policy()
        
        # 记录本回合的奖励
        episode_rewards.append(total_reward)
        
        # 管理历史对手集合
        manage_historical_opponents(historical_opponents, agent, env, episode, save_opponent_freq, should_render)
        
        # 更新进度可视化
        if episode > 0:
            update_visualization(ax2, episode_rewards, win_history)
        
        # 打印训练信息
        if should_render or episode == episodes - 1:
            print_episode_info(episode, episodes, total_reward, env.winner, agent_plays_black, win_history)
    
    print("===== Training complete =====")
    return episode_rewards


if __name__ == "__main__":
    # 创建环境
    env = GomokuEnv(board_size=15)
    
    # 创建智能体
    agent = DiscretePolicyGomokuAgent(
        board_size=15,
        hidden_dim=256,
        learning_rate=0.001,
        gamma=0.99,
    )
    
    # 使用自我对弈进行训练（唯一的训练选项）
    rewards = self_play_training(env, agent, episodes=500, render_freq=50)
    
    # 测试模式选择
    print("\nSelect test mode:")
    print("1. Play against random player")
    print("2. Play against human")
    
    try:
        test_mode = int(input("Enter mode (1-2): "))
    except ValueError:
        test_mode = 1  # 如果输入无效则默认为随机玩家
    
    # 基于所选模式进行测试
    if test_mode == 2:
        # 人类对抗智能体
        test_agent_vs_human(env, agent)
    else:
        # 智能体对抗随机玩家
        test_agent_vs_random(env, agent, games=20)
    
    # 绘制训练奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(agent.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.show() 