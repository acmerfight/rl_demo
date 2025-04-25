import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Tuple, Optional, Union, Any


class GomokuEnv:
    """
    五子棋环境
    
    状态(State): 棋盘状态
    动作(Action): 连续值 [x, y] 表示放置棋子的位置
    """
    
    def __init__(self, board_size: int = 15) -> None:
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 代表黑棋, -1 代表白棋
        self.done = False
        self.winner = 0
        self.last_move = None
        
        # 用于可视化
        self.history = []
    
    def reset(self) -> np.ndarray:
        """重置游戏环境"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.done = False
        self.winner = 0
        self.last_move = None
        self.history = []
        return self._get_state()
    
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
    
    def _continuous_to_discrete(self, action: np.ndarray) -> Tuple[int, int]:
        """
        将连续动作映射到离散棋盘位置
        
        输入动作范围: [-1, 1] x [-1, 1]
        映射到棋盘位置: [0, board_size-1] x [0, board_size-1]
        """
        x = round(((action[0] + 1) / 2) * (self.board_size - 1))
        y = round(((action[1] + 1) / 2) * (self.board_size - 1))
        
        # 确保位置在有效范围内
        x = max(0, min(self.board_size - 1, x))
        y = max(0, min(self.board_size - 1, y))
        
        return int(x), int(y)
    
    def _find_closest_empty(self, x: int, y: int) -> Tuple[int, int]:
        """
        寻找距离(x,y)最近的空位
        
        如果目标位置已经被占用，会找到最近的空位
        使用螺旋搜索模式提高效率
        """
        # 验证输入坐标
        x = max(0, min(self.board_size - 1, x))
        y = max(0, min(self.board_size - 1, y))
        
        if self.board[x, y] == 0:
            return x, y
        
        # 如果目标位置已占用，寻找最近的空位
        # 使用螺旋搜索模式提高效率
        for d in range(1, self.board_size):
            # 检查距离(x,y)为d的方形周围的位置
            for i in range(-d, d+1):
                for j in [-d, d]:  # 上下边缘
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                        return nx, ny
                        
            for i in [-d, d]:  # 左右边缘(不包括已检查的角落)
                for j in range(-d+1, d):
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == 0:
                        return nx, ny
            
        # 备选方案：如果螺旋搜索失败，搜索整个棋盘
        empty_positions = np.argwhere(self.board == 0)
        if len(empty_positions) > 0:
            distances = [(pos[0] - x)**2 + (pos[1] - y)**2 for pos in empty_positions]
            closest_idx = np.argmin(distances)
            return empty_positions[closest_idx][0], empty_positions[closest_idx][1]
            
        # 如果没有空位(在正常游戏中不应该发生)
        return x, y
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行一步
        
        参数:
        - action: 连续值 [x, y]，范围为[-1, 1]
        
        返回:
        - state: 新状态
        - reward: 奖励
        - done: 游戏是否结束
        - info: 额外信息
        """
        if self.done:
            return self._get_state(), 0.0, True, {'winner': self.winner}
        
        # 验证动作数组
        if not isinstance(action, np.ndarray) or action.shape != (2,):
            action = np.clip(np.array(action, dtype=np.float32).flatten()[:2], -1.0, 1.0)
        
        # 将连续动作映射到离散位置
        x, y = self._continuous_to_discrete(action)
        
        # 寻找最近的空位
        x, y = self._find_closest_empty(x, y)
        
        # 计算目标位置在连续动作空间中的表示
        target_continuous = np.array([
            2 * x / (self.board_size - 1) - 1,
            2 * y / (self.board_size - 1) - 1
        ])
        
        # 计算到目标位置的距离(用于奖励计算)
        target_dist = np.sqrt(np.sum((action - target_continuous) ** 2))
        
        # 执行动作
        self.board[x, y] = self.current_player
        self.last_move = (x, y)
        self.history.append((x, y, self.current_player))
        
        # 检查游戏是否结束
        win = self._check_win(x, y)
        draw = np.count_nonzero(self.board) == self.board_size ** 2
        
        reward = 0.0
        
        if win:
            self.done = True
            self.winner = self.current_player
            reward = 1.0  # 获胜奖励
        elif draw:
            self.done = True
            reward = 0.1  # 平局奖励
        else:
            # 基于动作精确度的小奖励/惩罚
            # 如果连续动作映射接近实际放置位置，给予小奖励
            # 否则给予小惩罚
            reward = max(0, 0.02 * (1 - target_dist))
            
            # 切换玩家
            self.current_player *= -1
        
        return self._get_state(), reward, self.done, {'winner': self.winner, 'position': (x, y)}
    
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


class ContinuousPolicyGomokuAgent:
    """
    五子棋连续策略梯度智能体
    
    使用高斯策略（正态分布）来表示连续动作空间
    策略参数包括均值网络和标准差
    """
    
    def __init__(
        self,
        board_size: int,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        learning_rate_log_std_factor: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        """
        初始化五子棋智能体
        
        参数:
        - board_size: 棋盘大小
        - hidden_dim: 隐藏层维度
        - learning_rate: 学习率
        - learning_rate_log_std_factor: 标准差学习率因子
        - gamma: 折扣因子
        """
        self.board_size = board_size
        self.learning_rate = learning_rate
        self.learning_rate_log_std_factor = learning_rate_log_std_factor
        self.gamma = gamma
        
        # 状态维度: 3个通道(黑棋位置, 白棋位置, 当前玩家)
        self.state_dim = 3 * board_size * board_size
        
        # 动作维度: 2 (x, y)
        self.action_dim = 2
        
        # 隐藏层维度
        self.hidden_dim = hidden_dim
        
        # 策略网络参数 (3层神经网络)
        # 输入层 -> 隐藏层
        self.w1 = np.random.randn(self.state_dim, hidden_dim) * np.sqrt(2 / self.state_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # 隐藏层 -> 隐藏层
        self.w2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        # 隐藏层 -> 输出层 (动作均值)
        self.w3 = np.random.randn(hidden_dim, self.action_dim) * np.sqrt(2 / hidden_dim)
        self.b3 = np.zeros(self.action_dim)
        
        # 标准差参数 - 使用log_std防止负值
        self.log_std = np.zeros(self.action_dim) - 0.5  # 初始标准差约为0.6
        
        # 轨迹记录
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 训练历史
        self.episode_rewards = []
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh激活函数"""
        return np.tanh(x)
    
    def compute_action_mean(self, state: np.ndarray) -> np.ndarray:
        """计算动作均值"""
        # 确保状态格式正确
        state = state.flatten()
        
        # 前向传播
        h1 = self._relu(np.dot(state, self.w1) + self.b1)
        h2 = self._relu(np.dot(h1, self.w2) + self.b2)
        mean = self._tanh(np.dot(h2, self.w3) + self.b3)  # Tanh输出范围[-1,1]
        
        return mean
    
    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """根据策略选择动作"""
        # 计算动作均值
        action_mean = self.compute_action_mean(state)
        
        if explore:
            # 基于均值和标准差采样动作
            std = np.exp(np.clip(self.log_std, -5.0, 2.0))
            noise = np.random.randn(self.action_dim)
            action = action_mean + noise * std
            
            # 将动作限制在[-1, 1]范围内
            action = np.clip(action, -1.0, 1.0)
        else:
            # 测试模式，直接使用均值
            action = action_mean
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        """存储一步轨迹"""
        self.states.append(state.copy())
        self.actions.append(action.copy())
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
        dlog_std = np.zeros_like(self.log_std)
        
        # 为每个经验计算梯度
        for t in range(len(self.states)):
            state = self.states[t].flatten()
            action = self.actions[t]
            G = returns[t]
            
            # 前向传播
            h1 = self._relu(np.dot(state, self.w1) + self.b1)
            h2 = self._relu(np.dot(h1, self.w2) + self.b2)
            mean = self._tanh(np.dot(h2, self.w3) + self.b3)
            
            # 计算对数概率梯度
            std = np.exp(np.clip(self.log_std, -5.0, 2.0))
            variance = std ** 2 + 1e-6  # 添加小常数防止除零
            action_diff = action - mean
            
            # 均值对数概率梯度
            # 正确公式: ∇_μ log π(a|s) = (a - μ) / σ²
            mean_grad = action_diff / variance
            
            # log_std对数概率梯度
            log_std_grad = ((action_diff ** 2) / variance - 1.0)
            
            # 反向传播 - 输出层梯度
            dz3 = mean_grad * (1 - mean ** 2)  # Tanh导数: 1 - tanh^2(x)
            dw3 += np.outer(h2, dz3) * G
            db3 += dz3 * G
            
            # 反向传播 - 隐藏层梯度
            dh2 = np.dot(dz3, self.w3.T)
            dz2 = dh2 * (h2 > 0)  # ReLU导数
            dw2 += np.outer(h1, dz2) * G
            db2 += dz2 * G
            
            # 反向传播 - 输入层梯度
            dh1 = np.dot(dz2, self.w2.T)
            dz1 = dh1 * (h1 > 0)  # ReLU导数
            dw1 += np.outer(state, dz1) * G
            db1 += dz1 * G
            
            # 累积log_std梯度
            dlog_std += log_std_grad * G
        
        # 按批次大小归一化梯度
        batch_size = len(self.states)
        dw1 /= batch_size
        db1 /= batch_size
        dw2 /= batch_size
        db2 /= batch_size
        dw3 /= batch_size
        db3 /= batch_size
        dlog_std /= batch_size
        
        # 应用梯度(带裁剪)
        lr = self.learning_rate
        
        # 裁剪并应用梯度
        self.w1 += lr * np.clip(dw1, -1.0, 1.0)
        self.b1 += lr * np.clip(db1, -1.0, 1.0)
        self.w2 += lr * np.clip(dw2, -1.0, 1.0)
        self.b2 += lr * np.clip(db2, -1.0, 1.0)
        self.w3 += lr * np.clip(dw3, -1.0, 1.0)
        self.b3 += lr * np.clip(db3, -1.0, 1.0)
        self.log_std += lr * self.learning_rate_log_std_factor * np.clip(dlog_std, -1.0, 1.0)
        
        # 限制标准差范围
        self.log_std = np.clip(self.log_std, -5.0, 2.0)
        
        # 记录本回合的总奖励
        episode_reward = sum(self.rewards)
        self.episode_rewards.append(episode_reward)
        
        # 清空历史记录
        self.states = []
        self.actions = []
        self.rewards = []


def test_agent_vs_random(
    env: GomokuEnv,
    agent: ContinuousPolicyGomokuAgent,
    games: int = 100,
    render_delay: float = 0.5,
) -> None:
    """Test agent against random player"""
    print("\n===== Testing agent against random player =====")
    
    # Create plot
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    win_count = 0
    loss_count = 0
    draw_count = 0
    
    for game in range(games):
        # Reset environment
        state = env.reset()
        done = False
        
        # Decide if agent plays first or second
        agent_player = 1 if np.random.random() < 0.5 else -1
        
        print(f"\nGame {game+1}, Agent plays {'black' if agent_player == 1 else 'white'}")
        ax = env.render(ax=ax)
        plt.pause(render_delay)
        
        while not done:
            if env.current_player == agent_player:
                # Agent's turn
                action = agent.get_action(state, explore=False)
                next_state, _, done, info = env.step(action)
                
                print(f"Agent plays at: {chr(65 + info['position'][0])}{info['position'][1] + 1}")
            else:
                # Random player's turn
                valid_moves = []
                for i in range(env.board_size):
                    for j in range(env.board_size):
                        if env.board[i, j] == 0:
                            valid_moves.append((i, j))
                
                if valid_moves:
                    random_move = valid_moves[np.random.choice(len(valid_moves))]
                    # Convert to continuous action
                    random_action = np.array([
                        2 * random_move[0] / (env.board_size - 1) - 1,
                        2 * random_move[1] / (env.board_size - 1) - 1
                    ])
                    next_state, _, done, info = env.step(random_action)
                    
                    print(f"Random player plays at: {chr(65 + info['position'][0])}{info['position'][1] + 1}")
                else:
                    # Board is full, draw
                    break
            
            # Update state
            state = next_state
            
            # Render
            ax = env.render(ax=ax)
            plt.pause(render_delay)
        
        # Check match result
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
    agent: ContinuousPolicyGomokuAgent,
    render_delay: float = 0.5,
) -> None:
    """Test agent against human player with interactive clicking"""
    print("\n===== Human vs Agent =====")
    
    # Create interactive plot
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Reset environment
    state = env.reset()
    done = False
    
    # Decide who plays first
    print("Do you want to play black (first) or white (second)?")
    user_color = input("Enter 'b' for black, 'w' for white: ").lower()
    
    agent_player = -1 if user_color == 'b' else 1
    human_player = 1 if user_color == 'b' else -1
    
    print(f"\nYou play {'black' if human_player == 1 else 'white'}")
    print(f"Agent plays {'black' if agent_player == 1 else 'white'}")
    print("Click on the board to place your stone")
    
    # Variable to store human's move
    human_move = None
    
    # Function to handle mouse clicks
    def on_click(event):
        nonlocal human_move
        
        # Only process clicks if it's human's turn and within the axes
        if env.current_player == human_player and event.inaxes == ax and not done:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            
            # Check if the position is valid
            if 0 <= x < env.board_size and 0 <= y < env.board_size and env.board[x, y] == 0:
                human_move = (x, y)
                plt.title(f"You placed at: {chr(65 + x)}{y + 1}", fontsize=14)
                plt.draw()
    
    # Connect the click event
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    
    # Initial render
    ax = env.render(ax=ax)
    
    if env.current_player == human_player:
        plt.title("Your turn - click to place a stone", fontsize=14)
    else:
        plt.title("Agent's turn...", fontsize=14)
        
    plt.draw()
    plt.pause(0.1)
    
    while not done:
        if env.current_player == agent_player:
            # Agent's turn
            plt.title("Agent's turn...", fontsize=14)
            plt.draw()
            plt.pause(0.5)  # Pause to show "Agent's turn" message
            
            action = agent.get_action(state, explore=False)
            next_state, _, done, info = env.step(action)
            
            # Show agent's move
            move_str = f"{chr(65 + info['position'][0])}{info['position'][1] + 1}"
            print(f"Agent plays at: {move_str}")
            plt.title(f"Agent placed at: {move_str}", fontsize=14)
            
            # Render
            ax = env.render(ax=ax)
            plt.draw()
            plt.pause(render_delay)
            
            if not done:
                plt.title("Your turn - click to place a stone", fontsize=14)
                plt.draw()
        else:
            # Human player's turn - wait for click
            human_move = None
            
            while human_move is None and not done:
                plt.pause(0.1)  # Check for clicks every 0.1 seconds
            
            if not done:  # Make sure game didn't end while waiting
                # Convert to continuous action
                x, y = human_move
                human_action = np.array([
                    2 * x / (env.board_size - 1) - 1,
                    2 * y / (env.board_size - 1) - 1
                ])
                
                next_state, _, done, _ = env.step(human_action)
                
                # Render
                ax = env.render(ax=ax)
                plt.draw()
                plt.pause(render_delay)
        
        # Update state
        state = next_state
    
    # Game ended
    if env.winner == human_player:
        result = "You win!"
    elif env.winner == agent_player:
        result = "Agent wins."
    else:
        result = "Draw."
        
    print(result)
    plt.title(f"Game over - {result}", fontsize=16, fontweight='bold')
    plt.draw()
    
    # Keep the window open until closed manually
    plt.show()


def self_play_training(
    env: GomokuEnv,
    agent: ContinuousPolicyGomokuAgent,
    episodes: int = 500,
    render_freq: int = 50,
    save_opponent_freq: int = 20,  # 每多少回合保存一次对手模型
) -> List[float]:
    """Self-play training with historical versions of the agent"""
    print("===== Starting self-play training with historical opponents =====")
    
    # Create interactive plot
    fig = plt.figure(figsize=(15, 8))
    
    # Game visualization
    ax1 = fig.add_subplot(1, 2, 1)
    
    # Training progress visualization
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Training Progress")
    
    episode_rewards = []
    win_history = []
    
    # Store historical opponents (keep copies of agent parameters)
    historical_opponents = []
    
    # Function to create a copy of the current agent
    def create_opponent_copy(agent):
        opponent = ContinuousPolicyGomokuAgent(
            board_size=env.board_size,
            hidden_dim=agent.hidden_dim,
            learning_rate=agent.learning_rate,
            gamma=agent.gamma
        )
        # Copy parameters
        opponent.w1 = agent.w1.copy()
        opponent.b1 = agent.b1.copy()
        opponent.w2 = agent.w2.copy()
        opponent.b2 = agent.b2.copy()
        opponent.w3 = agent.w3.copy()
        opponent.b3 = agent.b3.copy()
        opponent.log_std = agent.log_std.copy()
        return opponent
    
    # Save initial version as the first opponent
    historical_opponents.append(create_opponent_copy(agent))
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0.0
        done = False
        
        # Choose opponent from historical agents (random selection with preference for recent ones)
        if len(historical_opponents) > 1:
            # Higher probability to select more recent opponents
            p = np.array([1/(len(historical_opponents) - i) for i in range(len(historical_opponents))])
            p = p / np.sum(p)
            opponent_idx = np.random.choice(len(historical_opponents), p=p)
            opponent = historical_opponents[opponent_idx]
        else:
            # Only one opponent available (initial version)
            opponent = historical_opponents[0]
        
        # Decide who plays black (agent or opponent)
        # Alternate between playing black and white to ensure balanced learning
        agent_plays_black = episode % 2 == 0
        
        # Store trajectories
        agent_states = []
        agent_actions = []
        agent_rewards = []
        
        # Render if it's a visualization episode
        should_render = episode % render_freq == 0
        
        if should_render:
            print(f"Episode {episode+1}, Agent plays {'black' if agent_plays_black else 'white'}")
            print(f"Playing against opponent version {opponent_idx}" if len(historical_opponents) > 1 else "Playing against initial version")
            ax1 = env.render(ax=ax1)
        
        while not done:
            current_player = env.current_player
            is_agent_turn = (agent_plays_black and current_player == 1) or (not agent_plays_black and current_player == -1)
            
            if is_agent_turn:
                # Agent's turn
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store trajectory
                agent_states.append(state.copy())
                agent_actions.append(action.copy())
                agent_rewards.append(reward)
                
                # Accumulate reward for plotting
                total_reward += reward
            else:
                # Opponent's turn (historical agent)
                action = opponent.get_action(state, explore=False)  # No exploration for opponent
                next_state, _, done, _ = env.step(action)
                
                # If agent loses, give negative reward for its last move
                if done and ((env.winner == 1 and not agent_plays_black) or (env.winner == -1 and agent_plays_black)):
                    if agent_rewards:
                        agent_rewards[-1] = -1.0
                        total_reward -= 2.0  # Adjust total reward (subtract 2 since we already added +1)
            
            # Update state
            state = next_state
            
            # Render
            if should_render:
                ax1 = env.render(ax=ax1)
                plt.pause(0.1)
        
        # After game ends, update rewards
        winner = env.winner
        
        # Record win/loss/draw
        if winner == 1:  # Black wins
            win_history.append(1 if agent_plays_black else -1)
        elif winner == -1:  # White wins
            win_history.append(1 if not agent_plays_black else -1)
        else:  # Draw
            win_history.append(0)
        
        # Update final rewards if not already done
        if done and len(agent_rewards) > 0:
            if (agent_plays_black and winner == 1) or (not agent_plays_black and winner == -1):
                # Agent wins, ensure final reward is positive
                agent_rewards[-1] = 1.0
            elif winner == 0:
                # Draw
                agent_rewards[-1] = 0.2
        
        # Store trajectories in agent
        for s, a, r in zip(agent_states, agent_actions, agent_rewards):
            agent.store_transition(s, a, r)
        
        # Update policy
        agent.update_policy()
        
        # Record reward for this episode
        episode_rewards.append(total_reward)
        
        # Save current agent as opponent every save_opponent_freq episodes
        if (episode + 1) % save_opponent_freq == 0:
            historical_opponents.append(create_opponent_copy(agent))
            if should_render:
                print(f"Saved current agent as opponent version {len(historical_opponents) - 1}")
        
        # Update progress plot
        if episode > 0:
            ax2.clear()
            ax2.plot(episode_rewards, 'b-', alpha=0.3, label='Raw Reward')
            
            # Add moving average
            window = min(50, len(episode_rewards))
            if window > 1:
                moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
                ax2.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-ep Average')
            
            # Add win rate
            if len(win_history) > 10:
                win_rate = [np.mean([1 if w == 1 else 0 for w in win_history[max(0, i-50):i+1]]) * 100 
                           for i in range(len(win_history))]
                ax2.plot(win_rate, 'g-', linewidth=2, label='Win Rate (%)')
            
            ax2.set_xlabel("Episodes")
            ax2.set_ylabel("Total Reward / Win Rate (%)")
            ax2.set_title("Training Progress")
            ax2.legend(loc='upper left')
            plt.pause(0.01)
        
        # Print training info
        if should_render or episode == episodes - 1:
            print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")
            if winner == 1:
                print(f"{'Agent' if agent_plays_black else 'Opponent'} (Black) wins!")
            elif winner == -1:
                print(f"{'Agent' if not agent_plays_black else 'Opponent'} (White) wins!")
            else:
                print("Draw!")
            
            # Print win rate
            if len(win_history) > 10:
                recent_win_rate = np.mean([1 if w == 1 else 0 for w in win_history[-min(50, len(win_history)):]]) * 100
                print(f"Recent win rate: {recent_win_rate:.1f}%")
    
    print("===== Training complete =====")
    return episode_rewards


if __name__ == "__main__":
    # Create environment
    env = GomokuEnv(board_size=15)
    
    # Create agent
    agent = ContinuousPolicyGomokuAgent(
        board_size=15,
        hidden_dim=256,
        learning_rate=0.001,
        gamma=0.99,
    )
    
    # Train with self-play (only training option)
    rewards = self_play_training(env, agent, episodes=500, render_freq=50)
    
    # Test mode selection
    print("\nSelect test mode:")
    print("1. Play against random player")
    print("2. Play against human")
    
    try:
        test_mode = int(input("Enter mode (1-2): "))
    except ValueError:
        test_mode = 1  # Default to random player if input is invalid
    
    # Test based on selected mode
    if test_mode == 2:
        # Human vs Agent
        test_agent_vs_human(env, agent)
    else:
        # Agent vs Random
        test_agent_vs_random(env, agent, games=20)
    
    # Plot training reward curve
    plt.figure(figsize=(10, 6))
    plt.plot(agent.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.show() 