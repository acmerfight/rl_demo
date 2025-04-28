import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import  Dict, Tuple, Optional, Any

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
        
        ax.set_title('Gomoku')
        plt.pause(0.01)
        
        return ax