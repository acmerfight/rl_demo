import numpy as np
from time import sleep
from typing import List, Dict, Tuple, Optional, Any

# 迷宫环境定义
class MazeEnv:
    def __init__(self):
        self.states = ['A', 'B', 'C', 'D']  # 状态空间
        self.terminal_state = 'D'          # 终止状态（目标）
        self.trap_state = 'C'              # 陷阱状态
        self.current_state = 'A'
        
        # 状态转移表 (state: {action: next_state})
        self.transitions = {
            'A': {'right': 'B', 'down': 'D'},
            'B': {'left': 'A', 'right': 'C', 'down': 'D'},
            'C': {'left': 'B', 'down': 'D'},
            'D': {}  # 终止状态没有转移
        }
        
        # 迷宫布局（用于可视化）
        self.maze_layout = [
            ['A', 'B', 'C'],
            ['D', 'D', 'D']
        ]
        
    def reset(self) -> str:
        """重置环境到初始状态A"""
        self.current_state = 'A'
        return self.current_state
    
    def step(self, action: str) -> Tuple[str, int, bool]:
        """执行动作，返回新状态和奖励"""
        if self.current_state == self.terminal_state:
            return self.current_state, 0, True
        
        # 获取该状态下可用的动作
        available_actions = self.transitions[self.current_state]
        
        # 如果动作不可用，保持原状态
        next_state = available_actions.get(action, self.current_state)
        reward = self._get_reward(next_state)
        done = (next_state == self.terminal_state)
        self.current_state = next_state
        return next_state, reward, done
    
    def _get_reward(self, state: str) -> int:
        """奖励函数"""
        if state == self.terminal_state:
            return 100   # 到达终点
        elif state == self.trap_state:
            return -50   # 踩到陷阱
        else:
            return -1    # 普通移动成本
            
    def get_available_actions(self, state: str) -> List[str]:
        """获取特定状态下可用的动作"""
        return list(self.transitions[state].keys())
    
    def render(self) -> None:
        """可视化当前迷宫状态"""
        maze_vis = [
            ['[ ]', '[ ]', '[ ]'],
            ['[ ]', '   ', '   ']
        ]
        
        # 标记特殊状态
        for i in range(len(self.maze_layout)):
            for j in range(len(self.maze_layout[i])):
                if self.maze_layout[i][j] == self.terminal_state:
                    maze_vis[i][j] = '[D]'  # 目标
                elif self.maze_layout[i][j] == self.trap_state:
                    maze_vis[i][j] = '[C]'  # 陷阱
                elif self.maze_layout[i][j] == self.current_state:
                    maze_vis[i][j] = '[*]'  # 当前位置
                elif self.maze_layout[i][j] in self.states:
                    maze_vis[i][j] = f'[{self.maze_layout[i][j]}]'
        
        # 打印迷宫
        print("迷宫状态:")
        print("+---+---+---+")
        for row in maze_vis:
            print("|" + "|".join(row) + "|")
            print("+---+---+---+")

# Q-learning智能体
class QLearningAgent:
    def __init__(self, states: List[str], actions: List[str], alpha: float=0.5, gamma: float=0.9, epsilon: float=0.1):
        """
        初始化Q-learning智能体
        
        参数:
        - states: 状态列表
        - actions: 动作列表
        - alpha: 学习率 (0-1) - 控制新信息的接受程度
        - gamma: 折扣因子 (0-1) - 控制对未来奖励的重视程度
        - epsilon: 探索概率 (0-1) - 控制探索vs利用的平衡
        """
        self.q_table: Dict[str, Dict[str, float]] = {s: {a: 0.0 for a in actions} for s in states}  # 初始化Q表全为0
        self.alpha = alpha    # 学习率
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon# 探索概率
        self.actions = actions
        self.rewards_history: List[float] = []  # 记录每个episode的总奖励，用于可视化学习进度
        
    def choose_action(self, state: str, available_actions: Optional[List[str]]=None) -> str:
        """
        ε-greedy策略选择动作
        
        有epsilon的概率随机探索，(1-epsilon)的概率选择当前最优动作
        """
        # 如果提供了可用动作列表，就只从这些动作中选择
        action_space = available_actions if available_actions else self.actions
        
        if np.random.uniform(0, 1) < self.epsilon:
            # 探索: 随机选择一个动作
            return np.random.choice(action_space)
        else:
            # 利用: 选择当前状态下Q值最高的动作
            # 只考虑可用的动作
            if available_actions:
                return max(available_actions, key=lambda a: self.q_table[state][a])
            return max(self.q_table[state], key=lambda a: self.q_table[state][a])
    
    def update_q_table(self, state: str, action: str, reward: int, next_state: str, available_next_actions: Optional[List[str]]=None) -> None:
        """
        Q值更新公式: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        
        参数:
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - next_state: 转移到的下一个状态
        - available_next_actions: 下一个状态可用的动作列表
        """
        # 当前Q值
        old_q = self.q_table[state][action]
        
        # 下一状态的最大Q值 (如果是终止状态则为0)
        if next_state and next_state != 'D':
            if available_next_actions:
                # 只考虑可用的动作
                max_future_q = max(self.q_table[next_state][a] for a in available_next_actions)
            else:
                max_future_q = max(self.q_table[next_state].values())
        else:
            max_future_q = 0
            
        # Q-learning更新公式
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state][action] = new_q

# 训练与可视化相关的函数
def print_q_table(agent, actions):
    """美观地打印Q表"""
    print("\n当前Q表:")
    # 打印表头
    header = f"{'状态/动作':<10}"
    for a in actions:
        header += f"{a:<10}"
    print(header)
    print("-" * (10 + 10 * len(actions)))
    
    # 打印每个状态的Q值
    for s in agent.q_table:
        row = f"{s:<10}"
        for a in actions:
            value = agent.q_table[s][a]
            row += f"{value:>9.1f} "
        print(row)
    print("-" * (10 + 10 * len(actions)))
    print()

def explain_q_learning_step(state, action, next_state, reward, old_q, new_q, epsilon, alpha, gamma, max_future_q):
    """解释Q-learning算法的每一步更新"""
    print("\n===== Q-learning更新步骤说明 =====")
    print(f"当前状态: {state}，选择动作: {action}，新状态: {next_state}，获得奖励: {reward}")
    print(f"探索率 ε = {epsilon}，学习率 α = {alpha}，折扣因子 γ = {gamma}")
    print(f"当前Q值 Q({state},{action}) = {old_q:.1f}")
    print(f"下一状态最大Q值 max Q({next_state},a) = {max_future_q:.1f}")
    print(f"Q值更新: Q({state},{action}) ← {old_q:.1f} + {alpha} × [{reward} + {gamma} × {max_future_q:.1f} - {old_q:.1f}]")
    print(f"新的Q值: Q({state},{action}) = {new_q:.1f}")
    print("===============================")

# 训练过程
def train(env: MazeEnv, agent: QLearningAgent, episodes: int=100, verbose: int=1, visualization_speed: float=0.5) -> None:
    """
    训练Q-learning智能体
    
    参数:
    - env: 环境对象
    - agent: Q-learning智能体
    - episodes: 训练回合数
    - verbose: 详细程度 (0-2)
    - visualization_speed: 可视化速度 (秒)
    """
    print("===== 开始Q-learning训练 =====")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # 详细模式下显示初始迷宫状态
        if verbose >= 1 and episode < 3:
            print(f"\n回合 {episode+1} 开始:")
            env.render()
        
        while not done and steps < 20:  # 防止无限循环
            # 获取当前状态可用的动作
            available_actions = env.get_available_actions(state)
            
            # 选择动作
            action = agent.choose_action(state, available_actions)
            
            # 记录旧的Q值（用于解释）
            old_q = agent.q_table[state][action]
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 获取下一状态可用的动作
            next_available_actions = [] if done else env.get_available_actions(next_state)
            
            # 更新Q表
            # 计算下一状态最大Q值（用于解释）
            if next_state and next_state != 'D':
                if next_available_actions:
                    max_future_q = max(agent.q_table[next_state][a] for a in next_available_actions)
                else:
                    max_future_q = 0
            else:
                max_future_q = 0
                
            # 更新Q表
            agent.update_q_table(state, action, reward, next_state, next_available_actions)
            
            # 获取新的Q值（用于解释）
            new_q = agent.q_table[state][action]
            
            # 详细模式打印
            if verbose >= 2 and episode < 3:
                print(f"\n回合 {episode+1}, 步骤 {steps+1}:")
                print(f"状态: {state} -> 动作: {action} -> 新状态: {next_state}, 奖励: {reward}")
                
                # 解释Q值更新
                explain_q_learning_step(
                    state, action, next_state, reward,
                    old_q, new_q, agent.epsilon, agent.alpha, agent.gamma, max_future_q
                )
                
                # 打印Q表
                print_q_table(agent, agent.actions)
                
                # 显示迷宫状态
                env.render()
                
                # 控制显示速度
                sleep(visualization_speed)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # 记录本回合的总奖励
        agent.rewards_history.append(total_reward)
        
        # 每隔一段时间打印进度
        if verbose >= 1 and ((episode+1) % 10 == 0 or episode < 3):
            print(f"回合 {episode+1} 完成, 总奖励: {total_reward}, 步数: {steps}")
            if episode < 3:
                print_q_table(agent, agent.actions)

    print(f"\n===== 训练完成! 共{episodes}个回合 =====")

# 测试训练结果
def test_agent(env, agent):
    """测试训练好的智能体"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("\n===== 测试最优路径 =====")
    env.render()
    sleep(1)
    
    # 路径记录
    path = [state]
    actions_taken = []
    
    while not done and steps < 10:
        # 获取当前状态可用的动作
        available_actions = env.get_available_actions(state)
        
        # 选择最优动作 (不再探索)
        action = max(available_actions, key=lambda a: agent.q_table[state][a])
        
        # 记录动作
        actions_taken.append(action)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # 记录路径
        path.append(next_state)
        
        # 显示当前状态
        print(f"状态: {state} -> 动作: {action} -> 新状态: {next_state}, 奖励: {reward}")
        env.render()
        sleep(1)
        
        state = next_state
        steps += 1
    
    # 打印总结
    print(f"\n路径: {' -> '.join(path)}")
    print(f"动作: {' -> '.join(actions_taken)}")
    print(f"总奖励: {total_reward}, 步数: {steps}")
    
    return path, actions_taken, total_reward

# 教学辅助函数
def explain_q_learning():
    """解释Q-learning算法的基本原理"""
    print("\n===== Q-learning算法解释 =====")
    print("Q-learning是一种无模型（model-free）的强化学习算法，用于学习最优策略。")
    print("\n核心思想:")
    print("1. 维护一个Q表，记录每个状态-动作对的预期累积奖励")
    print("2. 通过与环境交互来更新Q表")
    print("3. 使用ε-greedy策略平衡探索与利用")
    
    print("\nQ值更新公式:")
    print("Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]")
    print("其中:")
    print("- Q(s,a): 状态s下采取动作a的当前Q值")
    print("- α: 学习率，控制新信息的接受程度")
    print("- r: 当前获得的奖励")
    print("- γ: 折扣因子，控制对未来奖励的重视程度")
    print("- max Q(s',a'): 下一状态s'下所有可能动作的最大Q值")
    
    print("\n关键参数:")
    print("- 学习率(α): 控制Q值更新的步长")
    print("- 折扣因子(γ): 控制对未来奖励的重视程度")
    print("- 探索率(ε): 控制探索新动作的概率")
    
    print("\n算法步骤:")
    print("1. 初始化Q表为零")
    print("2. 对每个回合:")
    print("   a. 选择动作（使用ε-greedy策略）")
    print("   b. 执行动作，获得奖励和新状态")
    print("   c. 更新Q值")
    print("   d. 转移到新状态")
    print("   e. 如果达到终止状态，开始新回合")
    print("===============================")

# 运行示例
if __name__ == "__main__":
    # 解释Q-learning算法
    explain_q_learning()
    
    # 初始化环境和智能体
    env = MazeEnv()
    actions = ['left', 'right', 'up', 'down']
    
    # 创建智能体
    agent = QLearningAgent(
        states=env.states,
        actions=actions,
        alpha=0.5,    # 学习率
        gamma=0.9,    # 折扣因子
        epsilon=0.1   # 探索概率
    )
    
    # 训练智能体
    # verbose=2表示详细输出，包括每步Q值更新
    train(env, agent, episodes=100000, verbose=2, visualization_speed=0.5)
    
    # 显示最终Q表
    print("\n最终Q表：")
    print_q_table(agent, actions)
    
    
    # 测试智能体性能
    test_agent(env, agent)
    
    # 注意事项和教学建议
    print("\n===== 教学建议 =====")
    print("1. 尝试调整参数(α, γ, ε)观察学习效果的变化")
    print("2. 修改迷宫布局和奖励函数，观察智能体的适应性")
    print("3. 比较不同初始Q值的影响")
    print("4. 讨论探索-利用权衡的重要性")
    print("===============================")

    
    
    
    

# 当前Q表:
# 状态/动作     left      right     up        down      
# --------------------------------------------------
# A               0.0      89.0       0.0     100.0 
# B              89.0      40.0       0.0     100.0 
# C              88.9       0.0       0.0     100.0 
# D               0.0       0.0       0.0       0.0 
# --------------------------------------------------

# 环境分析
# 首先回顾一下迷宫环境：
# 4个状态：A, B, C, D（D是终点，C是陷阱）
# 布局：A B C（上排）/ D D D（下排）
# 奖励设置：到达D得+100分，踩到C得-50分，普通移动-1分
# 折扣因子γ=0.9
# 状态A的Q值分析
# A→down(100.0)：直接到达终点D，获得100分奖励，是最优路径
# A→right(89.0)：移动到B，收到-1分，然后从B走down到D获得100分。计算：-1 + 0.9×100 = 89
# A→left/up(0.0)：这些动作在A点不可用，所以值为0
# 状态B的Q值分析
# B→down(100.0)：直接到达终点D，获得100分奖励，是最优路径
# B→left(89.0)：移动回A，收到-1分，然后可从A走down到D。计算：-1 + 0.9×100 = 89
# B→right(40.0)：移动到陷阱C，收到-50分，然后可从C走down到D。计算：-50 + 0.9×100 = 40
# B→up(0.0)：在B点不可用
# 状态C的Q值分析
# C→down(100.0)：直接到达终点D，获得100分奖励
# C→left(88.9)：移动回B，收到-1分，然后从B走down到D。计算：-1 + 0.9×100 = 89（有轻微误差）
# C→right/up(0.0)：在C点不可用
# 状态D的Q值分析
# 所有动作值为0：D是终点状态，没有后续动作可执行
# 结论
# 这个Q表正确地反映了最优策略：
# 在A状态：最好直接向下走到D
# 在B状态：最好直接向下走到D
# 在C状态：最好直接向下走到D
# Q值大小关系合理反映了到达目标的步数和途中惩罚。这表明智能体已成功学习到了环境的最优决策策略。