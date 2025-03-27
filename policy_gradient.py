import numpy as np
from time import sleep
from typing import List, Dict, Tuple, Optional, Any

# 迷宫环境定义 - 与Q-learning示例相同
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
            ['D', ' ', ' ']
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

# 基于策略的智能体（Policy Gradient）
class PolicyGradientAgent:
    def __init__(self, states: List[str], actions: List[str], learning_rate: float=0.01, gamma: float=0.9):
        """
        初始化策略梯度智能体
        
        参数:
        - states: 状态列表
        - actions: 动作列表
        - learning_rate: 学习率
        - gamma: 折扣因子，控制对未来奖励的重视程度
        """
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # 策略参数 (state -> action -> preference)
        self.policy_params = {s: {a: 0.0 for a in actions} for s in states}
        
        # 训练过程中的数据记录
        self.rewards_history = []  # 每轮总奖励
        
        # 当前episode的轨迹
        # 状态序列：s₁, s₂, ..., sₙ
        self.states_history = []  # 状态历史
        # 动作序列：a₁, a₂, ..., aₙ
        self.actions_history = []  # 动作历史
        # 奖励序列：r₁, r₂, ..., rₙ
        self.rewards_history_episode = []  # 奖励历史
        
    def get_action_probs(self, state: str, available_actions: List[str]) -> Dict[str, float]:
        """
        计算当前状态下各动作的概率分布
        使用softmax函数将偏好值转换为概率
        """
        # 只考虑可用的动作
        prefs = np.array([self.policy_params[state][a] for a in available_actions])
        
        # Softmax函数
        exp_prefs = np.exp(prefs - np.max(prefs))  # 减去最大值以提高数值稳定性
        probs = exp_prefs / np.sum(exp_prefs)
        
        return {a: p for a, p in zip(available_actions, probs)}
    
    def choose_action(self, state: str, available_actions: List[str]) -> str:
        """
        基于当前策略选择动作
        """
        # 获取动作概率
        action_probs = self.get_action_probs(state, available_actions)
        
        # 根据概率随机选择动作
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def store_transition(self, state: str, action: str, reward: float) -> None:
        """
        存储轨迹中的一步
        """
        self.states_history.append(state)
        self.actions_history.append(action)
        self.rewards_history_episode.append(reward)
    
    def update_policy(self) -> None:
        """
        基于REINFORCE算法更新策略参数
        使用完整轨迹的回报来更新参数
        """
        
        # 奖励 reward 是强化学习中最基础的即时反馈信号：
        # 即时性：奖励是智能体在特定状态下执行特定动作后，环境立即给予的反馈
        # 单步值：表示单个时间步的得失，通常记为r_t
        # 环境决定：由环境直接给出，是问题定义的一部分
        # 局部信息：只反映当前动作的好坏，不包含长期影响
        
        
        # 回报(Return)的定义
        # 回报是从当前时间步到序列结束的所有奖励的累积和：
        # 累积性：回报考虑的是整个序列的累积奖励
        # 长期价值：反映了一个动作的长期影响
        # 计算得到：需要通过奖励序列计算
        # 全局视角：考虑了未来所有可能获得的奖励
        
        # 计算每一步的回报（从当前步骤到结束的累积奖励）
        returns = []
        # gain 表示从当前时间步到结束的累积奖励
        G = 0
        
        # 从后向前计算回报,回报是从当前时间步到序列结束的所有奖励的累积和（考虑折扣因子）。
        # 这意味着每个时间步的回报是当前奖励加上未来奖励的折扣累积和。
        # 从序列末尾开始向前计算：
        # 最后一步的回报 = 最后一步的奖励
        # 倒数第二步的回报 = 倒数第二步的奖励 + γ×最后一步的回报
        # 依此类推...
        # 这样每一步的回报G反映了该步骤对未来总奖励的贡献。
        for r in reversed(self.rewards_history_episode):
            G = r + self.gamma * G  # 加入折扣因子
            returns.insert(0, G)

        
        # 为每个状态-动作对更新策略参数
        for t in range(len(self.states_history)):
            state = self.states_history[t]
            action = self.actions_history[t]
            G = returns[t]
            
            # 获取当前状态下的可用动作 - 修正获取可用动作的方式
            # 直接检查该状态在环境中可用的动作，而不是检查policy_params
            available_actions = []
            if state in self.states and state != 'D':  # 不是终止状态
                for a in self.actions:
                    if a in self.policy_params[state]:
                        available_actions.append(a)
            
            # 如果没有可用动作，跳过
            if not available_actions:
                continue
            
            # 获取动作概率
            action_probs = self.get_action_probs(state, available_actions)
            
            # 更新参数：增加选中动作的概率（如果回报为正）
            for a in available_actions:
                # 如果是选中的动作，梯度为(1-P(a|s))
                # 如果不是，梯度为(-P(a|s))
                if a == action:
                    gradient = 1 - action_probs[a]
                else:
                    gradient = -action_probs[a]
                
                # 参数更新（上升梯度，因为我们要最大化回报）
                self.policy_params[state][a] += self.learning_rate * gradient * G
        
        # 记录这一episode的总回报
        total_reward = sum(self.rewards_history_episode)
        self.rewards_history.append(total_reward)
        
        # 清空当前episode的历史
        self.states_history = []
        self.actions_history = []
        self.rewards_history_episode = []

# 打印策略    
def print_policy(agent, env):
    """美观地打印当前策略"""
    print("\n当前策略:")
    print("-" * 50)
    print(f"{'状态':<10}{'动作':<10}{'概率':<10}")
    print("-" * 50)
    
    for state in env.states:
        # 跳过终止状态
        if state == env.terminal_state:
            continue
            
        # 获取可用动作
        available_actions = env.get_available_actions(state)
        if not available_actions:
            continue
            
        # 计算动作概率
        action_probs = agent.get_action_probs(state, available_actions)
        
        # 打印每个动作的概率
        for i, (action, prob) in enumerate(action_probs.items()):
            if i == 0:
                print(f"{state:<10}{action:<10}{prob:.4f}")
            else:
                print(f"{'':10}{action:<10}{prob:.4f}")
        
        print("-" * 50)

def explain_policy_gradient_step(state, action, reward, action_probs, gradient, policy_update):
    """解释策略梯度算法的更新步骤"""
    print("\n===== 策略梯度更新步骤说明 =====")
    print(f"状态: {state}，选择动作: {action}，获得奖励: {reward}")
    print(f"当前动作概率分布: {action_probs}")
    print(f"对于选择的动作 {action}，梯度为 {gradient:.4f}")
    print(f"策略参数更新: +{policy_update:.4f}")
    print("===============================")

# 训练函数
def train(env: MazeEnv, agent: PolicyGradientAgent, episodes: int=200, verbose: int=1, visualization_speed: float=0.5) -> None:
    """
    训练策略梯度智能体
    
    参数:
    - env: 环境对象
    - agent: 策略梯度智能体
    - episodes: 训练回合数
    - verbose: 详细程度 (0-2)
    - visualization_speed: 可视化速度 (秒)
    """
    print("===== 开始策略梯度训练 =====")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # 详细模式下显示初始迷宫状态
        if verbose >= 1 and episode < 3:
            print(f"\n回合 {episode+1} 开始:")
            env.render()
        
        # 收集轨迹
        while not done and steps < 20:  # 防止无限循环
            # 获取当前状态可用的动作
            available_actions = env.get_available_actions(state)
            
            # 选择动作
            action = agent.choose_action(state, available_actions)
            
            # 计算当前动作概率（用于解释）
            action_probs = agent.get_action_probs(state, available_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储轨迹
            agent.store_transition(state, action, reward)
            
            # 详细模式打印
            if verbose >= 2 and episode < 3:
                print(f"\n回合 {episode+1}, 步骤 {steps+1}:")
                print(f"状态: {state} -> 动作: {action} -> 新状态: {next_state}, 奖励: {reward}")
                
                # 解释当前步骤的策略选择
                # 为教学目的计算并解释梯度和更新
                gradient = 1 - action_probs[action]
                policy_update = agent.learning_rate * gradient * reward  # 简化版，仅使用即时奖励
                explain_policy_gradient_step(state, action, reward, action_probs, gradient, policy_update)
                
                # 显示迷宫状态
                env.render()
                
                # 控制显示速度
                sleep(visualization_speed)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # 更新策略
        agent.update_policy()
        
        # 每隔一段时间打印进度
        if verbose >= 1 and ((episode+1) % 20 == 0 or episode < 3):
            print(f"回合 {episode+1} 完成, 总奖励: {total_reward}, 步数: {steps}")
            if episode < 3 or (episode+1) % 50 == 0:
                print_policy(agent, env)

    print(f"\n===== 训练完成! 共{episodes}个回合 =====")

# 测试训练结果
def test_agent(env, agent):
    """测试训练好的智能体"""
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("\n===== 测试学到的策略 =====")
    env.render()
    sleep(1)
    
    # 路径记录
    path = [state]
    actions_taken = []
    
    while not done and steps < 10:
        # 获取当前状态可用的动作
        available_actions = env.get_available_actions(state)
        
        # 根据策略选择最优动作（不是随机采样）
        action_probs = agent.get_action_probs(state, available_actions)
        action = max(action_probs, key=action_probs.get)
        
        # 记录动作
        actions_taken.append(action)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        # 记录路径
        path.append(next_state)
        
        # 显示当前状态
        print(f"状态: {state} -> 动作: {action} (概率: {action_probs[action]:.4f}) -> 新状态: {next_state}, 奖励: {reward}")
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
def explain_policy_gradient():
    """解释策略梯度算法的基本原理"""
    print("\n===== 策略梯度算法解释 =====")
    print("策略梯度是一种直接学习策略函数的强化学习方法，不需要通过值函数间接得到策略。")
    print("\n核心思想:")
    print("1. 直接参数化策略 π(a|s;θ)，表示在状态s下选择动作a的概率")
    print("2. 通过梯度上升最大化期望回报")
    print("3. 策略参数θ沿着提高期望回报的方向更新")
    
    print("\n策略梯度定理:")
    print("∇θJ(θ) = E[∇θ log π(a|s;θ) · G]")
    print("其中:")
    print("- J(θ): 期望回报")
    print("- π(a|s;θ): 在状态s下选择动作a的概率")
    print("- G: 回报（从当前步骤到结束的累积奖励）")
    
    print("\nREINFORCE算法步骤:")
    print("1. 初始化策略参数θ")
    print("2. 对每个回合:")
    print("   a. 生成完整轨迹 τ = (s₀,a₀,r₁,s₁,...,sT)")
    print("   b. 对轨迹中的每一步t:")
    print("      i. 计算回报Gt")
    print("      ii. 更新参数: θ ← θ + α·∇θ log π(at|st;θ)·Gt")
    
    print("\n相比Q-learning的区别:")
    print("1. 策略梯度直接学习策略，Q-learning学习动作值函数")
    print("2. 策略梯度是策略迭代，Q-learning是值迭代")
    print("3. 策略梯度通常用于连续动作空间，Q-learning更适合离散动作空间")
    print("4. 策略梯度能学习随机策略，Q-learning通常产生确定性策略")
    print("===============================")

# 运行示例
if __name__ == "__main__":
    # 解释策略梯度算法
    explain_policy_gradient()
    
    # 初始化环境和智能体
    env = MazeEnv()
    actions = ['left', 'right', 'up', 'down']
    
    # 创建智能体
    agent = PolicyGradientAgent(
        states=env.states,
        actions=actions,
        learning_rate=0.1,
        gamma=0.9    # 折扣因子
    )
    
    # 训练智能体
    train(env, agent, episodes=500, verbose=2, visualization_speed=0.5)
    
    # 显示最终策略
    print("\n最终学习的策略：")
    print_policy(agent, env)
    
    # 测试智能体性能
    test_agent(env, agent)
    
    # 注意事项和教学建议
    print("\n===== 教学建议 =====")
    print("1. 比较策略梯度和Q-learning在相同问题上的表现")
    print("2. 分析策略梯度方法学习到的随机性策略的优缺点")
    print("3. 尝试调整学习率观察其对收敛的影响")
    print("4. 讨论基于值的方法和基于策略的方法的权衡")
    print("5. 探索在更复杂环境中策略梯度的表现")
    print("===============================") 