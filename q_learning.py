import numpy as np

# 迷宫环境定义
class MazeEnv:
    def __init__(self):
        self.states = ['A', 'B', 'C', 'D']  # 状态空间
        self.terminal_state = 'D'          # 终止状态
        self.trap_state = 'C'              # 陷阱状态
        self.current_state = 'A'
        
        # 状态转移表 (state: {action: next_state})
        self.transitions = {
            'A': {'right': 'B', 'down': 'D'},
            'B': {'left': 'A', 'right': 'C', 'down': 'D'},
            'C': {'left': 'B', 'down': 'D'},
            'D': {}  # 终止状态没有转移
        }
        
    def reset(self):
        """重置环境到初始状态A"""
        self.current_state = 'A'
        return self.current_state
    
    def step(self, action):
        """执行动作，返回新状态和奖励"""
        if self.current_state == self.terminal_state:
            return self.current_state, 0, True
        
        next_state = self.transitions[self.current_state].get(action, self.current_state)
        reward = self._get_reward(next_state)
        done = (next_state == self.terminal_state)
        self.current_state = next_state
        return next_state, reward, done
    
    def _get_reward(self, state):
        """奖励函数"""
        if state == self.terminal_state:
            return 100   # 到达终点
        elif state == self.trap_state:
            return -50   # 踩到陷阱
        else:
            return -1    # 普通移动成本

# Q-learning智能体
class QLearningAgent:
    def __init__(self, states, actions, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {s: {a: 0 for a in actions} for s in states}  # 初始化Q表全为0
        self.alpha = alpha    # 学习率
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon# 探索概率
        self.actions = actions
        
    def choose_action(self, state):
        """ε-greedy策略选择动作"""
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)  # 探索
        else:
            return max(self.q_table[state], key=lambda action: self.q_table[state][action])  # 利用
    
    def update_q_table(self, state, action, reward, next_state):
        """Q值更新公式"""
        old_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values()) if next_state else 0
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state][action] = new_q

# 训练过程可视化
def train(env, agent, episodes=100):
    print("开始训练！")
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:  # 防止无限循环
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            
            # 打印每一步细节（前3个episode）
            if episode < 3:
                print(f"Episode {episode+1}, Step {steps+1}:")
                print(f"状态: {state} -> 动作: {action} -> 新状态: {next_state}, 奖励: {reward}")
                
                # 美观地打印Q表
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
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # 每10轮打印一次进度
        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}, 总奖励: {total_reward}")

# 测试训练结果
def test_agent(env, agent):
    state = env.reset()
    done = False
    print("\n测试最优路径：")
    while not done:
        action = max(agent.q_table[state], key=lambda action: agent.q_table[state][action])
        next_state, _, done = env.step(action)
        print(f"{state} --{action}-->", end=' ')
        state = next_state
    print("D（到达终点）")

# 运行示例
if __name__ == "__main__":
    # 初始化环境和智能体
    env = MazeEnv()
    actions = ['left', 'right', 'up', 'down']  # 虽然有些状态不支持所有动作
    
    agent = QLearningAgent(
        states=env.states,
        actions=actions,
        alpha=0.5,
        gamma=0.9,
        epsilon=0.1
    )
    
    # 训练100轮
    train(env, agent, episodes=100)
    
    # 显示最终Q表
    print("\n最终Q表：")
    for state in agent.q_table:
        print(f"{state}:")
        for action in agent.q_table[state]:
            print(f"  {action}: {agent.q_table[state][action]:.1f}")
    
    # 测试最优策略
    test_agent(env, agent)