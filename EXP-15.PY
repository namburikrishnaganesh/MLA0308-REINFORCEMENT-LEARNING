import numpy as np
class MarioEnvironment:
    def __init__(self):
        self.grid_size = (5, 5)  
        self.agent_position = (0, 0)  
        self.goal_position = (4, 4)  
        self.obstacle_positions = [(1, 1), (2, 2), (3, 3)] 
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']  
        self.num_actions = len(self.actions)
        self.Q_values = np.zeros((self.grid_size[0], self.grid_size[1], self.num_actions))  # Q-values initialized to 0
        self.alpha = 0.1  
        self.gamma = 0.9  
        self.epsilon = 0.1  
    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
    def get_reward(self, position):
        if position == self.goal_position:
            return 10  
        elif position in self.obstacle_positions:
            return -5  
        else:
            return -1 
    def update_agent_position(self, action):
        if action == 'UP':
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 'DOWN':
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 'LEFT':
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 'RIGHT':
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        else:
            return
        if self.is_valid_position(new_position):
            self.agent_position = new_position
    def q_learning(self, num_episodes):
        for episode in range(num_episodes):
            current_state = self.agent_position
            total_reward = 0 
            while current_state != self.goal_position:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.actions)
                else:
                    action = self.actions[np.argmax(self.Q_values[current_state[0], current_state[1]])]
                self.update_agent_position(action)
                new_state = self.agent_position
                reward = self.get_reward(new_state)
                total_reward += reward  
                self.Q_values[current_state[0], current_state[1], self.actions.index(action)] += \
                    self.alpha * (reward + self.gamma * np.max(self.Q_values[new_state[0], new_state[1]]) -
                                  self.Q_values[current_state[0], current_state[1], self.actions.index(action)])
                current_state = new_state
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
            self.agent_position = (0, 0)
env = MarioEnvironment()
env.q_learning(num_episodes=30)
