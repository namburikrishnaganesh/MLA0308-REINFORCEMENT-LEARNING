import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.95  
alpha = 0.001 
epsilon = 0.2  
def build_network(input_shape, output_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(output_shape, activation='softmax')
    ])
    return model
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.policy_network = build_network([state_size], action_size)
        self.old_policy_network = build_network([state_size], action_size)
        self.policy_network.compile(optimizer=Adam(learning_rate=alpha))
        self.gamma = gamma
        self.epsilon = epsilon
    def train_policy_network(self, states, actions, advantages, old_probs):
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)
            action_probs = tf.math.reduce_sum(actions * action_probs, axis=1)
            old_action_probs = self.old_policy_network(states, training=True)
            old_action_probs = tf.math.reduce_sum(actions * old_action_probs, axis=1)
            
            ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
            surrogate_loss = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate_loss) 
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.policy_network.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
    def update_old_policy_network(self):
        self.old_policy_network.set_weights(self.policy_network.get_weights())
def train_ppo_agent(env, agent, episodes, max_steps):
    for episode in range(episodes):
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        next_values = []
        state = env.reset()
        for step in range(max_steps):
            state = state.reshape([1, -1])
            action_probs = agent.policy_network.predict(state)[0]
            action = np.random.choice(action_size, p=action_probs)
            next_state, reward, done, _ = env.step(action)
            value = agent.old_policy_network.predict(state)[0][action]
            next_value = agent.old_policy_network.predict(next_state.reshape([1, -1]))[0][action]
            states.append(state)
            action_onehot = np.zeros(action_size)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(1 - done)
            values.append(value)
            next_values.append(next_value)
            state = next_state
            if done:
                break
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        next_values = np.array(next_values)
        deltas = rewards + agent.gamma * next_values * dones - values
        advantages = np.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + agent.gamma * advantage * dones[t]
            advantages[t] = advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)       
        agent.train_policy_network(states, actions, advantages, values)
        agent.update_old_policy_network()
        total_reward = sum(rewards)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
agent = PPOAgent(state_size, action_size)
train_ppo_agent(env, agent, episodes=10, max_steps=1)
