import numpy as np
import random
from movements import *

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99):
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table as a dictionary of dictionaries
        # E.g. Q["neutral"]["punch"] = 0.0
        self.Q = {state: {action: 0.0 for action in actions} for state in states}

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Choose the action with the highest Q-value in this state
            state_actions = self.Q[state]
            return max(state_actions, key=state_actions.get)

    def update_Q(self, state, action, reward, next_state):
        # Max Q value for the next state
        max_next = max(self.Q[next_state].values())

        # Current Q
        current_Q = self.Q[state][action]

        # Q-learning formula
        new_Q = current_Q + self.alpha * (reward + self.gamma * max_next - current_Q)
        self.Q[state][action] = new_Q

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000):
    # Create environment and agent
    env = WrestlingEnv()
    print("env", env)
    agent = QLearningAgent(STATES, ACTIONS, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99)

    for ep in range(episodes):
        # Reset the environment for each episode (for real environment you might do env.reset())
        env.current_state = random.choice(STATES)
        state = env.current_state
        done = False

        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take the action in environment
            next_state, reward, done = env.step(action)

            # Update Q
            agent.update_Q(state, action, reward, next_state)

            # Move to next state
            state = next_state

        # Decay exploration
        agent.decay_epsilon()

    return agent

if __name__ == "__main__":
    trained_agent = train_agent(episodes=500)
    # Let's see what the Q-table looks like
    for st in trained_agent.Q:
        print(f"State: {st}")
        for act in trained_agent.Q[st]:
            val = trained_agent.Q[st][act]
            print(f"  Action: {act}, Q-value: {val:.2f}")
        print()
