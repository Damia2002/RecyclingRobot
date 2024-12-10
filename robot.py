import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class RecyclingRobot:
    """
    Solves the Recycling Robot MDP using epsilon-greedy Q-learning.
    """

    def __init__(self, alpha, gamma, epsilon, episodes, seed, prob_alpha, prob_beta, r_search, r_wait):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.seed = seed
        self.prob_alpha = prob_alpha
        self.prob_beta = prob_beta
        self.r_search = r_search
        self.r_wait = r_wait
        self.episode_rewards = []
        self.states = ["high", "low"]
        self.actions = {
            "high": ["search", "wait"],
            "low": ["search", "wait", "recharge"]
        }
        self.q_table = self.initialize_q_table()
        np.random.seed(self.seed)

    def initialize_q_table(self):
        """Initialize the Q-table with zeros."""
        return {state: {action: 0 for action in self.actions[state]} for state in self.states}

    def get_next_state_and_reward(self, state, action):
        """
        Simulates the environment's response to the robot's action.
        """
        if state == "high":
            if action == "search":
                if np.random.rand() < self.prob_alpha:  
                    return "high", self.r_search  
                else:
                    return "low", self.r_search  
            elif action == "wait":
                return "high", self.r_wait  
            
        elif state == "low":
                if action == "search":
                    if np.random.rand() < self.prob_beta:  
                        return "low", self.r_search 
                    else: 
                        return "high", -3  # After saving, it returns to high
                elif action == "wait":
                    return "low", self.r_wait
                elif action == "recharge":
                    return "high", 0  

        raise ValueError("Invalid state-action pair.")

    def epsilon_greedy(self, state):
        """
        Chooses an action using the epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions[state])  # Explore
        return max(self.q_table[state], key=self.q_table[state].get)  # Exploit

    def train(self):
        """
        Trains the robot using Q-learning.
        """
        for _ in tqdm(range(self.episodes)):
            state = "high"  # Start in high energy
            episode_reward = 0
            while True:
                self.episode_rewards.append(episode_reward)
                action = self.epsilon_greedy(state)
                next_state, reward = self.get_next_state_and_reward(state, action)
                episode_reward += reward

                # Update Q-value
                max_next_q = max(self.q_table[next_state].values())
                self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * max_next_q - self.q_table[state][action]
                )

                # Transition to the next state
                state = next_state

                # Terminate if rescue occurred
                if reward == -3:
                    break
        print("Average reward per episode:", np.mean(self.episode_rewards))


    def display_q_table(self):
        """Displays the Q-table as a DataFrame."""
        df = pd.DataFrame(self.q_table).T
        print(df)

    def plot_q_values(self):
        """
        Plots the Q-values for each action in each state as bar plots in two subplots.
        """
        states = ["high", "low"]
        # Collect all unique actions across states and sort them for consistent ordering
        actions = sorted(set(action for state in states for action in self.q_table[state].keys()))

        # Data for the bar plot
        q_values_high = [self.q_table["high"].get(action, 0) for action in actions]
        q_values_low = [self.q_table["low"].get(action, 0) for action in actions]

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

        # Bar plot for "high" state
        axes[0].bar(actions, q_values_high, color=plt.cm.tab10.colors[:len(actions)])
        axes[0].set_title("State: High")
        axes[0].set_xlabel("Action")
        axes[0].set_ylabel("Q-value")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Bar plot for "low" state
        axes[1].bar(actions, q_values_low, color=plt.cm.tab10.colors[:len(actions)])
        axes[1].set_title("State: Low")
        axes[1].set_xlabel("Action")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        # Overall figure adjustments
        fig.suptitle("Q-values by State and Action", fontsize=16)
        plt.tight_layout()
        plt.show()