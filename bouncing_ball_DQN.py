# import pygame
import numpy as np
import matplotlib.pyplot as plt
import random

import time

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    # not possible to use normal q_learning in continuous space.  Instead, I'll use function 
    # approximation, such as the discretization of the environment.
    # I create 1000 bins
    q_table = np.zeros((1001, env.action_space.n)) 

    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        done = False
        old_action = None
        discret_state = int(round((state[0])*100, 0))
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
                # while old_action == action and action in (1, 2):
                #     action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[discret_state])  # Exploit learned values
            
            next_state, reward, done, _ = env.step(action, old_action)
            
            old_value = q_table[discret_state][action]
            next_max = np.max(q_table[int(round((next_state[0])*100, 0))])
            
            # Q-learning formula
            q_table[discret_state][action] = old_value + alpha * (reward + gamma * next_max - old_value)
            
            state = next_state
            total_reward += reward
            old_action = action
            
        rewards.append(total_reward)

        # plot the dynamic of the ball and for the paddle
        plotObj.small_plot(env.trace_time, env.trace_states_ball, env.trace_states_paddle) 
        plotObj.saveGraph("dynamic_"+str(episode)+".pdf")
    
    return q_table, rewards

import bouncing_env as my_env
import plotting_functions as my_plots

# Initialize environment
env = my_env.BouncingBallEnv()
plotObj = my_plots.plotDynamic()

# Train the agent
num_episodes = 5
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

q_table, rewards = q_learning(env, num_episodes, alpha, gamma, epsilon)
print("Simulation done!")
plotObj.show()

print(rewards)
# print(q_table)

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Training')
plt.show()

env.close()
