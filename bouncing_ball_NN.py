import matplotlib.pyplot as plt
import bouncing_env as my_env
import plotting_functions as my_plots
import agent_nn as my_agent

# Initialize environment
env = my_env.BouncingBallEnv()
plotObj = my_plots.plotDynamic()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = my_agent.DQNAgent(state_size, action_size)
done = False
batch_size = 32
num_episodes = 5 #1000
final_reward = []

for e in range(num_episodes):
    state = env.reset()
    total_reward = 0
    previous_action = None
    
    for time in range(13000):
        action = agent.act(state[:3])
        next_state, reward, done, _ = env.step(action, previous_action)
        # I avoid saving the time and the constant, because they are only
        # relevant for malbrid, but not for the rl-agent
        agent.memorize(state[:3], action, reward, next_state[:3], done)
        state = next_state
        total_reward += reward
        previous_action = action
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # print("time ", time)
        # print(action, " ---", state, "----", total_reward, "---", done)
    
    print(f"episode: {e}/{num_episodes}, score: {total_reward}, e: {agent.epsilon:.2}")
    final_reward.append(total_reward)
    plotObj.small_plot(env.trace_time, env.trace_states_ball, env.trace_states_paddle) # plot the dynamic of the ball
    # print(env.trace_time)
    plotObj.saveGraph("dynamicff_NN_"+str(e)+".pdf")

print("Simulation done!")

plotObj.show()
# Plot the rewards
plt.plot(final_reward)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training')
plt.show()

env.close()