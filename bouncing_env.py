
import malbrid  
import scipy
import gymnasium as gym
from gym import spaces
import random
import numpy as np
import math
import pygame


# Initialize the simulator
simulator = malbrid.LinearSystemSimulator(["x", "xspeed", "p", "t", "const"])

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
scaling_factor = 4
random.seed(1234)

# Dynamic of the environment
def get_dynamics_and_zero_crossing_functions_controller(state_name):
    ATimeCount = np.array([[0, 0, 1]])

    x = simulator.get_var("x")
    t = simulator.get_var("t")

    def nobumpUp(x):
        return "Go", np.array([x[0], 0, x[2]]), False

    def nobumpDown(x):
        return "End", np.array([x[0], 0, x[2]]), False

    def nobumpStill(x):
        return "Wait", np.array([x[0], 0, x[2]]), False

    if state_name == "Wait":
        zero_crossing_end_up = x <= 1
        return ATimeCount, [(zero_crossing_end_up, "GoUp", nobumpUp)]
    if state_name == "Go":
        zero_crossing_end_up = t >= 0.5
        return ATimeCount, [(zero_crossing_end_up, "GoDown", nobumpDown)]
    if state_name == "End":
        zero_crossing_end_up = t >= 0.5
        return ATimeCount, [(zero_crossing_end_up, "GoStill", nobumpStill)]
    else:
        raise Exception("Internal Test error:" + str(state_name))

# Dynamic of the paddle
def get_dynamics_and_zero_crossing_functions_system(state_name):
    APaddleDown = np.array([[0, 1, 0, 0], [0, 0, 0, -9.81], [0, 0, 0, -1], [0, 0, 0, 0]], dtype=np.float64)
    APaddleUp = np.array([[0, 1, 0, 0], [0, 0, 0, -9.81], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float64)
    APaddleStill = np.array([[0, 1, 0, 0], [0, 0, 0, -9.81], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64)

    x = simulator.get_var("x")
    xs = simulator.get_var("xspeed")
    p = simulator.get_var("p")

    def bumpPaddleUp(x):
        return "PaddleUp", np.array([x[0], -0.9 * x[1] + 0.9, x[2], x[3]]), False

    def bumpPaddleDown(x):
        return "PaddleDown", np.array([x[0], -0.9 * x[1] - 0.9, x[2], x[3]]), False

    def bumpPaddleStill(x):
        return "PaddleStill", np.array([x[0], -0.9 * x[1], x[2], x[3]]), False

    def nobumpPaddleToUp(x):
        return "PaddleUp", np.array([x[0], x[1], x[2], x[3]]), False

    def nobumpPaddleToDown(x):
        return "PaddleDown", np.array([x[0], x[1], x[2], x[3]]), False

    def nobumpPaddleToStill(x):
        return "PaddleStill", np.array([x[0], x[1], x[2], x[3]]), False

    if state_name == "PaddleUp":
        zero_crossing_end_up = simulator.get_true_condition()
        zero_crossing_bump = (x <= p) & (xs < 1)
        return APaddleUp, [(zero_crossing_end_up, "GoDown", nobumpPaddleToDown), (zero_crossing_bump, "bump", bumpPaddleUp)]
    if state_name == "PaddleDown":
        zero_crossing_end_up = simulator.get_true_condition()
        zero_crossing_bump = (x <= p) & (xs < -1)
        return APaddleDown, [(zero_crossing_end_up, "GoStill", nobumpPaddleToStill), (zero_crossing_bump, "bump", bumpPaddleDown)]
    if state_name == "PaddleStill":
        zero_crossing_end_up = simulator.get_true_condition()
        zero_crossing_bump = (x <= p) & (xs < 0) & (~zero_crossing_end_up)
        return APaddleStill, [(zero_crossing_end_up, "GoUp", nobumpPaddleToUp), (zero_crossing_bump, "bump", bumpPaddleStill)]
    else:
        raise Exception("Internal Test error:" + str(state_name))

product_dynamics = malbrid.compute_product_dynamics(
    [("x", 1, [0, 1]), ("xspeed", 1, [1]), 
    ("p", 1, [1]), ("t", 0, [0]), ("const", 1, [0, 1])],
    get_dynamics_and_zero_crossing_functions_controller,
    get_dynamics_and_zero_crossing_functions_system, ["GoUp", "GoDown", "GoStill"]
)

class BouncingBallEnv(gym.Env):
    def __init__(self):
        super(BouncingBallEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Three actions: up, down, stay
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, -10]), 
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )
        self.mvt = "PaddleStill"
        self.ctrl = "Wait"

        self.trace_time = []
        self.trace_states_ball = []
        self.trace_states_paddle = []
        
        self.reset()

    def reset(self):
        self.state = np.array([10.0, 0, 0, 0, 1.0])
        self.time_step = 0  


        self.trace_time = []
        self.trace_states_ball = []
        self.trace_states_paddle = []

        return self.state
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-(x-5))) # The function outputs values in [0,1]
    
    def calculateReward(self, state, action, previous_action):
        reward = 0
        threshold = 5 # reached after approximately 8 bouncing

        distance = state[0] - threshold # I only select the position of the ball
        # if distance >= 0:
        reward = 2*self.sigmoid(distance)

        # Penalize for not changing the action
        if action == previous_action and action != 0:
            reward -= 0.1  # Small penalty for repeating the same action

        return reward
    
    def step(self, action, previous_action):
        # Apply action to update the paddle position
        if action == 0:
            self.mvt = "PaddleStill"
            self.ctrl = "Wait"
        elif action == 1:
            self.mvt = "PaddleDown"
            self.ctrl = "End"
        else:
            self.mvt = "PaddleUp"
            self.ctrl = "Go"

        # Simulate one step
        simulator.simulate(product_dynamics, (self.ctrl, self.mvt), self.state, max_time= 0.01)
        # simulator.simulate(product_dynamics, (self.ctrl, self.mvt), self.state, max_time=self.time_step + 0.01)

        # Get new state
        self.state = simulator.continuous_states[-1]
        time_points = simulator.time_points[-1]
        reward = 0

        self.trace_time.append(self.time_step)
        self.trace_states_ball.append(self.state[0]) # position of the ball
        self.trace_states_paddle.append(self.state[2]) # position of the paddle
        
        # Calculate reward
        if self.state[2] < 0:
            self.state[2] = 0
            reward = -0.01  # Penalize the agent when it tries to move the paddle below 0
        else:
            reward = self.calculateReward(self.state, action, previous_action) 
        # reward = self.calculateReward(self.state, action, previous_action) 
        
        # Check if done
        done = False
        if self.time_step >= 200.0:
            done = True
            print("End with Time : ", self.time_step)
        
        self.time_step += time_points

        return self.state, reward, done, {}
    
    def render(self, mode='human'):
        self.window.fill(WHITE)
        position_ball = self.state[0]
        position_paddle = self.state[2]
        self.render_state(self.window, position_ball, position_paddle)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

    def render_state(self, window, position_ball, position_paddle):
        paddle_width = 0.2
        paddle_height = 10

        y_pos = int(position_ball)
        x_pos = window.get_height() // 2
        y_pos_paddle = int(position_paddle)
        x_pos_paddle = window.get_height() // 2

        pygame.draw.circle(window, BLACK, (x_pos, window.get_height() - (y_pos * scaling_factor + window.get_width() * 0.2)), 10)
        y_screen = window.get_height() - paddle_height - int(y_pos_paddle)
        pygame.draw.rect(window, BLACK, (x_pos_paddle - (paddle_width / 2), window.get_height() - (y_pos_paddle * scaling_factor - 10 + window.get_width() * 0.2), int(paddle_width * window.get_width() * 0.4), paddle_height))