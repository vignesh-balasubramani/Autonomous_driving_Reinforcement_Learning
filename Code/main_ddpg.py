"""Reinforcement learning code for learning and simulating
the race car environment using DDQN algorithm and a simple vehicle model
This code initialises all the necessary components and
controls the looping of the simulation"""
import shutil
import csv
import psutil
import os
import time
import GPUtil
import threading
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pygame
import pygame_simulation
from rl_agent_ddpg import DDPGAgent
from simple_vehicle_model import Car
from simple_vehicle_model import State

def monitor():
    process = psutil.Process(os.getpid())
    while True:
        # CPU and Memory usage
        print(f"Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")
        print(f"CPU usage: {process.cpu_percent(interval=1):.2f} %")
        
        # GPU usage
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}")
            print(f"GPU Name: {gpu.name}")
            print(f"GPU Load: {gpu.load * 100:.2f} %")
            print(f"GPU Memory Free: {gpu.memoryFree:.2f} MB")
            print(f"GPU Memory Used: {gpu.memoryUsed:.2f} MB")
            print(f"GPU Memory Total: {gpu.memoryTotal:.2f} MB")
            print(f"GPU Temperature: {gpu.temperature} °C")
        print("=" * 20)
        time.sleep(100)  # Adjust the sleep time as needed

# Run the monitor in a separate thread if needed
monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()

def clear_tensorboard_logs(log_dir):
    """Clear the log directory before every run"""
    shutil.rmtree(log_dir, ignore_errors=True)
    print("TensorBoard logs cleared.")

#------------------loading the config file-------------------
def load_config(filename):
    """loads the config file"""
    with open(filename, "r") as file:
        cnfg = yaml.safe_load(file)

    return cnfg

config_file_name = Path.cwd() / "ddpg_parameters.yaml"
config = load_config(config_file_name)
#------------------------------------------------------------
testing_flag = config["testing"] # should testing be done
validation_flag = config["validation"] # should validation be done

# random seeding
SEED = config["random_seed"]
np.random.seed(SEED)
tf.random.set_seed(SEED)

#--------------------------Setting up parameters------------------------------
#game parameters
game_params = config["game_parameters"]
FPS = game_params["fps"]
AVG_WINDOW = game_params["avg_window"]
TOTAL_GAMETIME = game_params["total_gametime"] # max game time for one episode
N_EPISODES = game_params["n_episodes"] # number of episodes
TEST_AFTER = game_params["test_after"] # testing after n episodes
N_TEST_EPISODES = game_params["n_test_episodes"] # number of test episodes
RANDOMISE_AFTER = game_params["randomise_after"] # after how many epidoses to randomise the track
SAVE_AFTER = game_params["save_after"] # saves the model after n episodes

# nertwork parameters
network_params = config["network_parameters"]
FC1 = network_params["fc1"]
FC2 = network_params["fc2"]
ALPHA = network_params["alpha"] # learning factor of online network
BETA = network_params["beta"] # learning factor of target network
GAMMA = network_params["gamma"] # discount factor
TAU = network_params["tau"] # Replace factor
BATCH_SIZE = network_params["batch_size"] # mini batch size for learning
NOISE = network_params["noise"] # standard deviation of the noise

# setting up the gaming environment
game = pygame_simulation.RacingEnv()
game_env_params = config["game_environment"]
game.fps = FPS
game.width = game_env_params["width"]
game.height = game_env_params["height"]
game.track_scale = game_env_params["track_scale"]
game.goal_length = game_env_params["goal_length"]
game.number_of_rays = game_env_params["number_of_rays"]
game.rays_flag = game_env_params["rays_flag"] # Set true to display the rays on the screen
game.check_points_flag = game_env_params["check_points_flag"] # Set true to display the checkpoints on the screen
game.full_track = game_env_params["full_track"] # Set true to run for full tracks
game.track_select()
game.track_reset()

#setting up the vehicle parameters
vehicle = Car()
vehicle.fps = FPS

# episode history
ddpg_scores = []

# setting up the action space
action_space = np.arange(0,config["n_actions"])

# setting up the agent
ddpg_agent = DDPGAgent(input_dims = game.number_of_rays,
                       alpha = ALPHA,
                       beta = BETA,
                       gamma = GAMMA,
                       n_actions = config["n_actions"],
                       buffer_size = 10000,
                       tau = TAU,
                       fc1 = FC1,
                       fc2 = FC2,
                       batch_size = BATCH_SIZE,
                       noise = NOISE)


# if simulation stops in between runs from the last saved episode
game.reset()
vehicle.reset()
vehicle.velocity = [config["vehicle_velocity"]] # initial vehicle velocity
state = State(vehicle.pos_x, vehicle.pos_y,
                vehicle.yaw_deg, vehicle.velocity)
observation_temp, reward, done, track_complete = game.update(state)
observation = np.array(observation_temp)
action = ddpg_agent.choose_action(observation)
state = State(vehicle.pos_x, vehicle.pos_y,
                vehicle.yaw_deg, vehicle.velocity)
observation_temp, reward, done, track_complete = game.update(state)
observation_next = np.array(observation_temp)
ddpg_agent.remember(observation, action, reward, observation_next, done)
ddpg_agent.memory.buffer_index = ddpg_agent.batch_size
ddpg_agent.learn()
last_episode = ddpg_agent.load_progress()
if validation_flag:
    last_episode = 0

ddpg_agent.memory.buffer_index = 0
#----------------------------Creating logs-------------------------------------
# creating a summary writer for tensorboard
clear_tensorboard_logs("logs")
LOG_DIR = "logs"
summary_writer = tf.summary.create_file_writer(LOG_DIR)

# Creating a csv file for logging
csv_filename = "ddpg_episode_log.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Episode", "score", "avg_score", "critic_loss",
                     "actor_loss","Track_id", "Track_complete"])

# Creating a csv file for validation logging
csv_validation_filename = "ddpg_validation_episode_log.csv"
csv_validation_file = open(csv_validation_filename, 'w', newline='')
csv_validation_writer = csv.writer(csv_validation_file)
csv_validation_writer.writerow(["Episode", "score", "avg_score",
                                "Track_id", "Track_complete"])

# Creating a csv fle for test logging
csv_test_filename = "ddpg_test_log.csv"
csv_test_file = open(csv_test_filename, "w", newline="")
csv_test_writer = csv.writer(csv_test_file)
csv_test_writer.writerow(["Episode", "Test", "Score",
                          "Track_id", "Track_complete"])


#----------------------------Simulation loop-----------------------------------
def run():
    """Simulation loop"""
    for episode in range(last_episode, N_EPISODES):
        game.reset()
        vehicle.reset()
        vehicle.velocity = [config["vehicle_velocity"]] # initial vehicle velocity
        done = False
        track_complete = False
        score = 0
        counter = 0
        g_time = 0

        # initial state
        state = State(vehicle.pos_x, vehicle.pos_y,
                        vehicle.yaw_deg, vehicle.velocity)

        observation_temp, reward, done, track_complete = game.update(state)
        observation = np.array(observation_temp)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("pygame closed")
                    return

            # getting the action based upon the policy
            action = ddpg_agent.choose_action(observation, validation_flag)
            vel_delta, yaw_delta = vehicle.action_cont(action)
            vehicle.update(vel_delta, yaw_delta)

            # generating the state
            state = State(vehicle.pos_x, vehicle.pos_y,
                            vehicle.yaw_deg, vehicle.velocity)

            observation_temp, reward, done, track_complete = game.update(state)
            observation_next = np.array(observation_temp)

            # if no reward is collected the car will be done within 1000 ticks
            if reward < 0:
                counter += 1
                if counter > 1000:
                    print("killed because no reward")
                    done = True
            else:
                counter = 0

            score += reward

            # simulate the steps in pygame
            game.draw(score, episode, state.v)
            pygame.display.flip()
            game.clock.tick(game.fps)

            # store the state and action in memory buffer
            ddpg_agent.remember(observation, action, reward,
                                observation_next, int(done))
            observation = observation_next

            # Learning from the experiments
            if not validation_flag:
                ddpg_agent.learn()

            g_time += 1
            if g_time >= TOTAL_GAMETIME:
                done = True
                print("killed because epsiode time reached")

        #-----------------------testing the policy-----------------------------
        if episode % TEST_AFTER == 0 and episode > 0 and testing_flag:
            print("Testing started")
            for test_episode in range(N_TEST_EPISODES):
                game.track_reset()
                game.reset()
                vehicle.reset()
                vehicle.velocity = [config["vehicle_velocity"]] # intial vehicle velocity
                test_done = False
                test_track_complete = False
                test_score = 0
                test_counter = 0
                test_g_time = 0

                state = State(vehicle.pos_x, vehicle.pos_y,
                              vehicle.yaw_deg, vehicle.velocity)

                _, test_reward, test_done, _ = game.update(state)

                while not test_done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("pygame test window closed")
                            test_done = True

                    action = ddpg_agent.choose_action(observation, True)
                    vel_delta, yaw_delta = vehicle.action_cont(action)
                    vehicle.update(vel_delta, yaw_delta)

                    state = State(vehicle.pos_x, vehicle.pos_y,
                                  vehicle.yaw_deg, vehicle.velocity)

                    _, test_reward, test_done, test_track_complete = game.update(state)

                    if test_reward < 0:
                        test_counter += 1
                        if test_counter > 1000:
                            print("killed because no reward")
                            test_done = True
                    else:
                        test_counter = 0

                    test_score += test_reward
                    game.draw(test_score, test_episode, state.v)
                    pygame.display.flip()
                    game.clock.tick(game.fps)

                    test_g_time += 1
                    if test_g_time >= TOTAL_GAMETIME:
                        test_done = True
                        print("test exceeded game time")

                csv_test_writer.writerow([episode,
                                          test_episode,
                                          f"{test_score:.2f}",
                                          game.track_index,
                                          int(test_track_complete)])
                print(f"Test episode:{test_episode}, test_score:{test_score}")
            print("Testing finished")
        #----------------------------testing done ----------------------------#
        #logging the episode data in tensorboard
        with summary_writer.as_default():
            tf.summary.scalar("Cumulative Reward",
                               score,
                               step = episode)
            tf.summary.scalar("Critic Loss function",
                              ddpg_agent.critic_loss,
                              step = episode)
            tf.summary.scalar("Actor loss function",
                              ddpg_agent.actor_loss,
                              step = episode)

        ddpg_scores.append(score)
        avg_score = np.mean(ddpg_scores[max(0, episode-AVG_WINDOW):(episode+1)])

        # track randomising after n episodes
        if episode % RANDOMISE_AFTER == 0 and episode > 0:
            game.track_reset()

        #saving the model
        if not validation_flag:
            if episode % SAVE_AFTER == 0 and episode > 0:
                ddpg_agent.save_progress(episode)
                print("save model")

            #logging the data in csv files
            csv_writer.writerow([episode, f"{score:.2f}", f"{avg_score:.2f}",
                                f"{ddpg_agent.critic_loss:.4f}", f"{ddpg_agent.actor_loss: 0.4f}",
                                game.track_index,
                                int(track_complete)])
        else:
            csv_validation_writer.writerow([episode, f"{score:.2f}",
                                            f"{avg_score:.2f}",
                                            game.track_index,
                                            int(track_complete)])

        #Episode verbose
        print(f"episode: {episode}", f"score: {score:.2f}",
            f"average score: {avg_score: .2f}",
            f"memory size: {ddpg_agent.memory.buffer_index % ddpg_agent.memory.buffer_size}")

run()
