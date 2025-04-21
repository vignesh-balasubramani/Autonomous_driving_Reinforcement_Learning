"""Reinforcement learning code for learning and simulating
the race car environment using DDQN algorithm and a simple vehicle model
This code initialises all the necessary components and
controls the looping of the simulation"""
import shutil
import csv
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pygame
import pygame_simulation
from rl_agent_ddqn import DDQNAgent
from simple_vehicle_model import Car
from simple_vehicle_model import State


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

config_file_name = Path.cwd() / "ddqn_parameters.yaml"
config = load_config(config_file_name)
#--------------------------------------------------------------
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
ALPHA = network_params["alpha"] # learning factor
GAMMA = network_params["gamma"] # discount factor
FC1 = network_params["fc1"] # number of neurons in the first layer
FC2 = network_params["fc2"] # number of neurons in the second layer
BATCH_SIZE = network_params["batch_size"] # mini batch size for learning
EPS = network_params["eps_start"] # epsilon start
EPS_END = network_params["eps_end"] # epsilon end
EPS_DECAY_STEPS = network_params["eps_decay_steps"] # epsilon decay steps
REPLACE_TARGET = network_params["replace_target"] # replace target weights

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
game.track_reset(not validation_flag)

#setting up the vehicle parameters
vehicle = Car()
vehicle.fps = FPS

# episode history
ddqn_scores = []
epsilon_history = []

# setting up the action space
action_space = np.arange(0,config["n_actions"])

# setting up the agent
ddqn_agent = DDQNAgent(alpha = ALPHA,
                       gamma = GAMMA,
                       n_actions = action_space.shape[0],
                       n_states = game.number_of_rays,
                       batch_size = BATCH_SIZE,
                       fc1 = FC1,
                       fc2 = FC2,
                       eps = EPS,
                       eps_end = EPS_END,
                       decay_steps = EPS_DECAY_STEPS,
                       replace_target= REPLACE_TARGET)

# if simulation stops in between runs from the last saved episode
last_episode = ddqn_agent.load_model()
if validation_flag:
    last_episode = 0
    ddqn_agent.eps = 0.0
#----------------------------Creating logs-------------------------------------
# creating a summary writer for tensorboard
clear_tensorboard_logs("logs")
LOG_DIR = "logs"
summary_writer = tf.summary.create_file_writer(LOG_DIR)

# Creating a csv file for logging
csv_filename = "ddqn_episode_log.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Episode", "score", "avg_score", "epsilon", "loss",
                     "Track_id", "Track_complete"])

# Creating a csv file for validation logging
csv_validation_filename = "ddqn_validation_episode_log.csv"
csv_validation_file = open(csv_validation_filename, 'w', newline='')
csv_validation_writer = csv.writer(csv_validation_file)
csv_validation_writer.writerow(["Episode", "score", "avg_score",
                                "Track_id", "Track_complete",
                                "max_center_distance", "center_dist"])

# Creating a csv fle for test logging
csv_test_filename = "ddqn_test_log.csv"
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
            action = ddqn_agent.choose_action(observation, ddqn_agent.eps)
            vel_delta, yaw_delta = vehicle.action(action)
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
            ddqn_agent.remember(observation, action, reward,
                                observation_next, int(done))
            observation = observation_next

            # Learning from the experiments
            if not validation_flag:
                print("this")
                ddqn_agent.learn()

            g_time += 1
            if g_time >= TOTAL_GAMETIME:
                done = True
                print("killed because epsiode time reached")

        # epsilon decay
        if not validation_flag:
            if ddqn_agent.eps > ddqn_agent.eps_end:
                ddqn_agent.eps -= ddqn_agent.decay
            else:
                ddqn_agent.eps = ddqn_agent.eps_end

        #logging the episode data in tensorboard
        with summary_writer.as_default():
            tf.summary.scalar("Cumulative Reward",
                               score,
                               step=episode)
            if not validation_flag:
                tf.summary.scalar("Loss function",
                                ddqn_agent.loss_value[0],
                                step = episode)

        epsilon_history.append(ddqn_agent.eps)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, episode-AVG_WINDOW):(episode+1)])

        # replacing the target weights
        if episode % REPLACE_TARGET == 0 and episode > 0:
            ddqn_agent.update_network_parameters()

        #saving the model
        if not validation_flag:
            if episode % SAVE_AFTER == 0 and episode > 0:
                ddqn_agent.save_progress(episode)
                print("save model")

            #logging the data in csv files
            csv_writer.writerow([episode, f"{score:.2f}", f"{avg_score:.2f}",
                                ddqn_agent.eps,
                                f"{ddqn_agent.loss_value[0]:.4f}",
                                game.track_index,
                                int(track_complete)])
        else:
            csv_validation_writer.writerow([episode, f"{score:.2f}",
                                            f"{avg_score:.2f}",
                                            game.track_index,
                                            int(track_complete),
                                            max(game.center_distances),
                                            *game.center_distances])

        #plotting the vehicle path:
        if episode % 20 == 0 and validation_flag:
            label_size=18
            x_start_l, y_start_l = game.track.left_bounds[0]
            x_start_r, y_start_r = game.track.right_bounds[0]
            x_finish_l, y_finish_l = game.track.left_bounds[-1]
            x_finish_r, y_finish_r = game.track.right_bounds[-1]
            x_start = [x_start_l, x_start_r]
            y_start = [y_start_l, y_start_r]
            x_finish = [x_finish_l, x_finish_r]
            y_finish = [y_finish_l, y_finish_r]

            plt.figure(figsize=(10,10))
            plt.plot(np.array(game.track.left_bounds)[:, 0],
                     np.array(game.track.left_bounds)[:, 1],
                     "black", linewidth=1)
            plt.plot(np.array(game.track.right_bounds)[:, 0],
                     np.array(game.track.right_bounds)[:, 1],
                     "black", linewidth=1)
            plt.plot(x_start, y_start, "green", label="Track start", linewidth=2)
            plt.plot(x_finish, y_finish, "magenta", label="Track finish",linewidth=2)
            plt.plot(np.array(game.track.center_line)[:, 0],
                     np.array(game.track.center_line)[:,1],
                      "blue", label="Center line", linewidth=1)
            plt.plot(np.array(game.current_positions)[:, 0],
                     np.array(game.current_positions)[:,1], "red",
                     label="Path traced by the vehicle", linewidth=1)
            plt.xlabel("X distance in m", fontsize=label_size)
            plt.ylabel("Y distance in m", fontsize=label_size)
            plt.title("Path traced by the vehicle", fontsize=label_size)
            plt.axis("equal")
            plt.legend(fontsize=label_size)
            plt.grid()

        #-----------------------testing the policy-----------------------------
        if episode % TEST_AFTER == 0 and episode > 0 and testing_flag:
            print("Testing started...")
            for test_episode in range(N_TEST_EPISODES):
                # game.track_reset()
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

                    action = ddqn_agent.choose_action(observation, 0)
                    vel_delta, yaw_delta = vehicle.action(action)
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
        #----------------------------testing done ----------------------------

        # track randomising after n episodes
        if episode % RANDOMISE_AFTER == 0 and episode > 0:
            game.track_reset(not validation_flag)

        #Episode verbose
        print(f"episode: {episode}", f"score: {score:.2f}",
            f"average score: {avg_score: .2f}",
            f"epsilon: {ddqn_agent.eps: .2f}",
            f"memory size: {ddqn_agent.memory.buffer_index % ddqn_agent.memory.buffer_size}")

run()
