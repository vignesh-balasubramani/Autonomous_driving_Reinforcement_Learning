"""Defines the pygame simulation environment class
and displays the simulation in a pygame environment"""
import pickle
from pathlib import Path
from tkinter import filedialog
import numpy as np
import pygame
import track_utilities as track_section
from goals import Goals
from state_estimation_q_learn import StateEstimation
import vector


def load_vehicle_hull(vehicle_file):
    """Loads the vehicle data points"""
    vehicle_hull = []
    with open(vehicle_file, 'rb') as file:
        vehicle_hull = pickle.load(file)

    return vehicle_hull

def scaling(points, scale):
    """Scales the points with the scale value"""
    new_points = []
    for point in points:
        new_points.append(np.array(point)*scale)

    return new_points

class RacingEnv():
    """class for the simulation environment"""
    def __init__(self):
        pygame.init()
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.width = 1000
        self.height = 1000
        self.track_index = 0
        self.track_scale = 1
        self.goal_length = 10
        self.goal_width = 30
        self.number_of_rays = 12
        self.rays_flag = False
        self.check_points_flag = False
        self.full_track = False
        self.current_path = Path.cwd()
        self.track_path = []

    def track_select(self):
        """Selects the track path from the folder"""
        if not self.full_track:
            track_folder_path = (self.current_path.parents[0] / "TUM_database"
                                / "tracks" / "track_sections_db_video")
            track_paths = track_folder_path.iterdir()

            for track_path in track_paths:
                if not track_path.is_dir():
                    self.track_path.append(track_path)
        else:
            track_folder_path = (self.current_path.parents[0] / "TUM_database"
                                / "tracks")
            print(f"Selected folder: {str(track_folder_path)}")

            # Open a file dialog to choose a file within the selected folder
            self.track_path = filedialog.askopenfilename(
                                                initialdir = track_folder_path,
                                                title = "Select File")
            # Check if a file was selected
            if self.track_path:
                print(f"Selected file: {str(self.track_path)}")
            else:
                print("No file selected.")


    def track_reset(self, randomise=True):
        """Selects the nect track and also sets the goals for the track"""
        if not self.full_track:
            self.track = track_section.TrackSection(self.track_path[self.track_index],
                                                    self.width, self.height,
                                                    self.track_scale)
        else:
            self.track = track_section.TrackSection(self.track_path,
                                                    self.width, self.height,
                                                    self.track_scale)
        self.goal_width = self.goal_width * self.track_scale
        self.goals = Goals(self.goal_length, self.goal_width,
                           self.track.center_line,
                           self.track.left_bounds, self.track.right_bounds)

        if randomise:
            self.track_index = np.random.choice(len(self.track_path))
        else:
            self.track_index += 1

    def reset(self):
        """resets the environment before every episode"""
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racing_reinforcement_learning")
        self.screen.fill((0, 0, 0))
        self.check_point = 0
        self.game_reward = 0
        self.track_completed = False
        self.done = False
        self.center_distances = []
        self.current_positions = []

        self.center_point = (self.width/2, self.height/2)

        # import vehicle data points
        self.vehicle_path = (self.current_path.parents[0]
                             / "vehicle" / "TAURUS.bin")
        self.hull = load_vehicle_hull(self.vehicle_path)

        # scaling the values
        self.hull = scaling(self.hull, self.track_scale)


    def update(self, state):
        """Updates the game environment each step"""
        # extract the state values
        self.pos_x = state.x * self.track_scale
        self.pos_y = state.y * self.track_scale
        self.yaw_deg = state.yaw_deg
        self.velocity = state.v
        self.state_vector = []

        # rotate the current position according to the track angle
        self.pos_x, self.pos_y = vector.rot((self.pos_x, self.pos_y),
                                            (0, 0), self.track.angle)

        # convert the current position to pygame co-ordinates
        self.current_pos = track_section.convert_to_pygameXY((self.pos_x,
                                                              self.pos_y),
                                                             self.width,
                                                             self.height)
        self.current_positions.append(self.current_pos)

        # fetch the state vector for the current position
        self.observation = StateEstimation(self.current_pos,
                                           self.number_of_rays,
                                           self.yaw_deg,
                                           self.track.left_bounds,
                                           self.track.right_bounds,
                                           self.track.angle)
        self.state_vector = self.observation.state_vector

        # update the vehicle points w.r.t the current poistion
        self.new_hull = self.update_pose()

        # check for goals
        self.game_reward, self.track_completed, self.done = self.reward()

        return self.state_vector, self.game_reward, self.done, self.track_completed

    def reward(self):
        """calculates the reward"""
        check_point_reward = 0
        crash_reward = 0
        goal_reward = 0
        track_completed = False
        done = False
        # check for chekpoint crossing
        if self.goals.check_goals(self.new_hull, self.check_point):
            if self.check_point == len(self.goals.goals)-1:
                goal_reward = 1
                track_completed = True
                done = True
                print("Track completed")
            else:
                check_point_reward = 1
                self.check_point += 1

        # check for crash:
        if self.goals.check_crash(self.new_hull):
            crash_reward = 1
            done = True

        # distance from the center line
        center_dist = vector.shortest_distance(center=self.current_pos,
                                               line=self.track.center_line)
        center_dist = center_dist / self.track_scale
        self.center_distances.append(center_dist)
        # reward function
        game_reward = (-0.1 - 200*crash_reward + 25*check_point_reward
                        + 2000*goal_reward - 0.0*center_dist)

        return game_reward, track_completed, done

    def update_pose(self):
        """updates the pose of the vehicle points"""
        angle = -self.track.angle - self.yaw_deg
        rotated_hull = []
        for point in self.hull:
            rotated_hull.append(vector.rot(point, (0,0), angle))

        new_hull = []
        for point in rotated_hull:
            new_hull.append(vector.add(point, self.current_pos))

        return new_hull

    def draw(self, score, episode, velocity):
        """displays every update on the pygame window"""
        self.screen.fill((19, 109, 21)) # green fill
        # drawing the track
        pygame.draw.polygon(self.screen, (128, 128, 128), self.draw_track(), 0)
        pygame.draw.aalines(self.screen, (255, 0,0), False,
                            self.draw_offset(self.track.left_bounds), 4)
        pygame.draw.aalines(self.screen, (255, 0, 0), False,
                            self.draw_offset(self.track.right_bounds), 4)
        pygame.draw.aalines(self.screen, (0, 0, 255), False,
                            self.draw_offset(self.track.center_line), 1)

        # drawing the vehicle
        pygame.draw.aalines(self.screen, (255, 0, 0), False,
                            self.draw_vehicle(), 1)

        # drawing the rays
        if self.rays_flag:
            for ray in self.observation.intersection_point:
                pygame.draw.aalines(self.screen, (0, 0, 0), False,
                                    self.draw_offset([(self.current_pos[0],
                                                       self.current_pos[1]),
                                                        (ray.x, ray.y)]), 5)

        # drawing the checkpoints
        if self.check_points_flag:
            for check_point in self.goals.goals:
                pygame.draw.aalines(self.screen, (0, 255, 0), False,
                                    self.draw_offset(check_point), 5)

        # displaying the scores and velocity
        font = pygame.font.Font(None, 36)
        reward_text = f"Reward: {score:.2f}"
        episode_text = f"Episode: {episode}"
        velocity_text = f"Velocity(m/s):{velocity:.2f}"
        text_surface_reward = font.render(reward_text, True, (0, 0, 0))
        text_surface_episode = font.render(episode_text, True, (0, 0, 0))
        text_surface_velocity = font.render(velocity_text, True, (0, 0, 0))
        self.screen.blit(text_surface_reward, (10, 10))
        self.screen.blit(text_surface_episode, (10, 40))
        self.screen.blit(text_surface_velocity, (10, 70))

    def draw_track(self):
        """Returns the track points for drawing"""
        patch_points = self.track.left_bounds + self.track.right_bounds[::-1]
        patch_points = self.draw_offset(patch_points)

        return patch_points

    def draw_vehicle(self):
        """Returns the vehicle points for drawing"""
        polygon = self.new_hull
        polygon = [(x, y) for x, y in polygon]
        polygon.append(polygon[0])
        polygon = self.draw_offset(polygon)

        return polygon

    def draw_offset(self, points):
        """Moves all the components for a moving window effect"""
        diff_loc_vector = vector.sub(self.current_pos, self.center_point)
        new_points = []
        for point in points:
            new_points.append(vector.sub(point, diff_loc_vector))

        return new_points
