"""Track section import and setup for the environment"""
from pathlib import Path
import math
import numpy as np


def convert_to_pygameXY(vector, width, height):
    """Converts the co-ordinates to pygame co-ordinates"""
    Vx, Vy = vector
    Cx, Cy = width/2, height/2

    x, y = Cx + Vx, Cy - Vy

    return x, y

def center_adjust(vector, starting_pos):
    """Adjusts the points for the pygame center"""
    Vx, Vy = vector
    Cx, Cy = starting_pos

    x, y = Vx - Cx, Vy - Cy

    return x, y

class TrackSection():
    """Generates the boundaries of the track and coverts 
    the co-ordinates to pygame cordinates"""
    def __init__(self, track_path, width=1000, height=1000, track_scale=1):
        self.track_path = track_path
        self.width = width
        self.height = height
        self.track_scale = track_scale
        self.left_bounds = []
        self.right_bounds = []
        self.center_line = []
        self.angle = 0
        self.starting_pos = (0, 0)
        self.track_parser()
        self.center_adjust()
        self.track_angle()
        self.pygame_convert()


    def track_parser(self):
        """Generates the boundaries for the given track"""
        track_data = np.genfromtxt(self.track_path, delimiter=",")
        x = track_data[:, 0] * self.track_scale
        y = track_data[:, 1] * self.track_scale
        w_right = track_data[:, 2] * self.track_scale
        w_left = track_data[:, 3] * self.track_scale

        vectors = np.zeros((len(x),2))
        norm_vectors = np.zeros((len(x),2))
        x_intercepts_L = np.zeros(len(x))
        y_intercepts_L = np.zeros(len(y))
        x_intercepts_R = np.zeros(len(x))
        y_intercepts_R = np.zeros(len(y))

        for (i,_) in enumerate(x):
            if i != len(x)-1:
                vector = ((x[i+1] - x[i]), (y[i+1] - y[i]))
            else:
                vector = ((x[i] - x[i-1]), (y[i] - y[i-1]))
            vectors[i,:] = vector
            norm_vector = (-vector[1], vector[0])
            if np.all(norm_vector == 0):
                norm_vector = np.array((1, 0))
            norm_vectors[i,:] = norm_vector
            unit_norm_vector = norm_vector / np.linalg.norm(norm_vector)

            # calculating the left intercepts
            x_intercept_L = w_left[i] * unit_norm_vector[0]
            y_intercept_L = w_left[i] * unit_norm_vector[1]
            x_intercepts_L[i] = x_intercept_L
            y_intercepts_L[i] = y_intercept_L

            # calculating the right intercepts
            x_intercept_R = w_right[i] * (-1*unit_norm_vector[0])
            y_intercept_R = w_right[i] * (-1*unit_norm_vector[1])
            x_intercepts_R[i] = x_intercept_R
            y_intercepts_R[i] = y_intercept_R

        # calculating the x and y for the bounds
        x_left_bounds = x + x_intercepts_L
        y_left_bounds = y + y_intercepts_L

        x_right_bounds = x + x_intercepts_R
        y_right_bounds = y + y_intercepts_R

        # Calculating the left and right bounds
        self.left_bounds = np.column_stack((x_left_bounds, y_left_bounds))
        self.right_bounds = np.column_stack((x_right_bounds, y_right_bounds))
        self.center_line = np.column_stack((x, y))
        self.starting_pos = self.center_line[0]

        return None

    def track_angle(self):
        """Calculates the angle of the given track from [0, 360]"""
        x1, y1 = self.center_line[0]
        x2, y2 = self.center_line[4]

        if x2 - x1 == 0:
            angle = math.pi/2
        else:
            angle = math.atan((y2 - y1) / (x2 - x1))

        self.angle = math.degrees(angle)

        if x2 > 0 and y2 > 0: # first quadrant
            self.angle = self.angle
        elif x2 < 0 and y2 > 0: # second quadrant
            self.angle = 180 + self.angle
        elif x2 < 0 and y2 < 0: # third quadrant
            self.angle = 180 + self.angle
        else: # fourth quadrant
            self.angle = 360 + self.angle

        return None

    def center_adjust(self):
        """Adjust the bounds and center line for (0, 0)"""
        self.left_bounds = [center_adjust((x, y), (self.starting_pos))
                            for x, y in self.left_bounds]
        self.right_bounds = [center_adjust((x, y), (self.starting_pos))
                                for x, y in self.right_bounds]
        self.center_line = [center_adjust((x, y), (self.starting_pos))
                            for x, y in self.center_line]

    def pygame_convert(self):
        """Converts the points to pygame co ordinates"""
        self.left_bounds = [convert_to_pygameXY((x, y), self.width, self.height)
                            for x, y in self.left_bounds]
        self.right_bounds = [convert_to_pygameXY((x, y), self.width, self.height)
                             for x, y in self.right_bounds]
        self.center_line = [convert_to_pygameXY((x, y), self.width, self.height)
                            for x, y in self.center_line]

        return None

def main():
    """Testing code"""
    current_path = Path.cwd()
    folder_path = (current_path.parents[0] / "TUM_database" / "tracks"
                   / "track_sections")

    for track_path in folder_path.iterdir():
        if track_path.is_dir():
            continue
        _ = TrackSection(track_path)
    return None

if __name__ == "__main__":
    main()
