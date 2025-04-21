"""This code generates the checkpoints for a selected track
given the distance between the checkpoints as the number of steps of the track.
The checkpoints are generated normal to the center line"""
from pathlib import Path
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from shapely import Point
from shapely import LineString
import vector
from track_utilities import TrackSection

class Goals:
    """Goal class definition"""
    def __init__(self, length_of_goal, width_of_goal,
                 center_line, left_boundary, right_boundary):
        self.goals = np.empty((0,2,2), dtype=np.float64)
        self.center_line = center_line
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.goal_width = width_of_goal

        #converting to linestrings
        self.left_bounds = LineString(self.left_boundary)
        self.right_bounds = LineString(self.right_boundary)

        # generating the checkpoints
        number_of_goals = len(self.center_line) // length_of_goal
        for i in range(number_of_goals):
            pt1 = self.center_line[length_of_goal*(i+1) - 1]
            # Calculate normal vector to point
            try:
                pt2 = self.center_line[length_of_goal*(i + 1)]
            except IndexError:
                pt2 = self.center_line[-1]
            try:
                pt3 = self.center_line[length_of_goal*(i + 1) - 2]
            except IndexError:
                pt3 = self.center_line[0]

            v3 = np.array(pt2) - np.array(pt3)
            normal_vector = np.array(vector.perp(v3), dtype=np.float64)

            if np.all(normal_vector == 0):
                normal_vector = np.array([0, 1], dtype=np.float64)

            norm_normal_vector = normal_vector / np.linalg.norm(normal_vector)
            scaled_normal_vector = norm_normal_vector * self.goal_width
            pt1_norm = pt1 + scaled_normal_vector
            pt2_norm = pt1 - scaled_normal_vector

            # Find intersecting point to normal vector
            normal_line_1 = LineString([pt1, pt1_norm])
            normal_line_2 = LineString([pt1, pt2_norm])

            intersection_pt1 = self.intersection_pt(pt1, normal_line_1)
            intersection_pt2 = self.intersection_pt(pt1, normal_line_2)

            if intersection_pt1 is not None and intersection_pt2 is not None:
                row_1 = np.array([intersection_pt1.x, intersection_pt1.y])
                row_2 = np.array([intersection_pt2.x, intersection_pt2.y])
                row_stack_1 = np.row_stack((row_1, row_2))
                row_stack_3d_1 = np.expand_dims(row_stack_1, axis=0)
                self.goals = np.append(self.goals,
                                    row_stack_3d_1, axis=0)

                if (i == number_of_goals-1
                        and len(self.center_line) % length_of_goal == 0):
                    self.goals = self.goals[:-1]

        #adding the final goal
        row_stack_2 = np.row_stack((self.left_boundary[-1], self.right_boundary[-1]))
        row_stack_3d_2 = np.expand_dims(row_stack_2, axis=0)
        self.goals = np.append(self.goals,
                               row_stack_3d_2, axis=0)

    def intersection_pt(self, pt1, normal_line):
        """Calculates the intersection point between the normal line and bounds"""
        # if the normal line intersects both the left and right bounds
        if (normal_line.intersects(self.left_bounds)
            and normal_line.intersects(self.right_bounds)):

            double_intersection = []
            left_intersection = normal_line.intersection(self.left_bounds)
            right_intersection = normal_line.intersection(self.right_bounds)

            if left_intersection.geom_type == "Point":
                double_intersection.append(left_intersection)
            elif left_intersection.geom_type == "MultiPoint":
                for pt in left_intersection.geoms:
                    double_intersection.append(pt)
            else:
                print("the checkpoint intersects the left bounds but"
                      "does not produce points")

            if right_intersection.geom_type == "Point":
                double_intersection.append(right_intersection)
            elif right_intersection.geom_type == "MultiPoint":
                for pt in right_intersection.geoms:
                    double_intersection.append(pt)
            else:
                print("the checkpoint intersects the right bounds but"
                      "does not produce points")

            temporary_pt = Point([0,0])
            distance = self.goal_width
            for pt in double_intersection:
                dist_line = LineString([pt1, pt])
                if dist_line.length < distance:
                    distance = dist_line.length
                    temporary_pt = pt

            return temporary_pt

        #if the normal line intersects only the left bounds
        elif normal_line.intersects(self.left_bounds):
            intersection_pt = normal_line.intersection(self.left_bounds)
            if intersection_pt.geom_type == "Point":
                return intersection_pt
            elif intersection_pt.geom_type == "MultiPoint":
                distance = self.goal_width
                temporary_pt = Point([0,0])
                for pt in intersection_pt.geoms:
                    dist_line = LineString([pt1, pt])
                    if dist_line.length < distance:
                        distance = dist_line.length
                        temporary_pt = pt
                return temporary_pt
            else:
                print("the checkpoint intersects the left bounds but"
                       "does not produce any points")
                return None

        #if the normal line intersects only the right bounds
        elif normal_line.intersects(self.right_bounds):
            intersection_pt = normal_line.intersection(self.right_bounds)
            if intersection_pt.geom_type == "Point":
                return intersection_pt
            elif intersection_pt.geom_type == "MultiPoint":
                distance = self.goal_width
                temporary_pt = Point([0,0])
                for pt in intersection_pt.geoms:
                    dist_line = LineString([pt1, pt])
                    if dist_line.length < distance:
                        distance = dist_line.length
                        temporary_pt = pt
                return temporary_pt
            else:
                print("the checkpoint intersects the right bounds but"
                       "does not produce any points")
                return None
        else:
            print("it does not intersect increase the width of goal")
            return None

    def plot_goals(self):
        """This plots the checkpoints with the track for testing purposes"""
        _, axes = plt.subplots(1,1, figsize = (10,10), dpi = 100)
        axes.plot(np.array(self.left_boundary)[:,0],
                  np.array(self.left_boundary)[:,1],
                  "g")
        axes.plot(np.array(self.right_boundary)[:,0],
                  np.array(self.right_boundary)[:,1],
                  "g")
        for goal in self.goals:
            axes.plot(np.array(goal)[:,0], np.array(goal)[:,1], "b")
        axes.set_aspect("equal")
        axes.set_xlabel("X distance (m)")
        axes.set_ylabel("Y distance (m)")
        axes.set_title("Checkpoints of track section")
        plt.show()

    def check_goals(self, vehicle_pos, check_point):
        """Checks if the vehicle has crossed the checkpoint"""
        vehicle_shape = LineString(vehicle_pos)
        check_point_crossed = False
        goal_line = LineString(self.goals[check_point])
        if vehicle_shape.intersects(goal_line):
            check_point_crossed = True

        return check_point_crossed

    def check_crash(self, vehicle_pos):
        """Checks if the vehicle has crashed the bounds"""
        vehicle_shape = LineString(vehicle_pos)
        if (vehicle_shape.intersects(self.left_bounds)
            or vehicle_shape.intersects(self.right_bounds)):
            crashed = True
        else:
            crashed = False

        return crashed

def main():
    """Testing code"""
    folder_path = (Path.cwd().parents[0] / "TUM_database" / "tracks"
                    / "track_sections")
    # Toggle for testing full tracks
    # folder_path = Path.cwd().parents[0] / "TUM_database" / "tracks"

    track_path = filedialog.askopenfilename(initialdir = folder_path,
                                               title = "Select File")
    track_section = TrackSection(track_path=track_path,
                                 width=1000, height=1000, track_scale=3)
    a = Goals(length_of_goal=5, width_of_goal= 100,
              center_line=track_section.center_line,
              left_boundary=track_section.left_bounds,
              right_boundary=track_section.right_bounds)
    a.plot_goals()
    # print(a.goals)

if __name__ == "__main__":
    main()
