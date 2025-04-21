"""Generates the plots of the sections of track in white bg"""
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    """main block"""
    DPI = 100
    FIG_SIZE = (10,10)

    current_path = Path.cwd()
    folder_path = (current_path.parents[0] / "TUM_database"
                   / "tracks" / "track_sections_db_test")
    new_folder = folder_path / "track_section_plots"
    if new_folder.exists():
        shutil.rmtree(new_folder, ignore_errors=True)
    new_folder.mkdir(parents= True, exist_ok= True)
    for track_path in folder_path.iterdir():
        print(track_path)
        if track_path.is_dir():
            continue

        track_data = np.genfromtxt(track_path, delimiter=",")
        x = track_data[:, 0]
        y = track_data[:, 1]
        w_right = track_data[:, 2]
        w_left = track_data[:, 3]

        vectors = np.zeros((len(x),2))
        norm_vectors = np.zeros((len(x),2))
        x_intercepts_L = np.zeros(len(x))
        y_intercepts_L = np.zeros(len(x))
        x_intercepts_R = np.zeros(len(x))
        y_intercepts_R = np.zeros(len(y))

        for (i,_) in enumerate(x):
            if i != len(x)-1:
                vector = ((x[i+1] - x[i]), (y[i+1] - y[i]))
            else:
                vector = ((x[i] - x[i-1]), (y[i] - y[i-1]))
            vectors[i,:] = vector
            norm_vector = (-vector[1], vector[0])
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
        left_bounds = np.column_stack((x_left_bounds, y_left_bounds))
        right_bounds = np.column_stack((x_right_bounds, y_right_bounds))
        center_line = np.column_stack((x, y))

        #creating single array for plotting
        right_bounds = right_bounds[::-1]
        plot_values = np.row_stack((left_bounds, right_bounds))
        first_row = plot_values[0]
        plot_values = np.concatenate((plot_values, [first_row]), axis=0)

        # plotting and saving the sections of the track
        line_width = 0.5
        _, ax = plt.subplots(dpi = DPI, figsize = FIG_SIZE, facecolor = "white")
        ax.plot(x, y, 'r', linewidth = line_width)
        ax.plot(plot_values[:, 0], plot_values[:, 1], "b", linewidth = line_width)
        ax.set_title(track_path.stem, {"fontsize":20})
        ax.set_aspect("equal")
        ax.axis("equal")

        file_path = new_folder / track_path.stem 
        if file_path.exists():
            file_path.unlink()
        plt.savefig(str(file_path), format = None)
        plt.close()

if __name__ == "__main__":
    main()
