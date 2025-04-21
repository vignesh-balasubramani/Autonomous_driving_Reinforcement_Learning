"""Generates the sections of tracks based upon the number of points"""
from pathlib import Path
import numpy as np

def main():
    """main block"""
    SPLIT_AFTER = 50

    # create the folder "track sections"
    current_path = Path.cwd()
    folder_path = current_path.parents[0] / "TUM_database" / "tracks"
    new_folder_path = folder_path / "track_sections"
    new_folder_path.mkdir(parents = True, exist_ok = True)

    for track_path in folder_path.iterdir():
        if track_path.is_dir():
            continue

        # generate the sections
        track_data = np.genfromtxt(track_path, delimiter = ",")
        number_of_rows = track_data.shape[0]
        for i in range(number_of_rows // SPLIT_AFTER):
            file_path = new_folder_path / (track_path.stem + f"{i+1:02d}" + ".csv")
            start_index = i * SPLIT_AFTER
            end_index = (i+1) * SPLIT_AFTER
            if end_index > number_of_rows-1:
                end_index = -1
            split_data = track_data[start_index:end_index, :]
            header = "x_m,y_m,w_tr_right_m,w_tr_left_m"
            np.savetxt(file_path, split_data, delimiter=",",
                    header=header, fmt="%0.4f")

if __name__ == "__main__":
    main()
