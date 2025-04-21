"""Plots the cross track error"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

current_path = Path.cwd()
file_name = "q_learn_validation_episode_log.csv"
file_path = (current_path / "saved_models" / "q_learning"
             / "06_32_sections_new_reward_fn" / file_name)

data = pd.read_csv(file_path)

max_center_distance = data["max_center_distance"]
#plotting
label_font_size = 14
plt.figure(figsize=(10, 6))
plt.scatter(range(len(max_center_distance)), max_center_distance, c="blue", marker="o")
plt.title("Maximum cross track error", fontsize=16)
plt.xlabel("Test road sections", fontsize=label_font_size)
plt.ylabel("Distance from the center line (m)", fontsize=label_font_size)
plt.ylim((0,10))
plt.grid()
plt.show()
