# Autonomous Driving Simulation using Reinforcement Learning

This project demonstrates how to train a self-driving car to navigate racetracks using various **Reinforcement Learning (RL)** algorithms—**Q-learning**, **Deep Double Q-Learning (DDQN)**, and **Deep Deterministic Policy Gradient (DDPG)**—within a custom-built **Pygame simulation**.

The goal is to train an agent to autonomously drive a single-track car model through TUM racetracks using only perception (ray sensors) and rewards.

---

## Algorithms Implemented

### 1. Q-Learning
- **Type**: Tabular RL
- **State**: Discretized ray sensor values + angle to goal
- **Actions**: Discrete (e.g., accelerate, decelerate, turn)
- **Highlights**: Simple implementation for baseline performance.

### 2. Deep Double Q-Learning (DDQN)
- **Type**: Value-based Deep RL
- **Architecture**: Neural network with target network
- **Exploration**: Epsilon-greedy
- **Highlights**: Stable and scalable for large state spaces.

### 3. Deep Deterministic Policy Gradient (DDPG)
- **Type**: Actor-Critic, Continuous Action Space
- **Architecture**: Actor + Critic networks with target versions
- **Exploration**: Gaussian or Ornstein-Uhlenbeck noise
- **Highlights**: Enables fine-grained steering and throttle control.

---

## Environment Features

- **Framework**: Pygame
- **Simulation**: TUM racetrack with ray-based state representation
- **Vehicle**: Kinematic single-track model with throttle and steering
- **Sensors**: Simulated LiDAR (ray casts)
- **Reward**:
  - +Reward for forward movement and staying on track
  - -Penalty for collisions or stalling
  - Bonus for completing lap

---

## Logging & Visualization

- **TensorBoard**:
  - Episode rewards
  - Actor/Critic losses
- **CSV Logs**:
  - Per-episode scores, losses, steps, and distance
- **Track Visualization**:
  - Vehicle trajectory plotted using Matplotlib
- **Performance Monitoring**:
  - GPU, CPU, and RAM usage tracked during training

---

