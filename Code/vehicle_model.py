"""This module initialises the class Car and updates the step"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Car():
    """Defines the vehicle model"""
    def __init__(self):
        self.fps = 120
        self.dt = 1 / self.fps
        self.time = []
        self.step = 0

        # vehicle spec
        self.mass = 980 #kg
        self.weight = self.mass * 9.81 #N
        self.wheel_radius = 0.2876 #tires 175/70 R13 radius in m
        self.wheel_base = 2.96 # tesla model s
        self.max_steering_angle = 20 # 20 degrees on either side

        # powertrain values
        self.motor_nom_torque = 1500 #Nm
        self.motor_eff = 0.9
        self.brake_torque = 1200 #Nm

        # drag properties
        self.cd = 0.33 # drag co efficient
        self.frontal_area = 1.8 # m**2
        self.air_density = 1.2 #kg/m**3
        self.fr = 0.015 # rolling resistance co efficient

        self.pos_xx = []
        self.pos_x = []
        self.pos_y = []
        self.yaw_deg = []
        self.steering_angle = []
        self.velocity = []
        self.acceleration_x = []
        self.acceleration_y = []
        self.acceleration_factor = []
        self.steering_factor = []

    def reset(self):
        """Resets the vehicle parameters"""
        self.time = [0]
        self.pos_xx = [0]
        self.pos_x = [0]
        self.pos_y = [0]
        self.yaw_deg = [0]
        self.steering_angle = [0]
        self.velocity = [0]
        self.acceleration_x = []
        self.acceleration_y = []
        self.acceleration_factor = []
        self.steering_factor = []
        self.steering_angle = []
        self.step = 0

    def update(self, acceleration_factor, steering_factor):
        """Calculates the velocity values for the next step"""
        ###################### longitudinal dynamics ##########################
        self.acceleration_factor.append(acceleration_factor)
        f_drag = 0.5 * self.air_density * self.cd * self.frontal_area
        f_roll = self.fr * self.weight
        f_inertial = 1 / self.mass

        # accelerating conditions
        if acceleration_factor >= 0:
            f_trac = ((self.motor_nom_torque
                       * acceleration_factor
                       * self.motor_eff)
                      / self.wheel_radius)

            # modelling the ODE
            def model_throttle(t, x, f_inertial, f_drag, f_roll, f_trac):
                dxdt = x[1] # velocity is dx/dt
                d2xdt2 = f_inertial * (f_trac - f_drag * dxdt**2 - f_roll)

                return [dxdt, d2xdt2]

            # initial values
            t_span = (0, self.dt) # the amount of points
            x0 = [self.pos_xx[self.step], self.velocity[self.step]] # initial values [x, dx/dt]

            #Solving the ODE
            solution_throttle = solve_ivp(model_throttle,
                                          t_span,
                                          x0,
                                          args = (f_inertial,
                                                  f_drag,
                                                  f_roll,
                                                  f_trac),
                                          t_eval = np.linspace(0, self.dt, 10))
            # Solution_throttle
            time_values = solution_throttle.t
            x_values_throttle = solution_throttle.y[0]
            velocity_throttle = np.gradient(x_values_throttle, time_values)
            acceleration_throttle = np.gradient(velocity_throttle, time_values)
            self.pos_xx.append(x_values_throttle[-1])
            self.velocity.append(velocity_throttle[-1])
            self.acceleration_x.append(acceleration_throttle[0])

        # braking conditions
        else:
            f_brake = (self.brake_torque * -1*acceleration_factor
                       / self.wheel_radius)

            def model_brake(t, x, f_inertial, f_drag, f_roll, f_brake):
                dxdt = x[1]
                d2xdt2 = -f_inertial * (f_brake + f_drag * dxdt**2 - f_roll)

                return [dxdt, d2xdt2]

            if self.velocity[self.step] > 0:
                t_span = (0, self.dt)
                x0 = [self.pos_xx[self.step], self.velocity[self.step]]
                solution_brake = solve_ivp(model_brake,
                                           t_span,
                                           x0,
                                           args = (f_inertial,
                                                   f_drag,
                                                   f_roll,
                                                   f_brake),
                                           t_eval = np.linspace(0, self.dt, 10))
                time_values = solution_brake.t
                x_values_brake = solution_brake.y[0]
                velocity_brake = np.gradient(x_values_brake, time_values)
                acceleration_brake = np.gradient(velocity_brake, time_values)
                self.pos_xx.append(x_values_brake[-1])
                self.velocity.append(velocity_brake[-1])
                self.acceleration_x.append(acceleration_brake[0])
            else:
                self.pos_xx.append(self.pos_xx[-1])
                self.velocity.append(0)
                self.acceleration_x.append(0)

        ########################### lateral dynamnics ##########################
        beta_deg = (self.max_steering_angle / 2) * steering_factor
        beta_rad = np.deg2rad(beta_deg)
        yaw_rad = np.deg2rad(self.yaw_deg[self.step])
        self.time.append(self.time[self.step] + self.dt)

        self.steering_angle.append(beta_deg)
        self.steering_factor.append(steering_factor)

        delta_x1 = self.velocity[self.step] * np.cos(yaw_rad) * self.dt
        delta_y1 = self.velocity[self.step] * np.sin(yaw_rad) * self.dt

        acceleration_y = (self.velocity[self.step]**2
                          / self.wheel_base * np.tan(beta_rad))
        self.acceleration_y.append(acceleration_y)

        delta_x2 = acceleration_y * np.sin(yaw_rad) * self.dt**2 * 0.5
        delta_y2 = acceleration_y * np.cos(yaw_rad) * self.dt**2 * 0.5

        delta_x3 = (self.acceleration_x[self.step] * np.cos(yaw_rad)
                    * self.dt**2 * 0.5)
        delta_y3 = (self.acceleration_x[self.step] * np.sin(yaw_rad)
                    * self.dt**2 * 0.5)

        x_value = self.pos_x[self.step] + delta_x1 - delta_x2 + delta_x3
        y_value = self.pos_y[self.step] + delta_y1 + delta_y2 + delta_y3

        self.pos_x.append(x_value)
        self.pos_y.append(y_value)

        next_yaw_rad = yaw_rad + (self.velocity[self.step] / self.wheel_base
                                  * np.tan(beta_rad) * self.dt)
        next_yaw_deg = np.degrees(next_yaw_rad)
        self.yaw_deg.append(next_yaw_deg)

        self.step += 1

    def action(self, action_case):
        """Defines the action states for the vehicle"""
        if action_case == 0:
            # Left turn
            acceleration_factor = 0
            steering_factor = 0.99

        elif action_case == 1:
            # Right turn
            acceleration_factor = 0
            steering_factor = -0.99

        elif action_case == 2:
            # no steering
            acceleration_factor = 0
            steering_factor = 0

        elif action_case == 2:
            # Acceleration
            acceleration_factor = 1
            steering_factor = 0

        else:
            # Braking
            acceleration_factor = -0.5
            steering_factor = 0

        return acceleration_factor, steering_factor

class State():
    "creates a state representation"
    def __init__(self, pos_x, pos_y, yaw_deg, velocity):
        self.x = pos_x[-1]
        self.y = pos_y[-1]
        self.yaw_deg = yaw_deg[-1]
        self.v = velocity [-1]


def main():
    """Testing code"""
    test_vehicle = Car()
    test_vehicle.reset()
    for _ in range(1000):
        test_vehicle.update(1, 0)
    for _ in range(500):
        test_vehicle.update(0, 0.2)
    for _ in range(500):
        test_vehicle.update(0, -0.2)
    for _ in range(200):
        test_vehicle.update(1, 0)
    for _ in range(500):
        test_vehicle.update(-1, 0)

    plt.figure(figsize=(10, 20))
    label_font_size = 12

    plt.subplot(6, 1, 1)
    plt.plot(test_vehicle.time, test_vehicle.pos_x, label='X(t)')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('X(t)\nm', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(test_vehicle.time, test_vehicle.pos_y, label='Y(t)')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('Y(t)\nm', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(test_vehicle.time, test_vehicle.velocity, label='Velocity')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('v(t)\nm/s', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(test_vehicle.time[:-1], test_vehicle.acceleration_x, label='Acceleration_x')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('a_x(t)\nm/s^2', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(test_vehicle.time[:-1], test_vehicle.acceleration_y, label='Acceleration_y')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('a_y(t)\nm/s^2', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(test_vehicle.time, test_vehicle.yaw_deg, label='Yaw Angle')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('yaw(t)\ndegrees', fontsize=label_font_size)
    plt.grid()
    plt.legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    plt.show()

    plt.figure(figsize = (10,10))
    plt.plot(test_vehicle.pos_x, test_vehicle.pos_y)
    plt.xlabel("X(t) m", fontsize=label_font_size)
    plt.ylabel("Y(t) m", fontsize=label_font_size)
    plt.title("Path traced by the vehicle")
    plt.axis("equal")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
