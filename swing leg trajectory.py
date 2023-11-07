import numpy as np
import matplotlib.pyplot as plt

class SwingLegTrajectory:
    def __init__(self, initial_x, mid_x, final_x, initial_z, mid_z, final_z, step_time):
        """
        Initialize the SwingLegTrajectory.

        Args:
        - initial_x, mid_x, final_x: Initial, middle, and final x positions.
        - initial_z, mid_z, final_z: Initial, middle, and final z positions.
        - step_time: Time duration for each step.
        """
        self.initial_x = initial_x
        self.mid_x = mid_x
        self.final_x = final_x
        self.initial_z = initial_z
        self.mid_z = mid_z
        self.final_z = final_z
        self.step_time = step_time

    def generate_trajectory(self, num_points):
        """
        Generate the swing leg trajectory with Gaussian profiles in x and z directions.

        Args:
        - num_points: The number of points to generate along the trajectory.

        Returns:
        - x, x_velocity, z, and z_velocity trajectories as lists.
        """
        t_values = np.linspace(0, self.step_time, num_points)

        # Generate x velocity as a Gaussian function
        x_velocity = self.generate_gaussian_velocity_profile(t_values)
        x_position = self.integrate_velocity_profile(x_velocity, t_values)

        # Generate z position as a Gaussian function and calculate z velocity
        z_position = self.generate_gaussian_position_profile(t_values)
        z_velocity = np.gradient(z_position, t_values)

        return x_position, x_velocity, z_position, z_velocity

    def generate_gaussian_velocity_profile(self, t_values):
        sigma = self.step_time / 7  # Standard deviation
        mu = self.step_time / 2  # Mean
        x_velocity = np.exp(-(t_values - mu) ** 2 / (2 * sigma ** 2))
        return x_velocity

    def integrate_velocity_profile(self, velocity_profile, t_values):
        x_position = np.cumsum(velocity_profile) * (t_values[1] - t_values[0])
        x_position = x_position - x_position[0] + self.initial_x
        return x_position

    def generate_gaussian_position_profile(self, t_values):
        sigma_z = self.step_time / 7  # Standard deviation
        mu_z = self.step_time / 2  # Mean
        z_position = np.exp(-(t_values - mu_z) ** 2 / (2 * sigma_z ** 2))
        return z_position



    def plot_trajectory(self, num_points):
        x_position, x_velocity, z_position, z_velocity = self.generate_trajectory(num_points)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(x_position, label='X Position')
        plt.plot(z_position, label='Z Position')
        plt.xlabel('Time Steps')
        plt.ylabel('Position')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(x_velocity, label='X Velocity')
        plt.plot(z_velocity, label='Z Velocity')
        plt.xlabel('Time Steps')
        plt.ylabel('Velocity')
        plt.legend()


        plt.tight_layout()
        plt.show()

# Example usage:
initial_x = 0.0
mid_x = 0.5
final_x = 1.0
initial_z = 0.0
mid_z = 0.2
final_z = 0.0
step_time = 1.0
num_points = 100

trajectory_generator = SwingLegTrajectory(initial_x, mid_x, final_x, initial_z, mid_z, final_z, step_time)
trajectory_generator.plot_trajectory(num_points)
