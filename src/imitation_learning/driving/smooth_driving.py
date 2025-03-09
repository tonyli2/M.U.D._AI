import numpy as np

class DriveController:
    """
    A class to control and edit incoming moves, to smooth out the robot's driving.
    """

    """
    Constructor for DriveController class.

    :param l_samples: The number of samples held in linear.x buffer.
    :type l_samples: int.
    :param z_samples: The number of samples held in angular.z buffer.
    :type l_samples: int.
    """
    def __init__(self, l_samples, z_samples):
        self.linear_samples = l_samples
        self.li = 0
        self.linear_x_buffer = np.zeros(l_samples)
        self.linear_x_sum = 0

        self.z_samples = z_samples
        self.zi = 0
        self.angular_z_buffer = np.zeros(z_samples)
        self.angular_z_sum = 0

    """
    Smooths out a robot move, setting linear.x and angular.z to the arithmetic mean of the respective buffers.

    :param move: A Twist move, to be edited by this method.
    :type move: Twist_move.
    """
    def process_move(self, move):
        
        old = self.angular_z_buffer[self.zi]
        self.angular_z_sum += move.angular.z - old
        self.angular_z_buffer[self.zi] = move.angular.z
        self.zi = (self.zi + 1) % self.z_samples
        move.angular.z = self.angular_z_sum / self.z_samples

        old = self.linear_x_buffer[self.li]
        self.linear_x_sum += move.linear.x - old
        # print(f"Sum {self.linear_x_sum}")
        self.linear_x_buffer[self.li] = move.linear.x
        self.li = (self.li + 1) % self.linear_samples
        move.linear.x = self.linear_x_sum / self.linear_samples - 0.02 * abs(move.angular.z)
        print(f"Angular {move.angular.z}")
        print(f"Linear {move.linear.x}")