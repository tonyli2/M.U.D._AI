#! /usr/bin/env python3

# Ros imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

# Math
import numpy as np

# Exporting
from training_image_processing import export_frame
from twist_serialization import twist_2_dict
from file_indexer import increment_file_idx


# Class to create training data for imitation learning
class input_reader:

    def __init__(self):

        # Subscribe to car control, camera and clock topics
        self.sub_twist = rospy.Subscriber('/R1/cmd_vel', Twist, self.twist_callback)
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)
        self.sub_clk = rospy.Subscriber('/clock', Clock, self.clk_callback)

        # Buffer for storing Twist and Image data
        self.twist_buffer = []
        self.camera_buffer = []

        # Time variables for buffers
        self.sim_time = None
        self.buffer_threshold = 0.5

        # Publisher to send binary masked camera feed
        self.pub = rospy.Publisher('/debugging_topic', Image, queue_size=5)

    
    # Everytime an input is received, we store it and what the camera sees
    def twist_callback(self, twist):

        if self.sim_time != None:
            
            # Get time stamp of input
            cur_time = self.sim_time

            # Put input into buffer
            self.twist_buffer.clear()
            self.twist_buffer.append((cur_time, twist))

            # Go and check for associated image data
            self.sync_twist_camera() 


    # Callback function for car control
    def camera_callback(self, image):

        # Get time stamp of input
        cur_time = self.sim_time

        # Put input into buffer
        self.camera_buffer.clear()
        self.camera_buffer.append((cur_time, image))



    def clk_callback(self, clk):
        
        # Keep track of simulation time
        self.sim_time = clk.clock.secs



    def sync_twist_camera(self):

        if len(self.twist_buffer) == 0 or len(self.camera_buffer) == 0:
            return
        
        twist_time, cur_twist = self.twist_buffer[0]
        camera_time, cur_image = self.camera_buffer[0]
        
        # Check if Twist and Image data captures the same time
        if(np.abs(twist_time - camera_time) < self.buffer_threshold):
            
            # Convert Twist message to a dictionary so we can use it to name
            # our image jpeg
            twist_dict = twist_2_dict(cur_twist)

            # Process & export current image
            debugging_img = export_frame(cur_image, twist_dict, camera_time)

            # Debugging purposes
            self.pub.publish(debugging_img)


def reader():

    # Registera ROS node called 'input_reader' with master node
    rospy.init_node('input_reader', anonymous=True)
    
    # Instantiate an object to use channels created under 'input_reader' node
    inp_reader = input_reader()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    reader()


