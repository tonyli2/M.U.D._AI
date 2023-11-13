#! /usr/bin/env python3

# Ros imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

# Image processing imports
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


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

        # Image processing
        self.bridge = CvBridge()

        # Publisher to send binary masked camera feed
        self.pub = rospy.Publisher('/road_detect_camera', Image, queue_size=5)

    def twist_callback(self, twist):

        if self.sim_time != None:
            
            # Get time stamp of input
            cur_time = self.sim_time

            # Put input into buffer
            self.twist_buffer.clear()
            self.twist_buffer.append((cur_time, twist))

            # Go and check for associated image data
            self.sync_twist_camera()

            # TODO Change Twist msg to storable data type 


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
        
        # Check if Twist and Image data is within buffer threshold
        if(np.abs(twist_time - camera_time) < self.buffer_threshold):

            cv2_image = self.process_image(cur_image)

            ros_img = self.bridge.cv2_to_imgmsg(cv2_image, "bgr8")
            self.pub.publish(ros_img)

        

    def process_image(self, ros_image):
        
        img_cv2 = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        # Split image up into 3 channels
        red, green, blue = cv2.split(img_cv2)

        # Create gray mask by looking for pixels with similar RGB values
        # Also last condition removes gray pixels found on green trees
        gray_threshold = 5
        gray_mask = np.where((np.abs(red - green) < gray_threshold) 
                            & (np.abs(green - blue) < gray_threshold) 
                            & (np.abs(blue - red) < gray_threshold)
                            & (green > 60),
                            255, 0).astype(np.uint8)

        # Find all gray pixels in img
        thresholded_image = cv2.bitwise_and(img_cv2, img_cv2, mask=gray_mask)

        # Highlight only gray pixels
        inverted_img = cv2.bitwise_not(thresholded_image)

        cv2.erode(inverted_img, None, iterations=2)

        return inverted_img

        # # Check if Twist and Image data is available
        # if self.twist_timestamp != 0 and self.camera_timestamp != 0:

        #     # Check if Twist and Image data is within buffer threshold
        #     if self.twist_timestamp - self.camera_timestamp < self.buffer_threshold:

        #         # Store Twist and Image data in buffer
        #         self.twist_buffer.append(self.twist_timestamp)
        #         self.camera_buffer.append(self.camera_timestamp)

        #         # Reset Twist and Image timestamps
        #         self.twist_timestamp = 0
        #         self.camera_timestamp = 0






        


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


