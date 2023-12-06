#! /usr/bin/env python3

# Ros imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String

# Image processing
from cv_bridge import CvBridge, CvBridgeError
import sys

# To import image_processing functions
sys.path.insert(1, '/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/training')

from training_image_processing import process_off_road

# Class used to debug the thresholding for various parts of the off_roading and grass_road sections
class grass_road():

    def __init__(self) -> None:

        # Publishers for debugging
        self.pub_img_debug = rospy.Publisher('/grass_debug', Image, queue_size=5)
        # Subscribe camera topic
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)

    def camera_callback(self, car_view_img):

        bridge = CvBridge()
        image_to_process = bridge.imgmsg_to_cv2(car_view_img, "bgr8")
        model_ready = process_off_road(image_to_process)
        debug_img = bridge.cv2_to_imgmsg(model_ready, "mono8")
        self.pub_img_debug.publish(debug_img)
        

def main():

    rospy.init_node('grass_roads', anonymous=True)

    grass = grass_road()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


   

if __name__ == '__main__':
    main()
