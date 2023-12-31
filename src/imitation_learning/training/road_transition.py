#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

# Import pink detector
from training_image_processing import find_pink, find_parked_car

class road_transition():

    def __init__(self):
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.transition_callback)
        self.see_first_pink = rospy.Publisher('/first_pink_detector', String, queue_size=5)
        self.see_second_pink = rospy.Publisher('/second_pink_detector', String, queue_size=5)
        self.see_parked_car = rospy.Publisher('/parked_car_detector', String, queue_size=5)
        self.pub_img_debug = rospy.Publisher('/car_debug', Image, queue_size=5)

        self.bridge = CvBridge()
        # Contour area thresholds
        # Contour thresholds for different transitions
        self.first_area_threshold = 30000
        self.second_area_threshold = 30000
        self.parked_car_area_threshold = 1000
        # Flag to indicate if we already see the pink stripes
        self.seen_first_pink = False
        self.seen_second_pink = False
        # Time at which the robot sees the first pink stripe
        self.first_pink_time = None
        self.start_timer = False

    def transition_callback(self, car_view_img):
        try:
            # Convert Image datatype to numpy array
            img_cv2 = self.bridge.imgmsg_to_cv2(car_view_img, "bgr8")

            # Process the image to find pink
            binary_img = find_pink(img_cv2)
            height, width = binary_img.shape[:2]

            binary_img[:height//2, :] = 0


            # Find the contours on the binary image
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour based on the area
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                # print('Pink area is ' + str(area))

                # First check if the robot sees the first pink stripe
                if not self.seen_first_pink:
                    self.see_second_pink.publish('False')
                    # Check if the area is larger than the threshold
                    if area > self.first_area_threshold:
                        # Publish to the topic if area is greater than threshold
                        self.see_first_pink.publish('True')
                        # The robot has seen the first pink stripe
                        self.seen_first_pink = True
                        if not self.start_timer:
                            self.first_pink_time = time.time()
                            self.start_timer = True
                    else:
                        self.see_first_pink.publish('False')
                # If the robot is passed the second 
                else:
                    # Don't publish True to the second pink flag topic if it hasn't been a while
                    # since the robot sees the first pink stripe
                    # This is to prevent we mistakenly skip the grass section
                    if (time.time() - self.first_pink_time) < 5:
                        self.see_second_pink.publish('False')
                        return
                    # Check if the area is larger than the threshold
                    if area > self.second_area_threshold:
                        # Publish to the topic if area is greater than threshold
                        self.see_second_pink.publish('True')
                        self.seen_second_pink = True
                    else:
                        self.see_second_pink.publish('False')
                
                # If the robot has seen both pink stripes then we can proceed to the last transition
                if self.seen_first_pink and self.see_second_pink:
                    binary_img = find_parked_car(img_cv2)
                    height, width = binary_img.shape[:2]

                     # Debugging the input to model
                    debug_img = self.bridge.cv2_to_imgmsg(binary_img, "mono8")
                    self.pub_img_debug.publish(debug_img)
                    
                    # Find the contours on the binary image
                    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Find the largest contour based on the area
                        largest_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest_contour)
                        # print('Parked car area is ' + str(area))

                        if area > self.parked_car_area_threshold:
                            self.see_parked_car.publish('True')
                        else:
                            self.see_parked_car.publish("False")
                else:
                    self.see_parked_car.publish('False')
                
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

def main():
    rospy.init_node('road_transition', anonymous=True)
    rt = road_transition()  # Follow naming convention for instances

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
