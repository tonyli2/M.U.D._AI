#! /usr/bin/env python3
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class pedestrian_detector:
    def __init__(self):
        self.bridge = CvBridge()
        # Subscribes to the camera topic
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.pedestrian_callback)
        # Publishes to pedestrian topic to signal go or no go command
        self.pedestrian_signal_cmd = rospy.Publisher('/pedestrian_signal_cmd', String, queue_size=5)
        # Flag to indicate if the pedestrian has crossed the road
        self.has_crossed_road = False
        self.counter = 0
        self.area_history = []
        self.start_time = time.time()
    
    # Monitor the area of the second largest red stripe and switch on the pedestrian flag
    def pedestrian_monitor(self, area):
        print('Waiting for pedestrian...')
        # We wait for a bit to calculate the average area
        if self.counter < 10:
            self.area_history.append(area)
            self.counter += 1
            return
        else:
            avg_area = np.average(np.array(self.area_history))
            print('Avg area is ' + str(avg_area) + ' and curr area is ' + str(area))
            # Scenario 1: car stops and pedestrian is in the process of crossing the road
            if np.abs(avg_area - area) > 100:
                print('Starting timer...')
                # Wait for 1.5 seconds and keep going
                curr_time = time.time()
                if curr_time - self.start_time >= 1.5:
                    self.has_crossed_road = True
                    print('GO NOW!!!')
            # Scenario 2: car stops and pedestrian has already crossed the road
            # We don't do anything here because we are not risking colliding with pedestrian
            else:
                print('Pedestrian seems to be off the road but we are gonna wait for him to cross...')

    def pedestrian_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Convert color to HSV for thresholding
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            # Define the range of red color in HSV
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            lower_red = np.array([170, 120, 70])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            # Combine the masks for red color
            mask = mask1 + mask2

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Find the two largest contours and their areas
            if len(contours) >= 2:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                largest_contour = sorted_contours[0]
                second_largest_contour = sorted_contours[1]

                # Contour with largest area corresponds to the closest red stripe
                largest_area = cv2.contourArea(largest_contour)
                # Contour with second largest area corresponds to the red stripe further away
                second_largest_area = cv2.contourArea(second_largest_contour)

                # If area is too big then we know we are close to the cross so stop
                # True indicates wait for pedestrian to cross, False otherwise
                print('Largest red contour area is ' + str(largest_area))
                print('Second largest red contour are is ' + str(second_largest_area))
                if largest_area >= 30000 and not self.has_crossed_road:
                    self.pedestrian_signal_cmd.publish('True')
                    self.pedestrian_monitor(second_largest_area)
                else:
                    self.pedestrian_signal_cmd.publish('False')

                # # Draw the two largest contours
                # cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 3)
                # cv2.drawContours(cv_image, [second_largest_contour], -1, (0, 255, 0), 3)

                # # Display the areas of the two largest contours
                # cv2.putText(cv_image, f'Largest Area: {largest_area}', (cv_image.shape[1] - 500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(cv_image, f'Second Largest Area: {second_largest_area}', (cv_image.shape[1] - 500, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # If can't find two contours then robot is far from cross, don't need to wait and keep driving
            else:
                print('Less than 2 red contours detected')
                self.pedestrian_signal_cmd.publish('False')
        except CvBridgeError as e:
            print(e)

def main():
    rospy.init_node('pedestrian_detector', anonymous=True)
    pedestrian = pedestrian_detector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()