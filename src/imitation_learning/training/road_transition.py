#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# Import pink detector
from training_image_processing import find_pink

class road_transition():

    def __init__(self):
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.pink_callback)
        self.see_pink = rospy.Publisher('/pink_detector', String, queue_size=5)
        self.bridge = CvBridge()
        self.area_threshold = 30000

    def pink_callback(self, car_view_img):
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
                print('Pink area is ' + str(area))

                # Check if the area is larger than the threshold
                if area > self.area_threshold:
                    # Publish to the topic if area is greater than threshold
                    self.see_pink.publish('True')
                else:
                    self.see_pink.publish('False')
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
