#! /usr/bin/env python3
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
import std_msgs.msg as std_msgs
import numpy as np
import sys
import time

# To import image_processing functions
sys.path.insert(1, '/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/training')

from training_image_processing import find_pink, find_yoda

class baby_yoda_follower:

    def __init__(self):

        self.pub_twist = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=5)
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)
        self.start_PID = rospy.Subscriber('start_PID', String, self.start_PID_callback)
        self.pub_debug = rospy.Publisher('/debugging_topic', Image, queue_size=5)
        self.bridge = CvBridge()
        self.ready_2_start = False
        self.done_start = False

        # Flags for when we can start PID
        self.pid_is_ready = False
        self.yoda_last_position = None
        self.yoda_last_time = 0
        self.yoda_cur_time = 0
        self.yoda_position_threshold = 5
        self.dynamic_straight = False # To determine if we have started the dynamic straightening
        self.dynamic_straight_counter = 0 # Count 10 loops before we start dynamic straightening

        # PID horizontal
        self.prev_deriv_h = 0
        self.kp_h = 0.008
        self.kd_h = 0.00005
        self.des_x_pos = int(1280 * 3/4)
        
        # PID vertical
        self.area_percentage = 0.55
        self.prev_deriv_v = 0
        self.kp_v = 2.25
        self.kd_v = 0.8

        # Integral Error and timing
        self.integral_sum = 0
        self.integral_prev_time = 0
        self.integral_cur_time = 0
        self.ki_v = 0.1

        self.prev_road_center = 0

        # Flag to indicate if we start PID
        self.start_PID_flag = False
    
    def start_PID_callback(self, msg):
        if msg.data == 'True':
            self.start_PID_flag = True

    def camera_callback(self,data):
        if self.start_PID_flag:
            try:
                
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                frame_height, frame_width, channels  = cv_image.shape
                # self.update_cur_time()
                # if not self.pid_is_ready:
                #     print("Waiting on yoda to get in position")
                #     self.is_yoda_in_pos(cv_image)
                # else: # If so, PID on it
                #     print("PID on yoda")
                #     self.yoda_pid(cv_image)
                # return
                # TODO uncomment once done with 90 degree turn cactus
                if not self.ready_2_start and not self.done_start:
                    print("Before Start")
                    self.start_sequence(cv_image, frame_width)

                elif self.ready_2_start and not self.done_start:
                    print("Ready to start")
                    # Drive forward and over the pink strip
                    self.drive_toward_pink(cv_image, frame_height)
                    
                elif self.ready_2_start and self.done_start:
                    # Update time
                    self.update_cur_time()
                    # Is Yoda in position?
                    if not self.pid_is_ready:
                        print("Waiting on yoda to get in position")
                        self.is_yoda_in_pos(cv_image)
                    else: # If so, PID on it
                        print("PID on yoda")
                        self.yoda_pid(cv_image)
                else:
                    print("something went wrong this is the else condition")

                # self.pub.publish(movement)
            except CvBridgeError as e:
                print(e)
    
    # Update time
    def update_cur_time(self):
        self.yoda_cur_time = int(time.time() * 1000)


    def start_sequence(self, img_cv2, frame_width):
        
        # Find the pink strip
        # Experimentally determined optimal threshold for pink strip
        find_pink_img = find_pink(img_cv2)
        find_pink_img = cv2.GaussianBlur(find_pink_img, (5, 5), 0)

        # draw the contour box around this strip
        contours = cv2.findContours(find_pink_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # The first one will be the strip
        pink_strip = sorted_contours[0]

        # Compute how much we have to turn to straighten pink strip
        turning_value = self.straighten_pink_strip(find_pink_img, pink_strip, frame_width)

        if turning_value != 0:
            return
        else:
            # We are ready to start
            self.ready_2_start = True



    def drive_toward_pink(self, img_cv2, frame_height):

        # The move to be published at the end
        movement = Twist()

        # Find the pink strip
        # Experimentally determined optimal threshold for pink strip
        find_pink_img = find_pink(img_cv2)
        find_pink_img = cv2.GaussianBlur(find_pink_img, (5, 5), 0)

        # draw the contour box around this strip
        contours = cv2.findContours(find_pink_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(contours) == 0:
            
            movement.linear.x = 0
            self.pub_twist.publish(movement)
            self.done_start = True

            return

        # The first one will be the strip
        pink_strip = sorted_contours[0]

        # Find the box around the pink
        x, y, w, h = cv2.boundingRect(pink_strip)

        # Get vertical center of the box
        center = y + h//2
        error = abs(center - frame_height)

        # If the error is very large we want to cap it at 50 so that the
        # fastest linear x speed is 0,01 * 50 = 0.5
        if error > 50:
            error = 50

        # Escape condition:
        if error < 10:
            movement.linear.x = 0     
            self.done_start = True
        # Move forward until this center is past the bottom of the image
        else:
            movement.linear.x = 0.01 * error

        self.pub_twist.publish(movement)

        # Debugging
        debug_img = self.bridge.cv2_to_imgmsg(find_pink_img, "mono8")
        self.pub_debug.publish(debug_img)


    # Once we are done advancing on the pink strip and are ready to PID onto yoda
    # We will wait until yoda is found and has been in the same position for half a second
    def is_yoda_in_pos(self, img_cv2):
        
        # Get most up to date contour
        yoda_img, yoda_contour = find_yoda(img_cv2)

        # If yoda is not in the frame keep waiting
        if(yoda_contour is None):
            return False
        
        # If this is the first time yoda is seen, we set the last seen position to now
        if self.yoda_last_position is None:
            self.yoda_last_position = yoda_contour
            self.yoda_last_time = self.yoda_cur_time
            return False

        if self.yoda_cur_time - self.yoda_last_time > 1000:
            # Get the current contour bounding box
            # Get previous contour bounding box
            yoda_x, yoda_y, yoda_w, yoda_h = cv2.boundingRect(yoda_contour)
            old_yoda_x, old_yoda_y, old_yoda_w, old_yoda_h = cv2.boundingRect(self.yoda_last_position)

            # Update all old variables
            self.yoda_last_position = yoda_contour
            self.yoda_last_time = self.yoda_cur_time

            # Check if yoda has been in the same position
            is_static = abs(yoda_x - old_yoda_x) < self.yoda_position_threshold and abs(yoda_y - old_yoda_y) < self.yoda_position_threshold

            # Start PID
            if is_static:
                self.pid_is_ready = True

        # Debugging
        debug_img = self.bridge.cv2_to_imgmsg(yoda_img, "mono8")
        self.pub_debug.publish(debug_img)

        return

    def straighten_pink_strip(self, find_pink_img, pink_strip_contour, frame_width):

        # find the line/vector of best fit through this strip, fitLine() will return the direction
        # of this normalized vector and the coordinates of a point on this vector 
        # we will use Euclidean distance: root of square of components (DIST_L2)
        vector_x, vector_y, x, y = cv2.fitLine(pink_strip_contour, cv2.DIST_L2, 0, 0.5, 0.01)

        # Calculate two points (far enough)
        lefty = int((-x * vector_y / vector_x) + y)
        righty = int(((frame_width - x) * vector_y / vector_x) + y)

        # Draw the line
        find_pink_img = cv2.cvtColor(find_pink_img, cv2.COLOR_GRAY2BGR)
        cv2.line(find_pink_img, (frame_width - 1, righty), (0, lefty), (0, 0, 255), 2)

        # Calculate the angle this vector makes with the horizontal. 
        angle = np.arctan(vector_y, vector_x) * 180 / np.pi

        # Positive angles correspond to us needing to turn right (negative AZ)
        # Negative Angles correspond to us needing to turn left (positive AZ)

        movement = Twist()

        if abs(angle) < 0.5:
            movement.angular.z = 0
        else:
            movement.angular.z = -0.1 * angle
        
        self.pub_twist.publish(movement)
        debug_img = self.bridge.cv2_to_imgmsg(find_pink_img, "bgr8")
        self.pub_debug.publish(debug_img)


        return movement.angular.z

    def yoda_pid(self, img_cv2):

        # Movement to be sent
        movement = Twist()

        # Get dimensions so we can PID position
        height, width = img_cv2.shape[:2]

        # Get image with only yoda in it (if he is not on screen it will be black)
        yoda_img, yoda_contour = find_yoda(img_cv2)

        # If yoda is not in the frame keep waiting
        if(yoda_contour is None):

            # Delete this later
            des_x = int(width * 3/4)
            yoda_img = cv2.cvtColor(yoda_img, cv2.COLOR_GRAY2BGR)
            cv2.line(yoda_img, (self.des_x_pos, 0),(self.des_x_pos, height - 1),(0,0,255), 2)

            debug_img = self.bridge.cv2_to_imgmsg(yoda_img, "bgr8")
            self.pub_debug.publish(debug_img)


            # print("0")
            # Send in empty stop move if we dont see yoda
            self.pub_twist.publish(Twist())
            return 

        angular_error = self.yoda_x_pos_error(yoda_contour, width)
        
        linear_error = self.yoda_y_pos_error(yoda_contour, width, height)

        movement.angular.z = angular_error
        movement.linear.x = linear_error
        self.pub_twist.publish(movement)
        # Delete this later
        yoda_img = cv2.cvtColor(yoda_img, cv2.COLOR_GRAY2BGR)
        cv2.line(yoda_img, (self.des_x_pos, 0),(self.des_x_pos, height - 1),(0,0,255), 2)

        # Delete this later
        x, y, w, h = cv2.boundingRect(yoda_contour)
        cv2.rectangle(yoda_img, (x,y), (x + w, y+ h), (0, 0, 255), 2)

        debug_img = self.bridge.cv2_to_imgmsg(yoda_img, "bgr8")

        # debug_img = self.bridge.cv2_to_imgmsg(yoda_img, "mono8")
        self.pub_debug.publish(debug_img)


    # We want yoda to be on the left most quarter of the screen
    # We also do not want yoda to far or close
    # This function determines the centre of yoda's contour and whether it is in the
    # spot we want in the camera image
    def yoda_x_pos_error(self, yoda_contour, frame_w):

        # Get the contours bounding box
        yoda_x, yoda_y, yoda_w, yoda_h = cv2.boundingRect(yoda_contour)

        # Calculate the centroid of the bounding box
        centre_x = (yoda_x + (yoda_x + yoda_w))//2

        # Ideally we want yoda to be on the right side of the screen
        desired_horizontal_pos = self.des_x_pos

        # Proportional Horizontal error
        # Positive means we need to turn left
        # Negative means we need to turn right
        cur_error = (desired_horizontal_pos - centre_x)

        # Derivative Horizontal error
        d_error = cur_error - self.prev_deriv_h
        self.prev_deriv_h = cur_error

        # Cap out the max error
        # TODO negative number
        if cur_error > 250.0:
            cur_error = 250.0
        elif cur_error < -250.0:
            cur_error = -250.0
        if d_error > 250.0:
            d_error = 250.0
        elif d_error < -250.0:
            d_error = -250.0
        
        final_error = self.kp_h * cur_error + self.kd_h * d_error

        return final_error


    # The veritcal placement of yoda is a bit different, instead of position, 
    # we will generate a PID control loop based on the area of the contour
    def yoda_y_pos_error(self, yoda_contour, frame_w, frame_h):

        # Get the contours bounding box
        yoda_x, yoda_y, yoda_w, yoda_h = cv2.boundingRect(yoda_contour)
        
        # Calculate the centroid of the bounding box
        area_yoda = (yoda_x + yoda_w) * (yoda_y + yoda_h)
        area_frame = frame_w * frame_h
        area_occupied = area_yoda/area_frame

        # Ideally we want yoda to take up about 38% of the screen
        desired_percentage = self.area_percentage 

        # Proportional Vertical Error, already Kp'd
        # Positive means drive forward
        # negative means drive backward
        cur_error = (desired_percentage - area_occupied) * self.kp_v

        # As yoda gets farther and farther away we want to move the
        # desired x position closer and closer to the middle
        # as yoda gets closer  we want to move the desired line
        # back toward the 3/4 width mark
        if not self.dynamic_straight and self.dynamic_straight_counter < 20:
            print("Start of Dynamic 150")
            dynamic_adjustment = 150 * abs(cur_error)
            self.dynamic_straight_counter += 1
            
            if self.dynamic_straight_counter == 20:
                self.dynamic_straight = True

        elif self.dynamic_straight and self.dynamic_straight_counter >= 20:
            dynamic_adjustment = 500 * abs(cur_error)
            print("Dynamic Adjustment: ", dynamic_adjustment)
        
        else:
            print("SOMETHING WENT WRONG")
            
            
        self.des_x_pos = int(frame_w * 3/4 - dynamic_adjustment)

        # There is an imbalance in the zoom, so all positive errors need
        # to be boosted to match the effect that a negative error has
        if(cur_error > 0):
            cur_error *= 1.5

        # Derivative Vertical Error
        d_error = cur_error - self.prev_deriv_v
        self.prev_deriv_v = cur_error

        # Integral Vertical Error, only comput once we are going straight
        if self.dynamic_straight:
            self.integral_sum += cur_error * (0.5)
        else:
            self.integral_sum = 0

        # Cap out the max error
        if cur_error > 0.9:
            cur_error = 0.9
        elif cur_error < -0.9:
            cur_error = -0.9

        if d_error > 0.5:
            d_error = 0.5
        elif d_error < -0.5:
            d_error = -0.5

        if self.integral_sum > 5:
            self.integral_sum = 5
        elif self.integral_sum < -5:
            self.integral_sum = -5

        if abs(cur_error) < 0.1:
            self.integral_sum = 0

        print(self.integral_sum * self.ki_v)

        final_error = cur_error - self.kd_v * d_error + self.ki_v * self.integral_sum

        return final_error
    
    # def dynamic_adjust_x_pos(self, yoda_contour):

    #     # Get the contours bounding box
    #     yoda_x, yoda_y, yoda_w, yoda_h = cv2.boundingRect(yoda_contour)
        
    #     # Calculate the centroid of the bounding box
    #     centre_y = (yoda_y + (yoda_y + yoda_h))//2

    #     return 

def main():

    rospy.init_node('yoda_follower', anonymous=True)
    follower_node = baby_yoda_follower()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
if __name__ == '__main__':
    main()