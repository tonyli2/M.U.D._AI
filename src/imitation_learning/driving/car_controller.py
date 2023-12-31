#! /usr/bin/env python3

# Ros imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String

# To control model response rate
import time

# General Imports
import tensorflow as tf
import numpy as np

# Image processing
from cv_bridge import CvBridge, CvBridgeError
import sys

# To import image_processing functions
sys.path.insert(1, '/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/training')

from training_image_processing import process_img


class car_controller():

    def __init__(self):

        # Subscribe camera topic
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)

        # Score tracker
        self.score_tracker = rospy.Publisher('/score_tracker', String, queue_size=5)

        # Subscribe to the pink flag topic
        self.first_pink_flag = rospy.Subscriber('/first_pink_detector', String, self.first_pink_callback)
        self.second_pink_flag = rospy.Subscriber('/second_pink_detector', String, self.second_pink_callback)
        self.parked_car_flag = rospy.Subscriber('/parked_car_detector', String, self.parked_car_callback)

        # Publish to 
        self.pub_twist = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=5)
        # Publish to start PID command
        self.start_PID = rospy.Publisher('/start_PID', String, queue_size=5)

        # Subscribe to the pedestrian signal command topic
        self.pedestrian_signal_cmd = rospy.Subscriber('/pedestrian_signal_cmd', String, self.pedestrian_callback)

        # Flag indicating if we are waiting for pedestrian to cross the road
        self.wait4pedestrian = False

        # Publishers for debugging model predicted commands
        self.pub_twist_debug = rospy.Publisher('/twist_debug', String, queue_size=5)
        self.pub_img_debug = rospy.Publisher('/img_debug', Image, queue_size=5)

        # Dictionary that maps Model outputs to real Twist messages
        self.moves_dictionary = None

        # CNN to drive car
        self.road_model = None
        self.grass_model = None
        self.mountain_model = None

        # TODO change from real time to sim time 

        # Timer to prevent overloading car inputs
        self.cur_time = None
        self.last_cmd_time = 0
        self.input_period = 100 # 100 millisecs

        # Road transition flags
        self.seen_first_pink = False
        self.seen_second_pink = False
        self.seen_parked_car = False

        # Start Competition timer
        self.start_timer = True
    
    # Change the state of the first pink flag
    def first_pink_callback(self, msg):
        # Set the first pink flag to True if the pink stripe is big enough
        if msg.data == 'True':
            self.seen_first_pink = True
    
    # Change the state of the second pink flag
    def second_pink_callback(self, msg):
        # Set the second pink flag to True if the pink stripe is big enough
        if msg.data == 'True':
            self.seen_second_pink = True
    
    # Change the state of the parked car flag
    def parked_car_callback(self, msg):
        # Set the parked car flag to True if the car is close enough
        if msg.data == 'True':
            self.seen_parked_car = True

    # Takes in a camera image and prepares it to be sent to CNN
    def camera_callback(self, car_view_img):

        # Update time
        self.update_cur_time()

        # Subscribe to a topic
        # Topic returns a boolean (Stirng)
        # If topic returns true the stop otherwise drive
        if self.wait4pedestrian == True:
            # Publish the stop command to find wait for pedestrian crossing road
            twist_command = self.moves_dictionary[4]
            self.pub_twist.publish(twist_command)
            return

        # Only input commands once ever 100 ms
        if self.cur_time - self.last_cmd_time > self.input_period:

            # Convert Image datatype to numpy array
            bridge = CvBridge()
            img_cv2 = bridge.imgmsg_to_cv2(car_view_img, "bgr8")
     
            # Process image so it is ready to be thrown into the CNN
            if not self.seen_first_pink:
                mask_number = 0
            elif self.seen_first_pink and not self.seen_second_pink:
                mask_number = 1
            elif self.seen_second_pink and not self.seen_parked_car:
                mask_number = 2
            elif self.seen_second_pink and self.seen_parked_car:
                mask_number = 2
            # This is our last resort to prevent mask_number being None and passed into process_img
            else:
                mask_number = 2
            # mask_number = 1
            model_ready_img = process_img(img_cv2, mask_number)

            # Debugging the input to model
            debug_img = bridge.cv2_to_imgmsg(model_ready_img, "mono8")
            self.pub_img_debug.publish(debug_img)

            # We need to reshape before passing into CNN            
            model_ready_img = model_ready_img.reshape((1, 144, 256, 1))

            # Drive! (But only if the model is ready)
            if self.road_model != None and self.grass_model != None and self.mountain_model != None:
                if not self.seen_first_pink:
                    # print('Driving on road')
                    self.road_drive(model_ready_img)
                    self.start_PID.publish('False')
                elif self.seen_first_pink and not self.seen_second_pink:
                    # print('Driving on grass')
                    self.grass_drive(model_ready_img)
                    self.start_PID.publish('False')
                elif self.seen_second_pink and not self.seen_parked_car:
                    # print('Driving on Baby Yoda PID')
                    self.start_PID.publish('True')
                elif self.seen_second_pink and self.seen_parked_car:
                    # print("Mountain")
                    self.start_PID.publish('Kill')
                    self.mountain_drive(model_ready_img)

                    # Stop the car so as not to move past the 2nd pink stripe
                    # self.pub_twist.publish(Twist())

            # Change last command time to cur_time
            self.last_cmd_time = self.cur_time

     # Set the pedestrian flag to indicate if robot should wait or go
    def pedestrian_callback(self, msg):
        print("Pedestrian callback received:", msg.data)  # Debugging print statement
        if msg.data == 'True':
            self.wait4pedestrian = True
            # print("Waiting for pedestrian...")  # More debugging
        else:
            self.wait4pedestrian = False
            # print("No pedestrian, continuing...")  # More debugging

    # Update time
    def update_cur_time(self):
        self.cur_time = int(time.time() * 1000)


    # Given the car's perspective through the camera, navigate the robot on road condition
    def road_drive(self, model_input):
        
        # print(model_input.shape)

        # Get output from CNN, will be a 1D np vector
        model_pred = self.road_model.predict(model_input)
        
        # Find predicted move by taking max value in output vector
        pred_idx = np.argmax(model_pred)

        # # Look at how confident model is for this max
        max_confidence = model_pred[0][pred_idx]

        if max_confidence < 0.8:
            return

        # Go to lookup table to convert from index to Twist Msg
        twist_command = self.moves_dictionary[pred_idx]

        # Publish to robot
        self.pub_twist.publish(twist_command)
    
    # Given the car's perspective through the camera, navigate the robot on grass condition
    def grass_drive(self, model_input):
        
        # print(model_input.shape)

        # Get output from CNN, will be a 1D np vector
        model_pred = self.grass_model.predict(model_input)
        
        # Find predicted move by taking max value in output vector
        pred_idx = np.argmax(model_pred)

        # # Look at how confident model is for this max
        max_confidence = model_pred[0][pred_idx]

        if max_confidence < 0.8:
            return

        # Go to lookup table to convert from index to Twist Msg
        twist_command = self.moves_dictionary[pred_idx]

        # Publish to robot
        self.pub_twist.publish(twist_command)
    
    # Given the car's perspective through the camera, navigate the robot on mountain condition
    def mountain_drive(self, model_input):
        # print("1")
        # print(model_input.shape)
        # Get output from CNN, will be a 1D np vector
        model_pred = self.mountain_model.predict(model_input)
        
        # Find predicted move by taking max value in output vector
        pred_idx = np.argmax(model_pred)

        # # Look at how confident model is for this max
        max_confidence = model_pred[0][pred_idx]

        if max_confidence < 0.8:
            return

        # Go to lookup table to convert from index to Twist Msg
        twist_command = self.moves_dictionary[pred_idx]

        # Publish to robot
        self.pub_twist.publish(twist_command)


    # Called at start to populate class variables
    def setup_controller(self, road_model_path, grass_model_path, mountain_model_path):

        # Load the desired version of the CNN
        self.road_model = tf.keras.models.load_model(road_model_path)
        self.grass_model = tf.keras.models.load_model(grass_model_path)
        self.mountain_model = tf.keras.models.load_model(mountain_model_path)

        while self.road_model == None or self.grass_model == None or self.mountain_model == None:
            print("Waiting for model to load . . .")
            continue
        
        # Start the competition timer
        if self.start_timer == True:

            self.start_timer = False
            start_timer = "03,IAmHuntersDad,0, NA"
            self.score_tracker.publish(start_timer)

        # Setup the dictionary that maps CNN outputs to real Twist Msgs
        self.moves_dictionary = self.model_output_to_Twist()



    # Creates a dictionary that maps the output of the CNN to Twist Msgs
    def model_output_to_Twist(self):
        
        # All possibe moves defined as Twist messages
        curve_left = Twist(linear=Vector3(x=0.5), angular=Vector3(z=1.0))
        forward = Twist(linear=Vector3(x=0.5), angular=Vector3(z=0.0))
        curve_right = Twist(linear=Vector3(x=0.5), angular=Vector3(z=-1.0))
        spin_ccw = Twist(linear=Vector3(x=0.0), angular=Vector3(z=1.0))
        stop = Twist(linear=Vector3(x=0.0), angular=Vector3(z=0.0))
        spin_cw = Twist(linear=Vector3(x=0.0), angular=Vector3(z=-1.0))
        curve_bleft = Twist(linear=Vector3(x=-0.5), angular=Vector3(z=-1.0))
        backward = Twist(linear=Vector3(x=-0.5), angular=Vector3(z=0.0))
        curve_bright = Twist(linear=Vector3(x=-0.5), angular=Vector3(z=1.0))

        # Map output vector indices to Twist messages
        moves_dict = {
            0: curve_left,
            1: forward,
            2: curve_right,
            3: spin_ccw,
            4: stop,
            5: spin_cw,
            6: curve_bleft,
            7: backward,
            8: curve_bright,
        }

        return moves_dict
       


def main():

    rospy.init_node('car_controller', anonymous=True)
    
    # Create an instance of our controller
    controller = car_controller()

    # Startup sequence of controller
    road_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/road_models/imit_model_5.1.h5"
    grass_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/grass_models/grass_model_2.0.h5"
    off_road_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/off_road_models/off_road_model_3.1.h5"
    controller.setup_controller(road_model_path, grass_model_path, off_road_model_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()

    
