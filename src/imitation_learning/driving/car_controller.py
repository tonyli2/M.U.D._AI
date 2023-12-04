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
from training.training_image_processing import process_img
from cv_bridge import CvBridge, CvBridgeError

class car_controller():

    def __init__(self):

        # Subscribe camera topic
        self.sub_cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)

        # Publish to 
        self.pub_twist = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=5)

        # Publishers for debugging model predicted commands
        self.pub_twist_debug = rospy.Publisher('/twist_debug', String, queue_size=5)
        self.pub_img_debug = rospy.Publisher('/img_debug', Image, queue_size=5)

        # Dictionary that maps Model outputs to real Twist messages
        self.moves_dictionary = None

        # CNN to drive car
        self.model = None

        # TODO change from real time to sim time 

        # Timer to prevent overloading car inputs
        self.cur_time = None
        self.last_cmd_time = 0
        self.input_period = 100 # 100 millisecs

    # Takes in a camera image and prepares it to be sent to CNN
    def camera_callback(self, car_view_img):

        # Update time
        self.update_cur_time()

        # Only input commands once ever 100 ms
        if self.cur_time - self.last_cmd_time > self.input_period:

            # Convert Image datatype to numpy array
            bridge = CvBridge()
            img_cv2 = bridge.imgmsg_to_cv2(car_view_img, "bgr8")

            # Process image so it is ready to be thrown into the CNN
            mask_number = 1
            model_ready_img = process_img(img_cv2, mask_number)

            # Debugging the input to model
            debug_img = bridge.cv2_to_imgmsg(model_ready_img, "mono8")
            self.pub_img_debug.publish(debug_img)

            # We need to reshape before passing into CNN            
            model_ready_img = model_ready_img.reshape((1, 144, 256, 1))

            # Drive! (But only if the model is ready)
            if self.model != None:
                self.drive(model_ready_img)

            # Change last command time to cur_time
            self.last_cmd_time = self.cur_time

    # Update time
    def update_cur_time(self):
        self.cur_time = int(time.time() * 1000)


    # Given the car's perspective through the camera, navigate the robot
    def drive(self, model_input):
        
        # print(model_input.shape)

        # Get output from CNN, will be a 1D np vector
        model_pred = self.model.predict(model_input)
        
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
    def setup_controller(self, model_path):

        # Load the desired version of the CNN
        self.model = tf.keras.models.load_model(model_path)

        while self.model == None:
            print("Waiting for model to load . . .")
            continue
        
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
    road_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/road_models/imit_model_3.1.h5"
    grass_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/grass_models/grass_model_2.0.h5"
    # off_road_model_path = "/home/fizzer/ros_ws/src/controller_repo/src/imitation_learning/cnn_models/off_road_models/off_road_model_1.0.h5"
    controller.setup_controller(grass_model_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()

    
