import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


# Pre-process sprinkling before sending it off as training data
def export_frame(ros_image, twist_dict, camera_time):
    
    bridge = CvBridge()

    # Convert Image datatype to numpy array
    img_cv2 = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # Process image so it is ready to be thrown into the CNN
    export_ready = process_img(img_cv2)

    # Get name for image based on its corresponding twist message
    img_name = get_img_name(twist_dict, camera_time)

    # Export image as jpeg with specific name indexing
    cv2.imwrite(f'training_imgs/{img_name}.jpeg', export_ready)

    # Testing purposes
    ros_test_img = bridge.cv2_to_imgmsg(export_ready, "mono8")

    # return ros_img for debugging purposes
    return ros_test_img


# Apply filters, and rescale image to be CNN ready
def process_img(img_cv2):

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

    # Remove residual specks
    cv2.erode(inverted_img, None, iterations=2)

    # Scale down the image to reduce cost
    export_ready = cv2.resize(inverted_img, (256,144))

    # Convert to Gray-scale to reduce dimensionality
    export_ready = cv2.cvtColor(export_ready, cv2.COLOR_BGR2GRAY)

    return export_ready


# Takes the images corresponding twist command and creates a name for it 
def get_img_name(twist_dict, current_time):

    # The format I will be using does not include Linear y, z nor does it 
    # include Angular x,y since those are not ever used to properly drive the car

    # The formatting will be Time_LX_{Linear X}_AZ_{Angular Z} 

    linear_x = twist_dict["linear"]["x"]
    linear_y = twist_dict["linear"]["y"]
    linear_z = twist_dict["linear"]["z"]

    angular_x = twist_dict["angular"]["x"]
    angular_y = twist_dict["angular"]["y"]
    angular_z = twist_dict["angular"]["z"]

    str_current_time = str(current_time)

    format = "T_" + str_current_time + "_LX_{0:.1f}_AZ_{1:.1f}"

    return format.format(linear_x, angular_z)


# TODO create data augmentation functionality 