import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from file_indexer import get_file_idx


# Pre-process sprinkling before sending it off as training data
def export_frame(ros_image):
    
    bridge = CvBridge()
    file_index = get_file_idx()

    img_cv2 = bridge.imgmsg_to_cv2(ros_image, "bgr8")

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

    # Export image as jpeg with specific name indexing
    cv2.imwrite(f'training_imgs/{file_index}.jpeg', export_ready)

    # Testing purposes
    ros_test_img = bridge.cv2_to_imgmsg(export_ready, "bgr8")

    # return ros_img for debugging purposes
    return ros_test_img


# TODO create data augmentation functionality 