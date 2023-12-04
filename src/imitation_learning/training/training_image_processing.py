import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


# Pre-process sprinkling before sending it off as training data
def export_frame(ros_image, twist_dict, camera_time, mask_number):
    
    bridge = CvBridge()

    # Convert Image datatype to numpy array
    img_cv2 = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    # Process image so it is ready to be thrown into the CNN
    export_ready = process_img(img_cv2, mask_number)

    # Get name for image based on its corresponding twist message
    img_name = get_img_name(twist_dict, camera_time)

    # Export image as jpeg with specific name indexing
    cv2.imwrite(f'training_imgs/{img_name}.jpeg', export_ready)

    # Testing purposes
    ros_test_img = bridge.cv2_to_imgmsg(export_ready, "mono8")

    # return ros_img for debugging purposes
    return ros_test_img


# Apply filters, and rescale image to be CNN ready
def process_img(img_cv2, mask_number):

    # Three different masks for 3 parts of the environment
    mask_road = 0
    mask_grass = 1
    mask_offroad = 2

    if mask_number == mask_road:
        return process_road_img(img_cv2)
    
    elif mask_number == mask_grass:
        return process_grass_road(img_cv2)
    
    # elif mask_number == mask_offroad:
    #     return None
    #     # return process_off_road(img_cv2)


# Processes the image so that the road is highlighted
def process_road_img(img_cv2):

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


def process_grass_road(img_cv2):

    # Change image to Gray and apply blur
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (19, 19), 0)
    
    # Get image with white highlighted
    white = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)[1]
    white = cv2.erode(white, None, iterations=1)

    # # Get the dimensions of the image
    height, width = white.shape[:2]

    # Remove the sky from the contours by making all the top pixels dark
    white[:height//2,:] = 0

    # Find all contours in the image and sort them by area
    contours = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Create the blank image that we will draw on
    blank = np.zeros((height, width, 3), np.uint8)

    # Get the 5 largest contours on the image
    largest_contours = sorted_contours[:5]

    # Get bounding rectangles for the 5 largest contours
    pairings = [(contour, cv2.boundingRect(contour)) for contour in largest_contours]

    # Sort these rectangles by their top edge's proximity to the middle row
    # Rect objects return x, y, w, h so rect[1][1] returns the rect of the pairing 
    # and its top most edge of the rect
    pairings.sort(key=lambda pair: abs(pair[1][1] - (height//2)))
    valid_pairings = list()
    for i in range(len(pairings)):
        
        # Get the specified contour and its correspondign box from each pairing
        contour = pairings[i][0]
        contour_box = pairings[i][1]
        x, y, w, h = contour_box
        
        # Look at each box in the pairings, and only keep the ones that
        # have at least one of their sides on the edge of the image 
        # edges are x = 0 (left of frame), y = height (bottom), and x = width (right)
        if(x == 0 or (x + w) == width or (y + h) == height):
            valid_pairings.append(pairings[i])
    
    # Find which remaining contours are closest to the midline
    valid_pairings.sort(key=lambda pair: abs(pair[1][1] - (height//2)))

    # Choose to only draw the top 2 (corresponds hopefully to left and right road mark)
    for contour, contour_box in valid_pairings[:2]:
        x, y, w, h = contour_box
        cv2.drawContours(blank, [contour], -1, (255,255,255), -1)
        # cv2.rectangle(blank, (x, y), (x + w, y + h), (0, 0, 255), 2) Debugging line

    # Simplifying the data that the CNN will train on
    export_ready = cv2.resize(blank, (256,144))
    export_ready = cv2.cvtColor(export_ready, cv2.COLOR_BGR2GRAY)

    return export_ready



# Finds the pink strip in the image inputted
def find_pink(img_cv2):

    # Experimentally determined optimal threshold for pink strip
    upper_blue = 255
    upper_green = 0
    upper_red = 255

    lower_blue = 0
    lower_green = 0
    lower_red =  0

    lower_thresh = np.array([lower_blue, lower_green, lower_red])
    upper_thresh = np.array([upper_blue, upper_green, upper_red])

    export_ready = cv2.inRange(img_cv2, lower_thresh, upper_thresh)
    export_ready = cv2.dilate(export_ready, None, iterations=1)   

    return export_ready

# Finds and draws the largest contour while thresholding for baby yoda
def find_yoda(img_cv2):
    # Experimentally determined optimal thresholds for cactus
    upper_blue = 50
    upper_green = 42
    upper_red = 52

    lower_blue = 0
    lower_green = 0
    lower_red =  43

    lower_thresh = np.array([lower_blue, lower_green, lower_red])
    upper_thresh = np.array([upper_blue, upper_green, upper_red])

    yoda_img = cv2.inRange(img_cv2, lower_thresh, upper_thresh)
    yoda_img = cv2.dilate(yoda_img, None, iterations=1)

    height, width = yoda_img.shape[:2]

    contours = cv2.findContours(yoda_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # baby yoda has about 2000 units area, so only draw baby yoda on the export frame
    export_ready = np.zeros((height, width, 3), np.uint8)

    if len(sorted_contours) == 0:
        return export_ready, None
    else:
        yoda_contour = sorted_contours[0]

    # If the largest contour is not yoda it will not have a large area
    if cv2.contourArea(yoda_contour) > 1500: # Then this is baby yoda
        cv2.drawContours(export_ready, [yoda_contour], -1, (255,255,255), -1)

    else: # Return no contour if its not baby yoda
        yoda_contour = None
    
    export_ready = cv2.cvtColor(export_ready, cv2.COLOR_BGR2GRAY)

    return export_ready, yoda_contour

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