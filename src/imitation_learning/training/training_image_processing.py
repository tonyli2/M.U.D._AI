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

# def get_length_area(contour):
#     length = cv2.arcLength(contour, True)
#     # area = cv2.contourArea(contour)
#     return length

# Function to calculate straightness
def straightness(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

# Function to get length and straightness
def get_length_straightness(contour):
    length = cv2.arcLength(contour, True)
    straight = straightness(contour)
    return (length, straight)

# Takes in an np array of grass road and returns the mono8 processed np image
def process_grass_road(img_cv2):

    # Conver to Hue, Saturation, Value
    img_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    img_hsv_blur = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Mask that excludes everything but the white road
    lower_white = np.array([0, 0, 180])  # Lower value to include faint whites
    upper_white = np.array([180, 80, 220])
    white_mask = cv2.inRange(img_hsv_blur, lower_white, upper_white)

    # white_mask = cv2.erode(white_mask, None, iterations=2)
    white_mask = cv2.dilate(white_mask, None, iterations=2)

    # Get the dimensions of the image
    height, width = white_mask.shape[:2]

    # Calculate the midpoint of the height
    midpoint = height // 2

    # Set the top half of the image to black
    white_mask[0:midpoint, 0:width] = 0

    # Find how many contours it has
    contours = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # If we see two (left and right side of road)
    if len(sorted_contours) >= 2:
        
        # # Calculate the centroid (or any representative point) of each contour
        # m1 = cv2.moments(sorted_contours[0])
        # m2 = cv2.moments(sorted_contours[1])

        # if m1["m00"] != 0 and m2["m00"] != 0:  # Check for division by zero
        #     cx1 = int(m1["m10"] / m1["m00"])
        #     cx2 = int(m2["m10"] / m2["m00"])

        #     # Determine which contour is on the left and which is on the right
        #     if cx1 < cx2:
        #         left_contour = sorted_contours[0]
        #         right_contour = sorted_contours[1]
        #     else:
        #         left_contour = sorted_contours[1]
        #         right_contour = sorted_contours[0]
        # else:
        #     # Handle the case where a contour's area is zero
        #     pass

        left_contour = sorted_contours[0]
        right_contour = sorted_contours[1]

        # Make white_mask 3 channels
        white_mask = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        # white_mask = cv2.dilate(white_mask, None, iterations=2)
        # Create blank image to draw contour on
        blank = np.zeros((height, width, 3), np.uint8)

        # # Draw the contours on the image
        left = cv2.drawContours(blank, [left_contour], -1, (0, 255, 0), -1)
        right = cv2.drawContours(blank, [right_contour], -1, (0, 255, 0), -1)
        # third = cv2.drawContours(blank, [sorted_contours[2]], -1, (0, 255, 0), -1)
        # fourth = cv2.drawContours(blank, [sorted_contours[3]], -1, (0, 255, 0), -1)

        # result = cv2.bitwise_and(left, right)

    else:
        print("2 Contours not found")
        print(len(contours))


    return img_gray

def test(img_cv2):
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (19, 19), 0)
    
    white = cv2.threshold(blur, 165, 255, cv2.THRESH_BINARY)[1]

    # Get the dimensions of the image
    height, width = white.shape[:2]

    # Calculate the midpoint of the height
    midpoint = height // 2

    # Set the top half of the image to black
    white[0:midpoint, 0:width] = 0


    # lower_white = np.array([180, 180, 180], dtype="uint8")  # Lower value to include faint whites
    # upper_white = np.array([255, 255, 255], dtype="uint8")
    # white = cv2.inRange(img_cv2, lower_white, upper_white)

    white = cv2.erode(white, None, iterations=1)

    contours = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    one = sorted_contours[0]
    two = sorted_contours[1]
    three = sorted_contours[2]

    # Make white_mask 3 channels
    white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
    # white_mask = cv2.dilate(white_mask, None, iterations=2)
    # Create blank image to draw contour on
    blank = np.zeros((height, width, 3), np.uint8)

    # # Draw the contours on the image
    left = cv2.drawContours(blank, [one], -1, (0, 255, 0), -1)
    right = cv2.drawContours(blank, [two], -1, (0, 255, 0), -1)
    # third = cv2.drawContours(blank, [three], -1, (0, 255, 0), -1)
    
    # Find the extreme points for each contour
    left_extreme, _ = find_extreme_points(one, height, width)
    _, right_extreme = find_extreme_points(two, height, width)

    start_row = int(3 * height / 4)

    # Fill the area between contours in the bottom quarter
    for y in range(start_row, height):
        x_left = left_extreme[y]
        x_right = right_extreme[y]
        if x_right > x_left:  # Ensure there is space to fill
            print('test')
            blank[y, x_left:x_right] = (0, 255, 0) 


    return blank



# Function to find extreme left and right points on each row
def find_extreme_points(contour, height, width):
    # Initialize arrays to store the extreme points
    left_points = np.full(height, width, np.int32)
    right_points = np.zeros(height, np.int32)

    # Go through all points in the contour
    for point in contour:
        x, y = point[0]
        if x < left_points[y]:
            left_points[y] = x
        if x > right_points[y]:
            right_points[y] = x
    
    return left_points, right_points
