# Converts a Twist message to a dictionary
def twist_2_dict(twist):

    t_dict = {
        "linear": {
            "x": twist.linear.x,
            "y": twist.linear.y,
            "z": twist.linear.z
        },
        "angular": {
            "x": twist.angular.x,
            "y": twist.angular.y,
            "z": twist.angular.z
        }
    }

    return t_dict