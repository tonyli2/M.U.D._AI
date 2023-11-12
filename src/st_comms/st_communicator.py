#! /usr/bin/env python3

import rospy
from std_msgs.msg import String

def main():

    # Register a ROS node called 'score_tracker_publisher' with the master node
    rospy.init_node('score_tracker_publisher', anonymous=True)

    # Subscribe to score tracker
    publisher = rospy.Publisher('/score_tracker', String, queue_size=10)

    # Set a rate to publish messages
    rate = rospy.Rate(1)  # 1 Hz

    # Main loop to publish messages
    while not rospy.is_shutdown():

        publisher.publish("Hello World!")
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

