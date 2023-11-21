#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist

# Simulation time
sim_time = 0

def main():

    # Register a ROS node called 'score_tracker_publisher' with the master node
    rospy.init_node('score_tracker_publisher', anonymous=True)

    # Publisher to score tracker
    st_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

    # Subscribe to Clock
    clk_sub = rospy.Subscriber('/clock', Clock, clk_callback)

    # Publisher to cmd_vel
    cmd_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size = 5)

    # Set a rate to publish messages
    rate = rospy.Rate(1)  # 1 Hz

    # Variables to publish to Score tracker
    start_timer = 0
    end_timer = -1
    string_format = "03,IAmHuntersDad,{0}, NA"

    # Main loop to pass time trials
    first_message_sent = False
    last_message_sent = False
    while not rospy.is_shutdown():

        # Wait a set amount of time before starting timer
        if sim_time < 10:
            continue

        # Start the timer and move forward
        elif sim_time >= 10 and sim_time < 15:

            # Start the timer if we have not yet
            if not first_message_sent:

                pub_String = string_format.format(start_timer)
                st_pub.publish(pub_String)
                first_message_sent = True
            
            # Move robot forward
            move_fwd = Twist()
            move_fwd.linear.x = 0.5
            cmd_pub.publish(move_fwd)

        # End timer, Stop robot
        elif sim_time >= 15:

            if not last_message_sent:

                pub_String = string_format.format(end_timer)
                st_pub.publish(pub_String)
                last_message_sent = True
            
            move_stop = Twist()
            cmd_pub.publish(move_stop)
        
        rate.sleep()

def clk_callback(clk):

    global sim_time
    sim_time = clk.clock.secs


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass