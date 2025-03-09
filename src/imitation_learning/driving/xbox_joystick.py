#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import evdev
import time
import fcntl, os

class JoyNode:
    def __init__(self, device_path):
        rospy.init_node('joy_node', anonymous=True)
        
        self.publisher = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(240)  # 240 Hz

        self.last_event_time = time.time()

        # Open the Xbox controller input device
        self.controller = evdev.InputDevice(device_path)

        # Manually set the file descriptor as non-blocking
        fd = self.controller.fd
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Define joystick & trigger event codes
        self.JOY_CODE = evdev.ecodes.ABS_X   # Left joystick X-axis
        self.TRIG_CODE = evdev.ecodes.ABS_GAS  # Right trigger

        # Initial values
        self.trigger = 0     # range: [0..1023]
        self.joy_x   = 32767 # range: [0..65535]
        self.deadzone = 2000

        # Slew-rate limits and stored speeds
        self.prev_lin_speed = 0.0
        self.prev_ang_speed = 0.0
        self.last_pub_time  = time.time()

        self.run_joystick_loop()

    def run_joystick_loop(self):
        """Main loop: read events in a non-blocking fashion; stop if idle."""
        while not rospy.is_shutdown():
            events = []
            while True:
                try:
                    batch = self.controller.read()  # read all available events
                    if not batch: 
                        break
                    events.extend(batch)
                except OSError as e:
                    # Errno 11 means "Resource temporarily unavailable" (no events)
                    if e.errno == 11:
                        break
                    else:
                        raise  

            if events:
                self.last_event_time = time.time()
                for event in events:
                    self.handle_event(event)
                self.publish_command()
            else:
                # No new events this cycle; check if we've been idle too long
                if time.time() - self.last_event_time > 0.5:
                    # Publish zero Twist (stop) if idle > 0.5s
                    self.publisher.publish(Twist())
                    self.prev_lin_speed = 0.0
                    self.prev_ang_speed = 0.0

            self.rate.sleep()

    def handle_event(self, event):
        """Parse the evdev event and update local joystick/trigger variables."""
        if event.type == evdev.ecodes.EV_ABS:
            if event.code == self.JOY_CODE:
                self.joy_x = event.value
            elif event.code == self.TRIG_CODE:
                self.trigger = event.value

        # Apply dead zone for the left joystick’s X-axis
        if abs(self.joy_x - 32767) < self.deadzone:
            self.joy_x = 32767

    def publish_command(self):
        """Compute the Twist based on current joystick state and publish."""
        # Normalize joystick
        norm_joy_x = (self.joy_x - 32767) / 32767.0
        norm_trigger = self.trigger / 1023.0

        # Desired velocities before smoothing
        desired_rotation_velo = -norm_joy_x * 4.5           # up to ~ 5 rad/s
        desired_forward_velo  = 2.0 * norm_trigger - 0.05 * desired_rotation_velo         # up to ~ 2 m/s

        #-------------------------------------
        #      SLEW-RATE LIMITING SECTION
        #-------------------------------------
        current_time = time.time()
        dt = current_time - self.last_pub_time
        self.last_pub_time = current_time

        # You can set your own acceleration/deceleration limits (m/s^2).
        max_lin_accel = 2.0  # m/s^2  (how quickly you can speed up)
        max_lin_decel = 2.0  # m/s^2  (how quickly you can slow down)


        # Linear velocity smoothing
        lin_diff = desired_forward_velo - self.prev_lin_speed
        if lin_diff > 0:
            # Accelerating forward
            max_change = max_lin_accel * dt
            if lin_diff > max_change:
                desired_forward_velo = self.prev_lin_speed + max_change
        else:
            # Slowing down
            max_change = max_lin_decel * dt
            if lin_diff < -max_change:
                desired_forward_velo = self.prev_lin_speed - max_change

        # Update stored speeds
        self.prev_lin_speed = desired_forward_velo

        #-------------------------------------
        #      END SLEW-RATE LIMITING
        #-------------------------------------

        # Construct and publish Twist
        twist = Twist()
        twist.linear.x = desired_forward_velo
        twist.angular.z = desired_rotation_velo
        self.publisher.publish(twist)


if __name__ == '__main__':
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    controller_path = None

    for dev in devices:
        if 'Xbox' in dev.name:
            controller_path = dev.path
            print(f"Found Xbox Controller at {controller_path}")
            break

    if not controller_path:
        raise Exception("Controller not connected.")

    try:
        JoyNode(controller_path)
    except rospy.ROSInterruptException:
        pass