import time
from robomaster import robot

ep_robot = robot.Robot()
ep_robot.initialize(conn_type="sta")
ep_chassis = ep_robot.chassis
ep_gripper = ep_robot.gripper
speed = 2000
slp = 1

action_list = ["go_straight", "go_backward", "turn_right", "turn_left", "grip_object", "release_object"]

def go_straight(distance_cm):
    try:
        distance_m = distance_cm / 100.0
        linear_speed = 0.4
        time_needed = distance_m / linear_speed
        wheel_diameter = 0.100
        wheel_circumference = 3.14 * wheel_diameter
        wheel_rpm = (linear_speed * 60) / wheel_circumference
        wheel_rpm = -int(wheel_rpm)
        ep_chassis.drive_wheels(w1=wheel_rpm, w2=wheel_rpm, w3=wheel_rpm, w4=wheel_rpm, timeout = time_needed)
        return True
    except Exception as e:
        print(f"Error in go_straight: {e}")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return False

def go_backward(distance_cm):
    try:
        distance_m = distance_cm / 100.0
        linear_speed = 0.4
        time_needed = distance_m / linear_speed
        wheel_diameter = 0.100
        wheel_circumference = 3.14 * wheel_diameter
        wheel_rpm = (linear_speed * 60) / wheel_circumference
        wheel_rpm = int(wheel_rpm)
        ep_chassis.drive_wheels(w1=wheel_rpm, w2=wheel_rpm, w3=wheel_rpm, w4=wheel_rpm)
        time.sleep(time_needed)
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return True
    except Exception as e:
        print(f"Error in go_straight: {e}")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return False

def turn_right(angle):
    try:
        ep_chassis.move(x=0, y=0, z=-angle, z_speed=45)
        time.sleep(4)
        return True
    except Exception as e:
        print(f"Error in turn_right: {e}")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return False

def turn_left(angle):
    try:
        ep_chassis.move(x=0, y=0, z=angle, z_speed=45)
        time.sleep(4)
        return True
    except Exception as e:
        print(f"Error in turn_right: {e}")
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        return False

def grip_object(gripping_power):
    try:
        ep_gripper.open(50)
        time.sleep(2)
        ep_gripper.close(50)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Error in grip_object: {e}")
        return False

def release_object(gripping_power):
    try:
        ep_gripper.open(50)
        time.sleep(2)
        ep_gripper.close(50)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Error in grip_object: {e}")
        return False





