# Main
# By Using Action GPT, Make robot move

from camera import camera_main
from audio_record import audio_record
from action_GPT import gpt_action
from train import train
import time
import time
from robomaster import robot
import robot_action

command_mapping = {
    'go_straight': robot_action.go_straight,
    'go_backward': robot_action.go_backward,
    'turn_left': robot_action.turn_left,
    'turn_right': robot_action.turn_right,
    'grip_object': robot_action.grip_object,
    'release_object': robot_action.release_object
}

def execute_command(func, args):
    try:
        if args: 
            return func(*args)
        return func()
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def robot_control(commands, index=0):
    if index >= len(commands):
        return
    cmd = commands[index]
    cmd_name = cmd["command"]
    parameters = cmd.get("parameters", {})

    func = command_mapping.get(cmd_name)
    if func:
        print(f"Executing {func.__name__} with parameters {parameters}")
        success = execute_command(func, list(parameters.values()))
        if success:
            print(f"Successfully executed {func.__name__}.")
            time.sleep(5)
        else:
            print(f"Failed to execute {func.__name__}.")
        robot_control(commands, index + 1)
    else:
        print(f"Unknown command: {cmd_name}")
        robot_control(commands, index + 1)

def robot_execute(serverReceived): # Execute
    for command in serverReceived:
        print("Executing command:", command)
        time.sleep(1)
        robot_control([command])



file_idx = 0

while True:
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")
    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_gripper = ep_robot.gripper

    print(f"Processing episode {file_idx}")
    
    audio_record(ep_robot, ep_camera, file_idx)
    camera_main(file_idx, ep_camera) # Need to Fix!
    time.sleep(1)

    serverReceived, probs, audio_text_emb, vision_text_emb, data = gpt_action(file_idx, file_idx)

    robot_execute(serverReceived)
    train(probs, audio_text_emb, vision_text_emb)
    file_idx += 1
    ep_robot.close()
    flag = input("Do you want to continue training? (yes/no): ").strip().lower()
    if flag not in ['yes', 'y']:
        print("Exiting training loop.")
        break 
