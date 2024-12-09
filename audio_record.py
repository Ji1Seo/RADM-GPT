# Robotmaster Audio Record Calling

from robomaster import robot

def audio_record(ep_robot, ep_camera, audio_file_idx): # File index
    file_name = f"./Audio_file/{audio_file_idx}.wav"
    print("recording")
    ep_camera.record_audio(save_file=file_name, seconds=8, sample_rate=16000) # Seconds, Rate
