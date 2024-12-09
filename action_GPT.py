# Instructions
# Make Actions with GPT by Using Calculated Probabilities, Vison Text, Audio Text 

from client import client
import json
from robot_action import action_list
from audio_Whisper import whisper
from vision_text_GPT import vision_to_text
from embedding import action_embedding, get_embedding
 
# Instructions
instructions = """
    You are an assistant who will be given a string with natural language and other context information such as actions with probabilities and related data.
    Based on the user's audio transcription, image information, and calculated action probabilities, identify the most relevant actions and translate them into robot commands.
    
    Actions include: "go_straight", "go_backward", "turn_left", "turn_right", "grip_object", "release_object".
    You should include parameters for each command:
    - "distance_cm" for go_straight and go_backward.
    - "turning_angle" for turn_left and turn_right.
    - "gripping_power" for grip_object and release_object.

    You must only return a JSON array of the commands with parameters in the format:
    [
        {"command": "<command_name>", "parameters": {"<parameter_name>": <parameter_value>}}
    ]
    
    In case the input is unclear, respond with:
    {"commands": [{"command": "unknown", "parameters": {}}]}

    Additional Context Information:
    - Audio Transcription: {audio_transcription}
    - Image Information: {image_information}
    - Action Probabilities: {calculated_action_probabilities}

    Consider the following:
    - Use the action probabilities to decide the order and relevance of the commands.
    """

def gpt_action(image_file_idx, audio_file_idx):

    audio_text = whisper(audio_file_idx) # Audio to Text
    audio_text = "go straight and grip the object" # For Test, Delete!
    vision_text = vision_to_text(image_file_idx) # Vision to Text
    
    audio_text_emb = get_embedding(audio_text) # Emb1
    vision_text_emb = get_embedding(vision_text) # Emb2 
    
    probs, _ = action_embedding(action_list, audio_text_emb, vision_text_emb) # Probs
    
    # Data Structure
    data = {
        "audio_transcription": audio_text,
        "image_information": vision_text, 
        "calculated_action_probabilities": [
            {"action": action, "probability": prob} for action, prob in zip(action_list, probs)]
        }
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps({
                    "audio_transcription": data["audio_transcription"],
                    "image_information": data["image_information"],
                    "calculated_action_probabilities": data["calculated_action_probabilities"]
                })
            },
        ],
        temperature=0.0
    )

    try:
        serverMessage = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        serverMessage = {"commands": [{"command": "unknown", "parameters": {}}]}
    return serverMessage, probs, audio_text_emb, vision_text_emb
    
