# Vision to Text, GPT

from client import OpenAI_API_KEY
import base64
import requests

def image_encode(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def vision_to_text(image_file_idx):
    image_path = f"./Image_file/{image_file_idx}.jpg"
    base64_image = image_encode(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OpenAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
        {
            "role": "user",   
            "content": [
            {
                "type": "text",
                "text": "The image on the right is the image of Depth information on the left image. The depth of the center point known as 50cm. Based on this, explain the image by focusing on the objects with their distance and direction"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if 'choices' in response.json():
        for choice in response.json()['choices']:
            vision_text = choice['message']['content']
            
    return vision_text
