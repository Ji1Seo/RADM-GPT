# By Using Whisper Model, Convert Audio to Text

from client import client

def whisper(file_name): # Audio to Text
    with open(f"./Audio_file/{file_name}.wav", "rb") as audio_file:

        transcription = client.audio.translations.create(
            file = audio_file,
            model = "whisper-1",
            response_format = "text",
            temperature = 0,    
        )
    return transcription
