# OPENAI Client API Calling

import openai
from openai import OpenAI

OpenAI_API_KEY = "" # API Key
openai.api_key = OpenAI_API_KEY
client = OpenAI(api_key = OpenAI_API_KEY)
