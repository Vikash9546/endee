import os
from dotenv import load_dotenv
from google import genai

# Load .env to ensure GEMINI_API_KEY is in environment
load_dotenv(".env")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
    )
    print("--- SUCCESS ---")
    print(response.text)
except Exception as e:
    print(f"--- FAILED ---\n{e}")
