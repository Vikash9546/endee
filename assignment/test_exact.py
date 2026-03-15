from google import genai
import os
from dotenv import load_dotenv

# Load .env
load_dotenv(".env")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

try:
    # Try gemini-2.0-flash first as it's common
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    )
    print("--- SUCCESS ---")
    print(response.text)
except Exception as e:
    print(f"--- FAILED ---\n{e}")
