import os
from dotenv import load_dotenv
from google import genai

# Load .env from the assignment directory
load_dotenv(os.path.join(os.getcwd(), ".env"))

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

try:
    # Testing with a stable model first
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents="Explain how AI works in a few words"
    )
    print("--- SUCCESS ---")
    print(response.text)
except Exception as e:
    print(f"--- FAILED ---\n{e}")
