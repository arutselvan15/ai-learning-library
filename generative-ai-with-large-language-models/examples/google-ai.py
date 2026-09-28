from google import genai
import os
from dotenv import load_dotenv

# Load the .env file (looks in the current directory by default)
load_dotenv()

client = genai.Client(
    # This is the default and can be omitted
    api_key=os.getenv("GOOGLE_API_KEY"),
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain generative AI in simple words."
)

print(response.text)


