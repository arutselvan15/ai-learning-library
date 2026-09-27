from openai import OpenAI
import os
from dotenv import load_dotenv

# Load the .env file (looks in the current directory by default)
load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4.1",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)


