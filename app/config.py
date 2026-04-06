import os
from dotenv import load_dotenv
from groq import Groq

# Load env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize client
client = Groq(api_key=GROQ_API_KEY)


def llm_call(prompt: str) -> str:
    """
    Wrapper for Groq LLM calls
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content