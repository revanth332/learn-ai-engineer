from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from typing import Any
load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

def structured_call(query:str,output_structure:BaseModel):
    response = client.chat.completions.parse(
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": query}],
        response_format=output_structure
    )

    return response.choices[0].message.parsed

def call(messages:list[dict[str, Any]],tools:list[dict[str, Any]]):
    response = client.chat.completions.create(
            model="gemini-3.6-flash",
            messages=messages,
            tools=tools
        )
    
    return response.choices[0].message

