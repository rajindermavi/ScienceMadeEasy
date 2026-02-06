import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


from openai import OpenAI


class LLM:

    client = OpenAI()
    
    @staticmethod
    def generate_text(prompt: str) -> str:
        response = LLM.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def generate_json(prompt: str,schema:dict) -> str:
        response = LLM.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": schema}
        )
        return response.choices[0].message.content.strip()