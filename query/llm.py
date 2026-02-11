import os
import json
from dotenv import load_dotenv

from log.logger import get_logger
logger = get_logger(log_name= 'llm', log_path = 'llm.log')

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


from openai import OpenAI


def log_llm(function,system,prompt,response):
    logger.info('*'*50)
    logger.info('*'*50)
    logger.info('-'*20 + function + '-'*20)
    logger.info('SYSTEM:\n' + system)
    logger.info('PROMPT:\n' + prompt)
    logger.info('RESPONSE:\n' + response)
    logger.info('-'*50)
    logger.info('-'*50)


class LLM:

    client = OpenAI()
    
    @staticmethod
    def generate_text(system_prompt: str, user_prompt: str) -> dict:
        response = LLM.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        result = response.choices[0].message.content.strip()

        log_llm('generate_text',system_prompt,user_prompt,result)
    
        return result

    @staticmethod
    def generate_json(system_prompt: str, user_prompt: str, schema:dict) -> dict:
        response = LLM.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_schema", "json_schema": schema}
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        result_string = json.dumps(result,indent=2)
        log_llm('generate_json',system_prompt,user_prompt,result_string)

        return result
