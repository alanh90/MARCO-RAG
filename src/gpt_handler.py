# GPT Handler
# This file contains the implementation for handling interactions with the GPT models.
# It includes functionalities for sending prompts, receiving responses, and managing the conversation context.
# This handler is crucial for integrating language model capabilities into the BiCA framework.

import openai
from src import util


class GPTHandler:
    def __init__(self, model="gpt-3.5-turbo-1106"):
        self.openai_client = openai.OpenAI()
        self.openai_client.api_key = util.get_environment_variable("OPENAI_API_KEY")
        self.model = model

    def generate_response(self, prompt, temperature=0.7, max_tokens=150, top_p=1, frequency_penalty=0,
                          presence_penalty=0):
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content.strip()