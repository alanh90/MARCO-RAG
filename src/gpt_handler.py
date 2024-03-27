import openai
from src import util
from lm_interface import LanguageModelInterface


class GPTHandler(LanguageModelInterface):
    def __init__(self, model="gpt-3.5-turbo-1106"):
        self.openai_client = openai.OpenAI()
        self.openai_client.api_key = util.get_environment_variable("OPENAI_API_KEY")
        self.model = model

    def generate_response(self, prompt, max_tokens=100, temperature=0.7, n=1, stop=None):
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )
        if n == 1:
            return response.choices[0].message.content.strip()
        else:
            return [choice.message.content.strip() for choice in response.choices]

    def generate_summary(self, text, max_tokens=100, temperature=0.7, n=1, stop=None):
        prompt = f"Please provide a summary of the following text:\n\n{text}"
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )
        if n == 1:
            return response.choices[0].message.content.strip()
        else:
            return [choice.message.content.strip() for choice in response.choices]

    def generate_embeddings(self, text):
        response = self.openai_client.embeddings.create(
            model=self.model,
            input=text
        )
        return response["data"][0]["embedding"]