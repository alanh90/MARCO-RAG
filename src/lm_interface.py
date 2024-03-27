from abc import ABC, abstractmethod


class LanguageModelInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt, max_tokens=100, temperature=0.7, n=1, stop=None):
        """
        Generate a response based on the given prompt.

        Args:
            prompt (str): The input prompt for generating the response.
            max_tokens (int): The maximum number of tokens to generate in the response.
            temperature (float): The temperature value for controlling the randomness of the generated response.
            n (int): The number of responses to generate.
            stop (list): A list of strings that, if encountered, will stop the generation process.

        Returns:
            str or list: The generated response or a list of generated responses.
        """
        pass

    @abstractmethod
    def generate_summary(self, text, max_tokens=100, temperature=0.7, n=1, stop=None):
        """
        Generate a summary of the given text.

        Args:
            text (str): The input text to summarize.
            max_tokens (int): The maximum number of tokens to generate in the summary.
            temperature (float): The temperature value for controlling the randomness of the generated summary.
            n (int): The number of summaries to generate.
            stop (list): A list of strings that, if encountered, will stop the generation process.

        Returns:
            str or list: The generated summary or a list of generated summaries.
        """
        pass

    @abstractmethod
    def generate_embeddings(self, text):
        """
        Generate embeddings for the given text.

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            list: A list of embeddings representing the input text.
        """
        pass
