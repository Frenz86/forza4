from abc import ABC
import logging
from typing import Dict, Type, Self, List
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

logger = logging.getLogger(__name__)


class LLMException(Exception):
    pass


class LLM(ABC):
    """
    An abstract superclass for interacting with LLMs - subclass for Claude and GPT
    """

    model_names = []

    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.client = None
        self.temperature = temperature
        self.reasoning_effort = None

    def send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        """
        Send a message
        :param system: the context in which this message is to be taken
        :param user: the prompt
        :param max_tokens: max number of tokens to generate
        :return: the response from the AI
        """

        result = self.protected_send(system, user, max_tokens)
        left = result.find("{")
        right = result.rfind("}")
        if left > -1 and right > -1:
            result = result[left : right + 1]
        return result

    def protected_send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        """
        Wrap the send call in an exception handler, giving the LLM 3 chances in total, in case
        of overload errors. If it fails 3 times, then it forfeits!
        """
        retries = 3
        while retries:
            retries -= 1
            try:
                return self._send(system, user, max_tokens)
            except Exception as e:
                logging.error(f"Exception on calling LLM of {e}")
                if retries:
                    logging.warning("Waiting 2s and retrying")
                    time.sleep(2)
        return "{}"

    def _send(self, system: str, user: str, max_tokens: int = 3000) -> str:
        """
        Send a message to the model - this default implementation follows the OpenAI API structure
        :param system: the context in which this message is to be taken
        :param user: the prompt
        :param max_tokens: max number of tokens to generate
        :return: the response from the AI
        """
        if self.reasoning_effort:
            response = self.client.chat.completions.create(
                model=self.api_model_name(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=self.reasoning_effort,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.api_model_name(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
        return response.choices[0].message.content

    def api_model_name(self) -> str:
        """
        Return the actual model_name to be used in the call to the API; strip out anything after a space
        """
        if " " in self.model_name:
            return self.model_name.split(" ")[0]
        else:
            return self.model_name

    @classmethod
    def model_map(cls) -> Dict[str, Type[Self]]:
        """
        Generate a mapping of Model Names to LLM classes, by looking at all subclasses of this one
        :return: a mapping dictionary from model name to LLM subclass
        """
        mapping = {}
        for llm in cls.__subclasses__():
            for model_name in llm.model_names:
                mapping[model_name] = llm
        return mapping

    @classmethod
    def all_supported_model_names(cls) -> List[str]:
        """
        Return a list of all the model names supported by all subclasses of this one.
        """
        return list(cls.model_map().keys())

    @classmethod
    def all_model_names(cls) -> List[str]:
        """
        Return a list of all the model names supported.
        Use the ones specified in the model_map, but also check if there's an env variable set that restricts the models
        """
        models = cls.all_supported_model_names()
        allowed = os.getenv("MODELS")
        print(f"Allowed models: {allowed}")
        if allowed:
            allowed_models = allowed.split(",")
            return [model for model in allowed_models if model in models]
        else:
            return models

    @classmethod
    def create(cls, model_name: str, temperature: float = 0.5) -> Self:
        """
        Return an instance of a subclass that corresponds to this model_name
        :param model_name: a string to describe this model
        :param temperature: the creativity setting
        :return: a new instance of a subclass of LLM
        """
        subclass = cls.model_map().get(model_name)
        if not subclass:
            raise LLMException(f"Unrecognized LLM model name specified: {model_name}")
        return subclass(model_name, temperature)



class OpenRouter(LLM):
    """
    A class to act as an interface to OpenRouter API
    """
    model_names = [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.4",
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.1-flash-lite-preview",
        "deepseek/deepseek-chat",
    ]

    def __init__(self, model_name: str, temperature: float):
        """
        Create a new instance of the OpenAI client for OpenRouter
        """
        super().__init__(model_name, temperature)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
