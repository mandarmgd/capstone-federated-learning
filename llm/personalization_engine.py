# llm/personalization_engine.py

from llama_cpp import Llama
from llm.prompts import build_personalization_prompt


class PersonalizationEngine:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,        # hard cap to prevent runaway context
            n_threads=8,
            verbose=False
        )

    def generate(self, context: dict) -> str:
        """
        Generate personalization strictly from a pre-built prompt.
        """

        prompt = build_personalization_prompt(context)

        output = self.llm(
            prompt=prompt,
            max_tokens=512,
            temperature=0.3,   # lowered to reduce hallucination
            top_p=0.9,
            stop=["</s>"]
        )

        return output["choices"][0]["text"].strip()
