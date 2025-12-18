from llama_cpp import Llama

class LocalMistral:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            temperature=0.3,
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        out = self.llm(prompt, max_tokens=512, stop=["</s>"])
        return out["choices"][0]["text"].strip()
