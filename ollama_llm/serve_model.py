import torch
from transformers import AutoModel, AutoTokenizer

class OllamaLLMServer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def generate_response(self, input_text: str) -> str:
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def serve_model(model_name: str, input_text: str) -> str:
    server = OllamaLLMServer(model_name)
    return server.generate_response(input_text)
