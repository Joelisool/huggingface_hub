import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LOCAL_MODEL_PATH, MODEL_CONFIG
import os

class LocalModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
    async def load_model(self, model_type):
        if model_type not in self.models:
            model_path = os.path.join(LOCAL_MODEL_PATH, MODEL_CONFIG[model_type])
            if os.path.exists(model_path):
                self.models[model_type] = AutoModelForCausalLM.from_pretrained(model_path)
                self.tokenizers[model_type] = AutoTokenizer.from_pretrained(model_path)
            else:
                raise ValueError(f"Local model not found: {model_path}")
        return self.models[model_type], self.tokenizers[model_type]

    async def generate_text(self, prompt, model_type="text_generation"):
        model, tokenizer = await self.load_model(model_type)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0])

    # Add other model-specific methods here
