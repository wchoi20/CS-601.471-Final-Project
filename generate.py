import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model = AutoModelForCausalLM.from_pretrained("./rlhf-model-A")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "To sneak snacks into a movie theater, I "

input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(input_ids)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)