import torch
from transformers import GPTJForCausalLM

# The purpose of this file is to pre-download GPTJ model so that parallel experiments can be run
# without

# Load model
model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    revision="float16",
    torch_dtype=torch.float16
)
print("Fetched the Model.")