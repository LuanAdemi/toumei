import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from toumei.transformers.rome.tracing import CausalTracer

"""
Locating factual knowledge in GPT like models using causal tracing
"""

# load gpt2 from huggingface
model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

model.to(torch.device("cuda"))

# specify a prompt and it's subject for causal tracing
prompt = "The light bulb was invented by"
subject = "The light bulb"

# perform causal tracing
tracer = CausalTracer(model, tokenizer)
tracer.trace(prompt, subject, verbose=True)
