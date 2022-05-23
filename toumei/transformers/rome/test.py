import torch
from toumei.probe import get_modules
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from toumei.transformers.rome.tracing import CausalTracer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float16)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

model.to(torch.device("cuda"))

prompt = "Karlsruhe Institute of Technology is located in the country of"

tracer = CausalTracer(model, tokenizer)

tracer.trace(prompt, "Karlsruhe Institute of Technology")

