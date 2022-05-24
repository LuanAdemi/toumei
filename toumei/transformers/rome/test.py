import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from toumei.transformers.rome.tracing import CausalTracer

model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

model.to(torch.device("cuda"))

prompt = "Karlsruhe Institute of Technology is located in the country of"

subject = "Karlsruhe Institute of Technology"

tracer = CausalTracer(model, tokenizer)

tracer.trace(prompt, subject, verbose=True)

