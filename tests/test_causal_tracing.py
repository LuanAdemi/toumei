import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from toumei.transformers.rome.tracing import CausalTracer
from toumei.cnns.objectives.utils import set_seed

sys.path.append("../")


def test_causal_tracing():
    """
    Locating factual knowledge in GPT like models using causal tracing
    """

    # seed for reproducibility
    set_seed(42)

    # load gpt2 from huggingface
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

    model.to(torch.device("cuda"))

    # specify a prompt and it's subject for causal tracing
    prompt = "The light bulb was invented by"
    subject = "The light bulb"

    # perform causal tracing
    tracer = CausalTracer(model, tokenizer)
    results = tracer.trace(prompt, subject)

    assert (results["scores"].numpy() == (np.load("outputs/tracing_results.npy"))).all()
