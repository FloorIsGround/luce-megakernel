"""Benchmark bf16 megakernel decode."""
import os
import time
import torch
from model import DEFAULT_MODEL_NAME, Decoder
from transformers import AutoTokenizer

MODEL_NAME = os.environ.get("QWEN_MODEL", DEFAULT_MODEL_NAME)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
dec = Decoder(model_name=MODEL_NAME, verbose=True)

prompt = "The capital of France is"
ids = tok.encode(prompt, add_special_tokens=False)

# Warmup
dec.reset()
for t in ids[:-1]:
    dec.step(t)
first = dec.step(ids[-1])

# Benchmark decode
dec.reset()
for t in ids[:-1]:
    dec.step(t)

torch.cuda.synchronize()
t0 = time.perf_counter()
out = []
next_id = ids[-1]
for _ in range(200):
    next_id = dec.step(next_id)
    if next_id == tok.eos_token_id:
        break
    out.append(next_id)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

tps = len(out) / elapsed
text = tok.decode(out, skip_special_tokens=True)[:80]
print(f"Decode: {tps:.1f} tok/s ({len(out)} tokens in {elapsed*1000:.1f}ms)")
print(f"Output: {text}")
