"""Benchmark pp512 tg128 — standard llama.cpp benchmark format.
Also tests end-to-end correctness (prefill → decode handoff).

Scope: batch-size-1 single-stream decode, targeting local inference.
All measurements use torch.cuda.synchronize() barriers + perf_counter.
One warm-up run precedes each timed section."""
import os
import time, torch
from model import DEFAULT_MODEL_NAME, Decoder, allocate_prefill_buffers
import qwen35_megakernel_bf16_C
from transformers import AutoTokenizer

MODEL_NAME = os.environ.get("QWEN_MODEL", DEFAULT_MODEL_NAME)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
dec = Decoder(model_name=MODEL_NAME, verbose=True)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

# Allocate prefill buffers for max 512 tokens
S_MAX = 512
bufs = allocate_prefill_buffers(dec.config, S_MAX)

def prefill(ids):
    ids_t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    _pf(dec._out_token, ids_t,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._fa_k_cache, dec._fa_v_cache, dec._dn_states, dec._conv_bufs,
        bufs['hidden'], bufs['residual'], bufs['normalized'],
        bufs['proj_buf'], bufs['proj_buf2'],
        bufs['attn_buf'], bufs['mlp_buf'],
        bufs['dn_out_buf'], bufs['beta_buf'], bufs['alpha_buf'],
        bufs['final_normed'], bufs['hidden_bf16_out'],
        bufs['lm_bmv'], bufs['lm_bmi'])
    # Handoff: copy hidden state for decode kernel
    dec._hidden.copy_(bufs['hidden_bf16_out'])
    dec._position = len(ids)
    return dec._out_token.item()

# ============================================================
# 1. End-to-end correctness test
# ============================================================
print("\n=== Correctness test ===", flush=True)
prompt = "The capital of France is"
ids = tok.encode(prompt, add_special_tokens=False)
dec.reset()
first = prefill(ids)
print(f"Prefill → first token: {first} = '{tok.decode([first])}'", flush=True)

# Continue with decode megakernel
out = [first]
nid = first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    out.append(nid)
text = tok.decode(out, skip_special_tokens=True)
print(f"Output: {text[:80]}", flush=True)

# Reference: pure decode (step-by-step)
dec.reset()
for t in ids[:-1]: dec.step(t)
ref_first = dec.step(ids[-1])
ref_out = [ref_first]
nid = ref_first
for _ in range(30):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    ref_out.append(nid)
ref_text = tok.decode(ref_out, skip_special_tokens=True)
print(f"Ref:    {ref_text[:80]}", flush=True)

if out == ref_out:
    print("PASS: megakernel output matches reference decode path", flush=True)
else:
    print("FAIL: output mismatch between megakernel and reference", flush=True)
    print(f"  Megakernel tokens: {out[:10]}...", flush=True)
    print(f"  Reference tokens:  {ref_out[:10]}...", flush=True)

# ============================================================
# 2. pp512 benchmark (prompt processing)
# ============================================================
print("\n=== pp512 benchmark ===", flush=True)
# Generate a 512-token prompt
long_prompt = "Explain in great detail the history of artificial intelligence, " * 30
long_ids = tok.encode(long_prompt, add_special_tokens=False)[:512]
print(f"Prompt tokens: {len(long_ids)}", flush=True)

# Warmup
dec.reset()
prefill(long_ids)

# Benchmark
dec.reset()
torch.cuda.synchronize()
t0 = time.perf_counter()
prefill(long_ids)
torch.cuda.synchronize()
pp_time = time.perf_counter() - t0
pp_tps = len(long_ids) / pp_time
print(f"pp{len(long_ids)}: {pp_tps:.1f} tok/s ({pp_time*1000:.1f}ms)", flush=True)

# ============================================================
# 3. tg128 benchmark (token generation)
# ============================================================
print("\n=== tg128 benchmark ===", flush=True)
# Prefill a short prompt, then generate 128 tokens
short_ids = tok.encode("Hello", add_special_tokens=False)
dec.reset()
first = prefill(short_ids)

torch.cuda.synchronize()
t0 = time.perf_counter()
gen_out = []
nid = first
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id: break
    gen_out.append(nid)
torch.cuda.synchronize()
tg_time = time.perf_counter() - t0
tg_tps = len(gen_out) / tg_time
print(f"tg{len(gen_out)}: {tg_tps:.1f} tok/s ({tg_time*1000:.1f}ms)", flush=True)

# ============================================================
# Summary
# ============================================================
print(f"\n=== Summary (RTX 4070-ti, {MODEL_NAME} BF16) ===", flush=True)
print(f"pp{len(long_ids):>3d}: {pp_tps:>7.1f} tok/s", flush=True)
print(f"tg{len(gen_out):>3d}: {tg_tps:>7.1f} tok/s", flush=True)
