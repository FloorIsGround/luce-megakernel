"""Final benchmark: pp520 tg128 - Our megakernel vs PyTorch naive.
Both properly warmed. Saves completions for verification."""

import os
import time

import torch
import qwen35_megakernel_bf16_C
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import DEFAULT_MODEL_NAME, Decoder, allocate_prefill_buffers

MODEL_NAME = os.environ.get("QWEN_MODEL", DEFAULT_MODEL_NAME)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================================================
# Build a 520-token prompt
# ============================================================
long_text = (
    "Explain in great detail the history of artificial intelligence, machine "
    "learning, deep learning, and neural networks. " * 40
)
prompt_ids = tok.encode(long_text, add_special_tokens=False)[:520]
print(f"Prompt: {len(prompt_ids)} tokens")

# ============================================================
# 1. Our megakernel (prefill cuBLAS + decode megakernel)
# ============================================================
print("\n=== Our BF16 Megakernel ===")
dec = Decoder(model_name=MODEL_NAME, verbose=False)
_pf = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

S = 520
b = allocate_prefill_buffers(dec.config, S)
ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")


def our_prefill():
    dec.reset()
    _pf(
        dec._out_token,
        ids_t,
        dec._embed_weight,
        dec._layer_weights_packed,
        dec._final_norm_weight,
        dec._lm_head_weight,
        dec._fa_k_cache,
        dec._fa_v_cache,
        dec._dn_states,
        dec._conv_bufs,
        b["hidden"],
        b["residual"],
        b["normalized"],
        b["proj_buf"],
        b["proj_buf2"],
        b["attn_buf"],
        b["mlp_buf"],
        b["dn_out_buf"],
        b["beta_buf"],
        b["alpha_buf"],
        b["final_normed"],
        b["hidden_bf16_out"],
        b["lm_bmv"],
        b["lm_bmi"],
    )
    dec._hidden.copy_(b["hidden_bf16_out"])
    dec._position = len(prompt_ids)
    return dec._out_token.item()


for _ in range(10):
    our_prefill()
    torch.cuda.synchronize()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    our_prefill()
    torch.cuda.synchronize()
our_pp_ms = (time.perf_counter() - t0) / 20 * 1000
our_pp_tps = len(prompt_ids) / our_pp_ms * 1000

first = our_prefill()
torch.cuda.synchronize()
out_ids = [first]
nid = first
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(128):
    nid = dec.step(nid)
    if nid == tok.eos_token_id:
        break
    out_ids.append(nid)
torch.cuda.synchronize()
our_tg_ms = (time.perf_counter() - t0) * 1000
our_tg_tps = len(out_ids) / our_tg_ms * 1000
our_text = tok.decode(out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {our_pp_tps:.0f} tok/s ({our_pp_ms:.1f}ms)")
print(f"tg{len(out_ids)}: {our_tg_tps:.0f} tok/s ({our_tg_ms:.1f}ms)")
print(f"Completion: {our_text[:120]}")

del dec
torch.cuda.empty_cache()

# ============================================================
# 2. PyTorch naive (HuggingFace)
# ============================================================
print("\n=== PyTorch HuggingFace ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
model.eval()
input_ids = torch.tensor([prompt_ids], device="cuda")

with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids)
        torch.cuda.synchronize()

with torch.no_grad():
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = model(input_ids)
        torch.cuda.synchronize()
    pt_pp_ms = (time.perf_counter() - t0) / 10 * 1000
    pt_pp_tps = len(prompt_ids) / pt_pp_ms * 1000

with torch.no_grad():
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1:].argmax(-1)
    pt_out_ids = [next_id.item()]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(128):
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1:].argmax(-1)
        if next_id.item() == tok.eos_token_id:
            break
        pt_out_ids.append(next_id.item())
    torch.cuda.synchronize()
    pt_tg_ms = (time.perf_counter() - t0) * 1000
    pt_tg_tps = len(pt_out_ids) / pt_tg_ms * 1000
    pt_text = tok.decode(pt_out_ids, skip_special_tokens=True)

print(f"pp{len(prompt_ids)}: {pt_pp_tps:.0f} tok/s ({pt_pp_ms:.1f}ms)")
print(f"tg{len(pt_out_ids)}: {pt_tg_tps:.0f} tok/s ({pt_tg_ms:.1f}ms)")
print(f"Completion: {pt_text[:120]}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print(f"FINAL RESULTS - {MODEL_NAME} BF16, RTX 4070-ti")
print(f"{'=' * 60}")
print(f"{'Method':<25} {'pp' + str(len(prompt_ids)):>8} {'tg128':>10}")
print(f"{'-' * 45}")
print(f"{'Our megakernel':<25} {our_pp_tps:>7.0f} t/s {our_tg_tps:>8.0f} t/s")
print(f"{'PyTorch HF':<25} {pt_pp_tps:>7.0f} t/s {pt_tg_tps:>8.0f} t/s")
print(f"{'llama.cpp BF16':<25} {'(run separately)':>19}")
print("")
print(f"Megakernel vs PyTorch:  pp {our_pp_tps / pt_pp_tps:.1f}x  tg {our_tg_tps / pt_tg_tps:.1f}x")
print("")
print("=== Completions ===")
print(f"Ours:    {our_text[:100]}")
print(f"PyTorch: {pt_text[:100]}")
