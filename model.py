"""Weight loading and decode API for Qwen3.5 bf16 megakernel variants."""

from dataclasses import dataclass
import json
import os
import struct
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import torch

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEFAULT_CACHE_SEQ_LEN = 2048


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    max_seq_len: int
    fa_num_q_heads: int
    fa_num_kv_heads: int
    fa_head_dim: int
    dn_num_heads: int
    dn_key_dim: int
    dn_value_dim: int
    dn_conv_kernel: int
    layer_type: tuple[int, ...]
    weight_prefix: str = "model"

    @property
    def fa_q_size(self) -> int:
        return self.fa_num_q_heads * self.fa_head_dim

    @property
    def fa_qproj_size(self) -> int:
        return self.fa_q_size * 2

    @property
    def fa_kv_size(self) -> int:
        return self.fa_num_kv_heads * self.fa_head_dim

    @property
    def dn_qk_size(self) -> int:
        return self.dn_num_heads * self.dn_key_dim

    @property
    def dn_v_size(self) -> int:
        return self.dn_num_heads * self.dn_value_dim

    @property
    def dn_conv_channels(self) -> int:
        return self.dn_qk_size * 2 + self.dn_v_size


KERNEL_BASELINE_CONFIG = ModelConfig(
    model_name=DEFAULT_MODEL_NAME,
    num_layers=24,
    hidden_size=1024,
    intermediate_size=3584,
    vocab_size=248320,
    max_position_embeddings=2048,
    max_seq_len=DEFAULT_CACHE_SEQ_LEN,
    fa_num_q_heads=8,
    fa_num_kv_heads=2,
    fa_head_dim=256,
    dn_num_heads=16,
    dn_key_dim=128,
    dn_value_dim=128,
    dn_conv_kernel=4,
    layer_type=(0, 0, 0, 1) * 6,
)

# Backward-compatible exports for the currently compiled kernel baseline.
NUM_LAYERS = KERNEL_BASELINE_CONFIG.num_layers
HIDDEN_SIZE = KERNEL_BASELINE_CONFIG.hidden_size
INTERMEDIATE_SIZE = KERNEL_BASELINE_CONFIG.intermediate_size
VOCAB_SIZE = KERNEL_BASELINE_CONFIG.vocab_size
MAX_SEQ_LEN = KERNEL_BASELINE_CONFIG.max_seq_len
FA_NUM_Q_HEADS = KERNEL_BASELINE_CONFIG.fa_num_q_heads
FA_NUM_KV_HEADS = KERNEL_BASELINE_CONFIG.fa_num_kv_heads
FA_HEAD_DIM = KERNEL_BASELINE_CONFIG.fa_head_dim
FA_Q_SIZE = KERNEL_BASELINE_CONFIG.fa_q_size
FA_QPROJ_SIZE = KERNEL_BASELINE_CONFIG.fa_qproj_size
FA_KV_SIZE = KERNEL_BASELINE_CONFIG.fa_kv_size
DN_NUM_HEADS = KERNEL_BASELINE_CONFIG.dn_num_heads
DN_KEY_DIM = KERNEL_BASELINE_CONFIG.dn_key_dim
DN_VALUE_DIM = KERNEL_BASELINE_CONFIG.dn_value_dim
DN_QK_SIZE = KERNEL_BASELINE_CONFIG.dn_qk_size
DN_V_SIZE = KERNEL_BASELINE_CONFIG.dn_v_size
DN_CONV_CHANNELS = KERNEL_BASELINE_CONFIG.dn_conv_channels
DN_CONV_KERNEL = KERNEL_BASELINE_CONFIG.dn_conv_kernel
LAYER_TYPE = list(KERNEL_BASELINE_CONFIG.layer_type)

_decode = None


def _load_op():
    global _decode
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode


def _fetch_hf_json(model_name: str, filename: str):
    url = f"https://huggingface.co/{model_name}/raw/main/{filename}"
    with urlopen(url) as response:
        return json.load(response)


def resolve_model_config(model_name=DEFAULT_MODEL_NAME, max_seq_len=DEFAULT_CACHE_SEQ_LEN):
    """Resolve a Qwen3.5 config without requiring local Transformers support."""
    if model_name == DEFAULT_MODEL_NAME and max_seq_len == DEFAULT_CACHE_SEQ_LEN:
        return KERNEL_BASELINE_CONFIG

    try:
        cfg = _fetch_hf_json(model_name, "config.json")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Unable to fetch config.json for {model_name}") from exc

    text_cfg = cfg.get("text_config", cfg)
    layer_types = text_cfg.get("layer_types")
    num_layers = text_cfg.get("num_hidden_layers")
    if layer_types is None or num_layers is None:
        raise ValueError(f"{model_name} is missing Qwen3.5 layer metadata in config.json")
    if len(layer_types) != num_layers:
        raise ValueError(
            f"{model_name} layer_types length {len(layer_types)} does not match num_hidden_layers {num_layers}"
        )

    cfg_obj = ModelConfig(
        model_name=model_name,
        num_layers=num_layers,
        hidden_size=text_cfg["hidden_size"],
        intermediate_size=text_cfg["intermediate_size"],
        vocab_size=text_cfg["vocab_size"],
        max_position_embeddings=text_cfg["max_position_embeddings"],
        max_seq_len=min(max_seq_len, text_cfg["max_position_embeddings"]),
        fa_num_q_heads=text_cfg["num_attention_heads"],
        fa_num_kv_heads=text_cfg["num_key_value_heads"],
        fa_head_dim=text_cfg["head_dim"],
        dn_num_heads=text_cfg["linear_num_key_heads"],
        dn_key_dim=text_cfg["linear_key_head_dim"],
        dn_value_dim=text_cfg["linear_value_head_dim"],
        dn_conv_kernel=text_cfg["linear_conv_kernel_dim"],
        layer_type=tuple(1 if x == "full_attention" else 0 for x in layer_types),
        weight_prefix="model.language_model" if "text_config" in cfg else "model",
    )
    validate_model_config(cfg_obj)
    return cfg_obj


def validate_model_config(model_config: ModelConfig):
    """Check whether a model stays within the kernel's architectural assumptions."""
    expected = {
        "num_layers": 24,
        "fa_num_q_heads": 8,
        "fa_num_kv_heads": 2,
        "fa_head_dim": 256,
        "dn_num_heads": 16,
        "dn_key_dim": 128,
        "dn_value_dim": 128,
        "dn_conv_kernel": 4,
        "vocab_size": 248320,
    }
    for field, expected_value in expected.items():
        actual = getattr(model_config, field)
        if actual != expected_value:
            raise ValueError(
                f"{model_config.model_name} is not compatible with the current kernel: "
                f"{field}={actual}, expected {expected_value}"
            )
    if model_config.layer_type != KERNEL_BASELINE_CONFIG.layer_type:
        raise ValueError(
            f"{model_config.model_name} layer schedule differs from the compiled hybrid 3:1 pattern"
        )


def _resolve_state_prefix(state, model_config: ModelConfig) -> str:
    prefixes = [model_config.weight_prefix, "model", "model.language_model"]
    for prefix in dict.fromkeys(prefixes):
        if f"{prefix}.embed_tokens.weight" in state:
            return prefix
    raise KeyError(
        f"Could not find embed tokens for {model_config.model_name}; "
        f"tried prefixes: {', '.join(dict.fromkeys(prefixes))}"
    )


def load_weights(model_name=DEFAULT_MODEL_NAME, verbose=True, model_config=None):
    """Load Qwen3.5 weights as bf16 (no quantization)."""
    if not verbose:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = model_config or resolve_model_config(model_name)
    if verbose:
        print(f"Loading {model_name} (bf16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda"
        )
    except ValueError as exc:
        raise RuntimeError(
            f"Transformers could not load {model_name}. "
            "Upgrade to a version with qwen3_5 support before trying this model."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()
    state_prefix = _resolve_state_prefix(state, model_config)

    layer_data = []
    for i in range(model_config.num_layers):
        p = f"{state_prefix}.layers.{i}."
        lt = model_config.layer_type[i]

        if lt == 1:
            # Full Attention: 11 pointers (all bf16)
            layer_data.append({
                "type": 1,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "self_attn.q_proj.weight"].contiguous(),
                    state[p + "self_attn.k_proj.weight"].contiguous(),
                    state[p + "self_attn.v_proj.weight"].contiguous(),
                    state[p + "self_attn.q_norm.weight"].contiguous(),
                    state[p + "self_attn.k_norm.weight"].contiguous(),
                    state[p + "self_attn.o_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })
        else:
            # DeltaNet: 14 pointers (all bf16)
            layer_data.append({
                "type": 0,
                "ptrs": [
                    state[p + "input_layernorm.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_qkv.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_z.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_b.weight"].contiguous(),
                    state[p + "linear_attn.in_proj_a.weight"].contiguous(),
                    state[p + "linear_attn.conv1d.weight"].contiguous(),
                    state[p + "linear_attn.A_log"].contiguous(),
                    state[p + "linear_attn.dt_bias"].contiguous(),
                    state[p + "linear_attn.norm.weight"].contiguous(),
                    state[p + "linear_attn.out_proj.weight"].contiguous(),
                    state[p + "post_attention_layernorm.weight"].contiguous(),
                    state[p + "mlp.gate_proj.weight"].contiguous(),
                    state[p + "mlp.up_proj.weight"].contiguous(),
                    state[p + "mlp.down_proj.weight"].contiguous(),
                ]
            })

    embed_key = f"{state_prefix}.embed_tokens.weight"
    final_norm_key = f"{state_prefix}.norm.weight"
    embed_weight = state[embed_key].contiguous()
    final_norm_weight = state[final_norm_key].contiguous()
    lm_head = state.get("lm_head.weight", embed_weight).contiguous()

    weights = {
        "config": ModelConfig(**{**model_config.__dict__, "weight_prefix": state_prefix}),
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"BF16 weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    return weights, tokenizer


def _pack_layer_weights(layer_data, num_layers):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(num_layers * struct_size)
    for i in range(num_layers):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def allocate_prefill_buffers(model_config: ModelConfig, seq_len: int):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    mx = max(model_config.dn_conv_channels, model_config.fa_qproj_size, model_config.intermediate_size)
    return dict(
        hidden=torch.empty(seq_len * model_config.hidden_size, **bf16),
        residual=torch.empty(seq_len * model_config.hidden_size, **bf16),
        normalized=torch.empty(seq_len * model_config.hidden_size, **bf16),
        proj_buf=torch.empty(seq_len * mx, **bf16),
        proj_buf2=torch.empty(seq_len * mx, **bf16),
        attn_buf=torch.empty(seq_len * max(model_config.fa_q_size, model_config.fa_kv_size), **bf16),
        mlp_buf=torch.empty(seq_len * model_config.intermediate_size, **bf16),
        dn_out_buf=torch.empty(seq_len * model_config.dn_v_size, **bf16),
        beta_buf=torch.empty(seq_len * model_config.dn_num_heads, **f32),
        alpha_buf=torch.empty(seq_len * model_config.dn_num_heads, **f32),
        final_normed=torch.empty(model_config.hidden_size, **bf16),
        hidden_bf16_out=torch.empty(model_config.hidden_size, **bf16),
        lm_bmv=torch.empty(1024, **f32),
        lm_bmi=torch.empty(1024, **i32),
    )


class Decoder:
    """Stateful decoder for kernel-compatible Qwen3.5 bf16 models."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name=DEFAULT_MODEL_NAME, verbose=True, max_seq_len=DEFAULT_CACHE_SEQ_LEN):
        _load_op()

        model_config = resolve_model_config(model_name, max_seq_len=max_seq_len)
        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose, model_config=model_config)
        else:
            model_config = weights.get("config", model_config)
        self.tokenizer = tokenizer
        self.config = model_config
        self._position = 0
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_data"], self.config.num_layers)

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in self.config.layer_type if t == 1)
        self._fa_k_cache = torch.zeros(
            n_fa, self.config.fa_num_kv_heads, self.config.max_seq_len, self.config.fa_head_dim, **bf16
        )
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in self.config.layer_type if t == 0)
        self._dn_states = torch.zeros(
            n_dn, self.config.dn_num_heads, self.config.dn_key_dim, self.config.dn_value_dim, **f32
        )
        self._conv_bufs = torch.zeros(n_dn, self.config.dn_conv_channels, self.config.dn_conv_kernel, **f32)

        self._hidden = torch.empty(self.config.hidden_size, **bf16)
        max_scratch = max(
            self.config.fa_qproj_size,
            self.config.dn_conv_channels,
            self.config.hidden_size * 8 + self.config.intermediate_size,
        )
        self._activations = torch.empty(max_scratch, **f32)
        self._residual = torch.empty(self.config.hidden_size, **bf16)
        self._qkv_scratch = torch.empty(max(self.config.fa_qproj_size, self.config.dn_conv_channels), **f32)
        self._kv_scratch = torch.empty(self.config.fa_kv_size * 2, **f32)
        self._attn_out = torch.empty(max(self.config.fa_q_size, self.config.dn_v_size), **f32)
        self._mlp_inter = torch.empty(self.config.intermediate_size, **f32)
        self._z_scratch = torch.empty(self.config.dn_v_size, **f32)
        self._beta_scratch = torch.empty(self.config.dn_num_heads, **f32)
        self._alpha_scratch = torch.empty(self.config.dn_num_heads, **f32)
        self._normalized = torch.empty(self.config.hidden_size, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        self._block_max_vals = torch.empty(1024, **f32)
        self._block_max_idxs = torch.empty(1024, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._out_token = torch.empty(1, **i32)

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._position, self.config.max_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out = []
        next_id = ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            next_id = self.step(next_id)
            if next_id == eos:
                break
            out.append(next_id)
        return self.tokenizer.decode(out, skip_special_tokens=True)
