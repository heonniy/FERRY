"""Model geometry, loading, and MoE-layer hooks.

Dense weights (attention, norms, router, shared experts) are resident on the GPU.
Routed-expert weights are packed into one rank-private pinned host pool with one
row per (MoE layer, expert): [gate | up | down], each row contiguous in BF16.
"""
import json
import re
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

EXPERT_RE = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
PARTS = ('gate_proj', 'up_proj', 'down_proj')


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    moe_layers: tuple      # decoder-layer indices that hold routed experts
    experts: int           # routed experts per MoE layer
    topk: int
    hidden: int
    intermediate: int      # routed-expert intermediate size

    @property
    def layers(self):
        return len(self.moe_layers)

    @property
    def expert_elements(self):
        return 3 * self.hidden * self.intermediate

    @property
    def expert_bytes(self):
        return 2 * self.expert_elements

    def cache_slots(self, cache_ratio, world):
        """Global expert-cache budget split evenly across ranks (lower ranks get the remainder)."""
        total = int(self.layers * self.experts * cache_ratio)
        return [total // world + (r < total % world) for r in range(world)]


def read_spec(model_path):
    c = json.loads((Path(model_path) / 'config.json').read_text())
    kind = c['model_type']
    n = c['num_hidden_layers']
    if kind == 'qwen3_moe':
        step = c.get('decoder_sparse_step', 1)
        dense = set(c.get('mlp_only_layers') or [])
        layers = tuple(i for i in range(n) if i not in dense and (i + 1) % step == 0)
        experts = c['num_experts']
    elif kind == 'deepseek_v2':
        first = c['first_k_dense_replace']
        freq = c.get('moe_layer_freq', 1)
        layers = tuple(i for i in range(n) if i >= first and i % freq == 0)
        experts = c['n_routed_experts']
        if c.get('topk_method', 'greedy') != 'greedy':
            raise NotImplementedError('only greedy top-k routing is supported for DeepSeek-V2')
    else:
        raise NotImplementedError(f'unsupported model_type {kind}')
    return ModelSpec(kind, layers, experts, c['num_experts_per_tok'], c['hidden_size'],
                     c['moe_intermediate_size'])


def load(model_path, spec, device='cuda'):
    """Return (model, pinned_pool). pinned_pool[i] is the flat BF16 row of expert key i."""
    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    index = model_path / 'model.safetensors.index.json'
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
    else:
        weight_map = {name: 'model.safetensors' for name in
                      safe_open(str(model_path / 'model.safetensors'), framework='pt').keys()}
    layer_of = {layer: i for i, layer in enumerate(spec.moe_layers)}
    h, inter = spec.hidden, spec.intermediate
    offsets = {'gate_proj': 0, 'up_proj': h * inter, 'down_proj': 2 * h * inter}
    pool = torch.empty((spec.layers * spec.experts, spec.expert_elements), dtype=torch.bfloat16,
                       pin_memory=True)
    loaded = torch.zeros((spec.layers * spec.experts, 3), dtype=torch.bool)
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(model_path / shard), framework='pt', device='cpu') as f:
            for name in sorted(n for n, s in weight_map.items() if s == shard):
                match = EXPERT_RE.fullmatch(name)
                if match is None:
                    set_module_tensor_to_device(model, name, device, value=f.get_tensor(name),
                                                dtype=torch.bfloat16)
                    continue
                key = layer_of[int(match[1])] * spec.experts + int(match[2])
                tensor = f.get_tensor(name)
                assert tensor.numel() == h * inter, name
                start = offsets[match[3]]
                pool[key, start:start + h * inter].copy_(tensor.reshape(-1))
                loaded[key, PARTS.index(match[3])] = True
    assert bool(loaded.all()), 'routed-expert weights are incomplete'
    for name, param in model.named_parameters():
        if not EXPERT_RE.fullmatch(name):
            assert param.device.type == 'cuda', f'{name} was not loaded'
    # Buffers are created on the host by init_empty_weights; rebuild RoPE on the GPU.
    for module in model.modules():
        if hasattr(module, 'rope_init_fn') and hasattr(module, 'inv_freq'):
            inv, module.attention_scaling = module.rope_init_fn(module.config, device=torch.device(device))
            module.register_buffer('inv_freq', inv, persistent=False)
            module.original_inv_freq = inv
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type != 'cuda':
                module._buffers[name] = buf.to(device)
    model.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return model, pool


def install_hooks(model, spec, runtime, history_window):
    """Route every MoE layer through runtime.execute(layer, hidden, selected, weights, probs_tail)."""
    for layer, index in enumerate(spec.moe_layers):
        mlp = model.model.layers[index].mlp
        if spec.model_type == 'qwen3_moe':
            mlp.forward = types.MethodType(_qwen3_forward(runtime, layer, spec, history_window), mlp)
        else:
            mlp.moe = types.MethodType(_deepseek_moe(runtime, layer, history_window), mlp)
        # Routed experts live in the pinned pool; drop the empty modules.
        mlp.experts = None


def _qwen3_forward(rt, layer, spec, window):
    def forward(module, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        logits = module.gate(flat)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        weights, selected = torch.topk(probs, spec.topk, dim=-1)
        if module.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        out = rt.execute(layer, flat, selected, weights.to(flat.dtype), probs[-window:])
        return out.view(shape), logits
    return forward


def _deepseek_moe(rt, layer, window):
    def moe(module, hidden_states, topk_ids, topk_weight):
        probs = None
        if rt.gather.score == 'probability':
            tail = hidden_states[-window:]
            probs = torch.softmax(F.linear(tail.float(), module.gate.weight.float()), dim=-1)
        return rt.execute(layer, hidden_states, topk_ids, topk_weight.to(torch.bfloat16), probs)
    return moe
