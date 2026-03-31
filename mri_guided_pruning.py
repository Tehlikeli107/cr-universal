"""
MRI-GUIDED PRUNING: Prune layers that Model MRI flags as spiked.

Hypothesis: correlation spike = redundant neurons = safe to prune.
Test: prune GPT-2 L11 (spike) vs L6 (no spike). Compare PPL impact.

If spike layer more prunable: Model MRI = automatic pruning guide.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

DEVICE = torch.device('cuda')

def ppl(model, tokenizer):
    text = ("The quick brown fox jumps over the lazy dog. "
            "Machine learning is transforming how we understand data. "
            "Neural networks have revolutionized computer vision and NLP. "
            "The development of transformer architectures enabled large language models. "
            "Attention mechanisms allow models to focus on relevant parts of input.")
    tokens = tokenizer(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**tokens, labels=tokens['input_ids'])
    return out.loss.exp().item()

def prune_layer(model, layer_idx, frac=0.5):
    """Zero out frac of c_proj neurons with smallest L2 norm."""
    W = model.transformer.h[layer_idx].attn.c_proj.weight.data
    norms = W.norm(dim=1)
    n_prune = int(W.shape[0] * frac)
    idx = norms.argsort()[:n_prune]
    W[idx] = 0
    b = model.transformer.h[layer_idx].attn.c_proj.bias
    if b is not None: b.data[idx] = 0

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

print("MRI-GUIDED PRUNING: Spike layer vs Normal layer")
print("=" * 55)

# Baseline
model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
base = ppl(model, tokenizer)
print(f"Baseline PPL: {base:.2f}")
del model; torch.cuda.empty_cache()

# Prune L11 (SPIKE — tri=0.300, density=0.595)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
prune_layer(model, 11, frac=0.5)
l11 = ppl(model, tokenizer)
print(f"L11 50% pruned (SPIKE): PPL={l11:.2f} (delta={l11-base:+.2f})")
del model; torch.cuda.empty_cache()

# Prune L6 (NO spike — density=0.001)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
prune_layer(model, 6, frac=0.5)
l6 = ppl(model, tokenizer)
print(f"L6  50% pruned (normal): PPL={l6:.2f} (delta={l6-base:+.2f})")
del model; torch.cuda.empty_cache()

# Prune L0 (lowest spike)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
prune_layer(model, 0, frac=0.5)
l0 = ppl(model, tokenizer)
print(f"L0  50% pruned (normal): PPL={l0:.2f} (delta={l0-base:+.2f})")
del model; torch.cuda.empty_cache()

# Prune L8 (mild anomaly)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
prune_layer(model, 8, frac=0.5)
l8 = ppl(model, tokenizer)
print(f"L8  50% pruned (mild):   PPL={l8:.2f} (delta={l8-base:+.2f})")
del model; torch.cuda.empty_cache()

print(f"\n{'='*55}")
print(f"RESULT:")
damages = [(11, l11-base, "SPIKE"), (6, l6-base, "normal"), (0, l0-base, "normal"), (8, l8-base, "mild")]
damages.sort(key=lambda x: x[1])
for layer, dmg, status in damages:
    print(f"  L{layer:2d} ({status:6s}): damage={dmg:+.2f}")

if l11 - base < min(l6-base, l0-base, l8-base):
    print(f"\n*** SPIKE LAYER MOST PRUNABLE! ***")
    print(f"  Model MRI correlation spike = pruning safety indicator")
    print(f"  CR fingerprint guides automatic model compression")
else:
    print(f"\n  Spike layer NOT most prunable. MRI finding is structural only.")
