"""
BLACKBOX: The most black-box AI architecture possible.

No attention. No convolution. No layers. No heads.
Just: state = activation(W @ state + input)
Repeated. W itself slowly evolves.

Architecture:
- State: [D] tensor (fixed size, carries ALL information)
- Transform: W [D, D] matrix (learned)
- Each step: state = gelu(W @ state + proj(input_token))
- N steps = N "layers" but same W (weight sharing)
- Meta-W: W gets small updates based on state (self-modification)
- Output: linear projection of final state

Total params: D*D (W) + D*vocab (input proj) + D*vocab (output proj)
For D=256: 65K + 256*vocab. TINY compared to transformer.

Test: language modeling on same task as NEXUS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

DEVICE = torch.device('cuda')


class BlackBox(nn.Module):
    """
    The most black-box architecture.
    No human-designed components except matrix multiply + nonlinearity.
    """
    def __init__(self, vocab_size, d_state=256, n_steps=12):
        super().__init__()
        self.d = d_state
        self.n_steps = n_steps

        # Input: token -> state contribution
        self.token_embed = nn.Embedding(vocab_size, d_state)

        # THE core: single matrix that transforms state
        self.W = nn.Parameter(torch.randn(d_state, d_state) * 0.02)

        # Meta-update: state -> small W modification
        self.meta = nn.Linear(d_state, d_state, bias=False)
        self.meta_scale = nn.Parameter(torch.tensor(0.01))

        # Output: state -> vocab logits
        self.output = nn.Linear(d_state, vocab_size)

        # Positional signal (minimal)
        self.pos_scale = nn.Parameter(torch.randn(n_steps) * 0.1)

    def forward(self, input_ids):
        """
        input_ids: [batch, seq_len]
        Returns: [batch, seq_len, vocab_size] logits
        """
        B, N = input_ids.shape
        embeds = self.token_embed(input_ids)  # [B, N, D]

        all_outputs = []

        for pos in range(N):
            # Initialize state from token embedding
            state = embeds[:, pos]  # [B, D]

            # Current W (may be modified by meta-learning)
            W_current = self.W

            # Run n_steps of state transformation
            for step in range(self.n_steps):
                # Core operation: state = gelu(W @ state)
                state = F.gelu(state @ W_current.T + self.pos_scale[step] * state)

                # Accumulate context from previous tokens
                if pos > 0:
                    # Simple: add weighted sum of previous outputs
                    ctx_weight = 1.0 / (pos + 1)
                    state = state + ctx_weight * prev_state

            # Meta-update: modify W slightly based on what state learned
            if self.training:
                meta_update = self.meta(state.detach().mean(dim=0))  # [D]
                W_current = self.W + self.meta_scale * meta_update.unsqueeze(0)

            prev_state = state.detach()
            all_outputs.append(state)

        # Stack all position outputs
        output = torch.stack(all_outputs, dim=1)  # [B, N, D]
        return self.output(output)  # [B, N, vocab]


class BlackBoxV2(nn.Module):
    """
    V2: Full sequence processing (not position-by-position).
    State accumulates across sequence. More like RNN but with meta-W.
    """
    def __init__(self, vocab_size, d_state=128, n_inner=4):
        super().__init__()
        self.d = d_state
        self.n_inner = n_inner

        self.embed = nn.Embedding(vocab_size, d_state)
        self.W = nn.Parameter(torch.randn(d_state, d_state) * 0.02)
        self.gate = nn.Linear(d_state * 2, d_state)  # gate for state update
        self.output = nn.Linear(d_state, vocab_size)

    def forward(self, input_ids):
        B, N = input_ids.shape
        embeds = self.embed(input_ids)  # [B, N, D]

        state = torch.zeros(B, self.d, device=input_ids.device)
        outputs = []

        for pos in range(N):
            token = embeds[:, pos]  # [B, D]

            # Inner loop: refine state with current token
            for _ in range(self.n_inner):
                new_state = F.gelu(state @ self.W.T + token)
                # Gated update: how much to change state
                gate = torch.sigmoid(self.gate(torch.cat([state, new_state], dim=-1)))
                state = gate * new_state + (1 - gate) * state

            outputs.append(state)

        return self.output(torch.stack(outputs, dim=1))


class StandardTransformer(nn.Module):
    """Standard transformer for comparison."""
    def __init__(self, vocab_size, d=128, n_heads=4, n_layers=4, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_seq, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, d*4, batch_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, input_ids):
        B, N = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(N, device=input_ids.device))
        mask = torch.triu(torch.ones(N, N, device=input_ids.device), 1).bool()
        return self.head(self.enc(x, mask=mask))


# ===== Task: next token prediction with RULES =====
def gen_data(vocab=30, seq=24, batch=64):
    """Multiple rules: copy-3-back, increment, repeat pattern."""
    t = torch.randint(0, vocab, (batch, seq), device=DEVICE)
    for b in range(batch):
        rule = b % 3
        if rule == 0:
            # Copy 3 back
            for i in range(3, seq): t[b, i] = t[b, i-3]
        elif rule == 1:
            # Increment mod vocab
            for i in range(1, seq): t[b, i] = (t[b, i-1] + 1) % vocab
        else:
            # Repeat first 4 tokens
            for i in range(4, seq): t[b, i] = t[b, i % 4]
    return t[:, :-1], t[:, 1:]


# ===== Benchmark =====
V = 30; SEQ = 16; EPOCHS = 80; BATCH = 32

print("BLACKBOX vs TRANSFORMER: Next Token Prediction")
print("=" * 50)
print(f"3 rules: copy-3-back, increment, repeat-4")
print(f"vocab={V}, seq={SEQ}, epochs={EPOCHS}\n")

for name, make_model in [
    ("Transformer", lambda: StandardTransformer(V, d=128, n_heads=4, n_layers=4, max_seq=SEQ)),
    ("BB-128-inner4", lambda: BlackBoxV2(V, d_state=128, n_inner=4)),
    ("BB-256-inner4", lambda: BlackBoxV2(V, d_state=256, n_inner=4)),
]:
    torch.manual_seed(42)
    model = make_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    for ep in range(EPOCHS):
        inp, tgt = gen_data(V, SEQ, BATCH)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

    elapsed = time.time() - t0

    # Test
    with torch.no_grad():
        inp, tgt = gen_data(V, SEQ, 1000)
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        ppl = loss.exp().item()
        acc = (logits.argmax(-1) == tgt).float().mean().item()
        # Per-rule accuracy
        rule_accs = []
        for r in range(3):
            r_idx = torch.arange(1000, device=DEVICE)
            r_mask = (r_idx % 3 == r)
            r_acc = (logits[r_mask].argmax(-1) == tgt[r_mask]).float().mean().item()
            rule_accs.append(r_acc)

    print(f"  {name:15s}: PPL={ppl:.2f} acc={acc*100:.1f}% "
          f"[copy={rule_accs[0]*100:.0f}% inc={rule_accs[1]*100:.0f}% rep={rule_accs[2]*100:.0f}%] "
          f"params={n_params:,} time={elapsed:.0f}s")
