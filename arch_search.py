"""
ARCHITECTURE SEARCH: Let the machine find the architecture.
Human designs NOTHING. Machine discovers EVERYTHING.

Search space: random computation graphs.
- Nodes = tensor operations (matmul, add, gelu, layernorm)
- Edges = data flow
- Graph structure = architecture

Evolution finds the graph. CR analyzes the structure.
Result: an architecture NO HUMAN would design.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

DEVICE = torch.device('cuda')

# ===== PRIMITIVE OPERATIONS =====
# These are the ONLY building blocks. No attention, no conv, no human design.

class OpMatMul(nn.Module):
    """Learned matrix multiplication."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d) * 0.02)
    def forward(self, x):
        return x @ self.W

class OpGELU(nn.Module):
    def forward(self, x): return F.gelu(x)

class OpLayerNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
    def forward(self, x): return self.norm(x)

class OpAdd(nn.Module):
    """Add two inputs (residual-like)."""
    def forward(self, x1, x2): return x1 + x2

class OpGate(nn.Module):
    """Learned gating: sigmoid(Wx) * y."""
    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d)
    def forward(self, x): return torch.sigmoid(self.W(x))

# ===== RANDOM ARCHITECTURE GENOME =====
# Genome = list of (op_type, input_indices)
# This defines a computation graph.

OP_TYPES = ['matmul', 'gelu', 'norm', 'gate']

def random_genome(n_nodes=6, d=64):
    """Generate a random computation graph."""
    genome = []
    for i in range(n_nodes):
        op = random.choice(OP_TYPES)
        # Input: any previous node (0 = original input)
        if i == 0:
            inp = 0
        else:
            inp = random.randint(0, i)
        genome.append((op, inp))
    return genome

def mutate_genome(genome):
    """Small mutation: change one node's op or input."""
    g = list(genome)
    idx = random.randint(0, len(g) - 1)
    mutation = random.choice(['op', 'input'])
    op, inp = g[idx]
    if mutation == 'op':
        op = random.choice(OP_TYPES)
    else:
        inp = random.randint(0, idx) if idx > 0 else 0
    g[idx] = (op, inp)
    return g

# ===== BUILD MODEL FROM GENOME =====

class GenomeModel(nn.Module):
    """Build a neural network from a genome (computation graph)."""
    def __init__(self, vocab, d, genome, max_seq=32):
        super().__init__()
        self.d = d
        self.genome = genome
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_seq, d)

        # Build operations from genome
        self.ops = nn.ModuleList()
        for op_type, _ in genome:
            if op_type == 'matmul':
                self.ops.append(OpMatMul(d))
            elif op_type == 'gelu':
                self.ops.append(OpGELU())
            elif op_type == 'norm':
                self.ops.append(OpLayerNorm(d))
            elif op_type == 'gate':
                self.ops.append(OpGate(d))

        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        B, N = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(N, device=ids.device))

        # Execute computation graph
        node_outputs = [x]  # node 0 = input
        for i, (op_type, inp_idx) in enumerate(self.genome):
            inp = node_outputs[inp_idx]
            out = self.ops[i](inp)
            # Residual: add to input
            out = out + inp
            node_outputs.append(out)

        # Output = last node
        return self.head(node_outputs[-1])

# ===== EVALUATION =====

def gen_data(V=20, seq=24, batch=32):
    t = torch.randint(0, V, (batch, seq), device=DEVICE)
    mid = seq // 2
    for b in range(batch):
        for i in range(2, mid): t[b, i] = t[b, i-2]
        for i in range(max(mid, 3), seq): t[b, i] = (t[b, i-1] + t[b, i-3]) % V
    return t[:, :-1], t[:, 1:]

def evaluate_genome(genome, V=20, d=32, seq=16, epochs=15):
    """Train model from genome, return accuracy."""
    try:
        model = GenomeModel(V, d, genome, seq).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            inp, tgt = gen_data(V, seq, 32)
            loss = F.cross_entropy(model(inp).reshape(-1, V), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            inp, tgt = gen_data(V, seq, 500)
            logits = model(inp)
            acc = (logits.argmax(-1) == tgt).float().mean().item()
        return acc, sum(p.numel() for p in model.parameters())
    except Exception:
        return 0.0, 0

# ===== EVOLUTIONARY SEARCH =====

def search(pop_size=20, generations=15, n_nodes=8):
    """Evolve architecture genomes."""
    print(f"Evolving architectures: pop={pop_size}, gens={generations}, nodes={n_nodes}")
    print("="*55)

    # Initial population
    population = [random_genome(n_nodes) for _ in range(pop_size)]
    best_acc = 0
    best_genome = None

    for gen in range(generations):
        # Evaluate all
        results = []
        for genome in population:
            acc, np_ = evaluate_genome(genome)
            results.append((genome, acc, np_))

        results.sort(key=lambda x: -x[1])
        gen_best = results[0]

        if gen_best[1] > best_acc:
            best_acc = gen_best[1]
            best_genome = gen_best[0]

        print(f"  Gen {gen+1:2d}: best={gen_best[1]*100:.1f}% "
              f"genome={[(op,inp) for op,inp in gen_best[0][:3]]}... "
              f"params={gen_best[2]:,}")

        # Selection + mutation
        elite = [r[0] for r in results[:pop_size//3]]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            parent = random.choice(elite)
            child = mutate_genome(parent)
            new_pop.append(child)
        population = new_pop

    return best_genome, best_acc

# ===== COMPARE WITH TRANSFORMER =====

class Transformer(nn.Module):
    def __init__(self, V, d=64, nh=4, nl=4, max_seq=32):
        super().__init__()
        self.embed=nn.Embedding(V,d); self.pos=nn.Embedding(max_seq,d)
        el=nn.TransformerEncoderLayer(d,nh,d*4,batch_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(el,nl); self.head=nn.Linear(d,V)
    def forward(self, ids):
        B,N=ids.shape; x=self.embed(ids)+self.pos(torch.arange(N,device=ids.device))
        mask=torch.triu(torch.ones(N,N,device=ids.device),1).bool()
        return self.head(self.enc(x,mask=mask))

if __name__ == "__main__":
    print("ARCHITECTURE SEARCH: Machine-designed vs Human-designed")
    print("="*55)

    V=20; d=32; SEQ=16

    # Transformer baseline
    print("\n[TRANSFORMER BASELINE]")
    torch.manual_seed(42)
    tf = Transformer(V, d, 4, 2, SEQ).to(DEVICE)
    opt = torch.optim.Adam(tf.parameters(), lr=1e-3)
    for ep in range(15):
        inp, tgt = gen_data(V, SEQ, 32)
        loss = F.cross_entropy(tf(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        inp, tgt = gen_data(V, SEQ, 500)
        tf_acc = (tf(inp).argmax(-1) == tgt).float().mean().item()
    tf_params = sum(p.numel() for p in tf.parameters())
    print(f"  Transformer: {tf_acc*100:.1f}% params={tf_params:,}")

    # Architecture search
    print("\n[MACHINE-DESIGNED ARCHITECTURE]")
    random.seed(42); torch.manual_seed(42)
    best_genome, best_acc = search(pop_size=8, generations=5, n_nodes=5)

    print(f"\n{'='*55}")
    print(f"RESULT:")
    print(f"  Transformer (human-designed): {tf_acc*100:.1f}%")
    print(f"  Evolved (machine-designed):   {best_acc*100:.1f}%")
    print(f"  Best genome: {best_genome}")

    if best_acc > tf_acc:
        print(f"\n*** MACHINE BEATS HUMAN! ***")
        print(f"  The machine found an architecture humans wouldn't design.")
        print(f"  This IS the 'insanlarin anlamayacagi arac'.")
    else:
        print(f"\n  Transformer still wins. Need more search budget.")
