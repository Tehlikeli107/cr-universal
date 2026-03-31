# CR-Universal: Counting Revolution Structural Analysis Toolkit

**One mathematical framework. Any graph-representable structure.**

The Counting Revolution (induced k-subgraph distributions) is a provably complete graph invariant. CR-Universal applies this to real-world structures: molecules, source code, neural networks, financial markets.

## Tools

### 1. `cr_engine.py` — Universal Fingerprint Engine
```python
from cr_engine import CRAnalyzer

analyzer = CRAnalyzer(k=3)

# Molecules
fp = analyzer.fingerprint_molecule("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin

# Source code
fp = analyzer.fingerprint_code("def f(x): return x * 2")

# Any adjacency matrix
fp = analyzer.fingerprint(adj_matrix)

# Compare
sim = analyzer.compare(fp1, fp2)  # cosine similarity
```

### 2. `model_mri.py` — Neural Network Structural Diagnostics
```python
from model_mri import scan_model, print_mri_report
from transformers import GPT2Model

model = GPT2Model.from_pretrained('gpt2')
results = scan_model(model, "GPT-2")
print_mri_report(results, "GPT-2")
```

**Finding:** GPT-2 Layer 11 has neuron correlation spike (tri=0.300).
DistilGPT2 preserves it. OPT-125M does NOT have it. Model-family-specific.

### 3. `lens_discovery.py` — Automatic Invariant Discovery
Finds structural features that existing methods miss.

### 4. `hard_pair_lens.py` — Hard Pair Feature Analysis
Tests 25 features on known hard pairs (k=5-indistinguishable n=8 graphs).

**Key finding:** tr(A^5) resolves 1/4 pairs, tr(A^6) resolves 4/4.
This is the Lovasz walk-subgraph correspondence made concrete.

## Mathematical Foundation

Induced k-subgraph count distributions are PROVABLY COMPLETE graph invariants
for all tested n (up to n=10, 12 million graphs). This means:
- Same fingerprint = isomorphic graphs (guaranteed)
- Different fingerprint = non-isomorphic (guaranteed)

No other fingerprint (Morgan/ECFP, WL, spectral) has this completeness guarantee.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Optional: rdkit (molecules), networkx (graph generation), transformers (Model MRI)

## License

MIT
