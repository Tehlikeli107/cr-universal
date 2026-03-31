"""Fast demo: molecules + code only (no GPT-2 loading)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cr_engine import CRAnalyzer, MoleculeAdapter, CodeAdapter
import torch

print("CR-UNIVERSAL: Quick Demo")
print("=" * 55)

analyzer = CRAnalyzer(k=3)  # k=3 fast; k=4 too slow for Python loops
# TODO: vectorize k=4 for production use

# Molecules
print("\n[MOLECULES]")
mols = {
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'ethanol': 'CCO',
    'benzene': 'C1=CC=CC=C1',
    'paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
    'naproxen': 'COC1=CC2=CC(C(C)C(=O)O)=CC=C2C=C1',
    'glycine': 'NCC(=O)O',
}
for name, smiles in mols.items():
    fp = analyzer.fingerprint_molecule(smiles)
    if fp is not None:
        analyzer.store(f"mol:{name}", fp)
        sig = ' '.join(f'{v:.3f}' for v in fp.tolist())
        print(f"  {name:12s}: [{sig}]")

# Code
print("\n[SOURCE CODE]")
codes = {
    'loop': 'for i in range(10):\n    x = i * 2\n    print(x)',
    'nested': 'for i in range(5):\n    for j in range(5):\n        x = i + j\n        y = x * 2',
    'branching': 'def f(x):\n    if x > 0:\n        return x * 2\n    elif x == 0:\n        return 0\n    else:\n        return -x',
    'class': 'class Foo:\n    def __init__(self, x):\n        self.x = x\n    def bar(self):\n        return self.x + 1\n    def baz(self, y):\n        return self.bar() * y',
    'recursive': 'def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)',
    'list_comp': 'result = [x**2 for x in range(100) if x % 3 == 0]\nfiltered = [r for r in result if r > 50]',
}
for name, code in codes.items():
    fp = analyzer.fingerprint_code(code)
    if fp is not None:
        analyzer.store(f"code:{name}", fp)
        sig = ' '.join(f'{v:.3f}' for v in fp.tolist())
        print(f"  {name:12s}: [{sig}]")

# Cross-domain similarity
print("\n[CROSS-DOMAIN: Which code is most structurally similar to which molecule?]")
mol_items = [(n, fp) for n, fp in analyzer.library.items() if n.startswith('mol:')]
code_items = [(n, fp) for n, fp in analyzer.library.items() if n.startswith('code:')]

for cn, cfp in code_items:
    best_mol = None; best_sim = -1
    for mn, mfp in mol_items:
        sim = analyzer.compare(cfp, mfp)
        if sim > best_sim:
            best_sim = sim; best_mol = mn
    print(f"  {cn:18s} <-> {best_mol:18s}  sim={best_sim:.3f}")

# Intra-domain: molecule clustering
print("\n[MOLECULE CLUSTERING by CR fingerprint]")
for i, (n1, fp1) in enumerate(mol_items):
    sims = []
    for n2, fp2 in mol_items:
        if n1 != n2:
            sims.append((n2, analyzer.compare(fp1, fp2)))
    sims.sort(key=lambda x: -x[1])
    nearest = sims[0] if sims else ("?", 0)
    print(f"  {n1:18s} nearest: {nearest[0]:18s} ({nearest[1]:.3f})")

# Structural report
print("\n[STRUCTURAL REPORT]")
adj = MoleculeAdapter.to_graph('CC(=O)OC1=CC=CC=C1C(=O)O')
if adj is not None:
    report = analyzer.structural_report(adj, "aspirin")
    for k, v in report.items():
        if isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.3f}' for x in v)}]")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

print(f"\nLibrary: {len(analyzer.library)} items, "
      f"{len(set(n.split(':')[0] for n in analyzer.library))} domains")
