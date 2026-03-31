"""CR-Universal: FDA 174 drug clustering by structural fingerprint."""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))
from cr_engine import CRAnalyzer, MoleculeAdapter
import torch
import numpy as np

analyzer = CRAnalyzer(k=3)

# Load FDA drugs
drugs = []
fda_path = r"C:\Users\salih\Desktop\gpu-assembly-index\fda_drugs_smiles.csv"
with open(fda_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        drugs.append((row['name'], row['SMILES']))
        if len(drugs) >= 50: break  # first 50 for speed

print(f"CR-UNIVERSAL: FDA Drug Clustering ({len(drugs)} drugs)")
print("=" * 55)

# Fingerprint all drugs
t0 = time.time()
fps = {}
for name, smiles in drugs:
    fp = analyzer.fingerprint_molecule(smiles)
    if fp is not None:
        fps[name] = fp
elapsed = time.time() - t0
print(f"Fingerprinted {len(fps)} drugs in {elapsed:.1f}s\n")

# Batch similarity via matrix ops
print("[COMPUTING SIMILARITY MATRIX]")
names = list(fps.keys())
fp_matrix = torch.stack([fps[n] for n in names])  # [N_drugs, 4]
# Cosine similarity matrix
norms = fp_matrix.norm(dim=1, keepdim=True) + 1e-10
fp_norm = fp_matrix / norms
sim_matrix = fp_norm @ fp_norm.T  # [N, N]

# Extract top pairs
pairs = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pairs.append((names[i], names[j], sim_matrix[i,j].item()))
pairs.sort(key=lambda x: -x[2])

for n1, n2, sim in pairs[:10]:
    print(f"  {n1:15s} <-> {n2:15s}  sim={sim:.4f}")

# Find top-10 most DIFFERENT pairs
print("\n[TOP 10 MOST DIFFERENT DRUG PAIRS]")
for n1, n2, sim in pairs[-10:]:
    print(f"  {n1:15s} <-> {n2:15s}  sim={sim:.4f}")

# Cluster: for each drug, find nearest neighbor
print("\n[NEAREST NEIGHBOR CLUSTERING]")
# Group into clusters based on high similarity (>0.99)
clusters = {}
assigned = set()
for n1, n2, sim in pairs:
    if sim < 0.995: break
    if n1 not in assigned and n2 not in assigned:
        clusters[n1] = [n1, n2]
        assigned.add(n1); assigned.add(n2)
    elif n1 in assigned and n2 not in assigned:
        for c in clusters.values():
            if n1 in c: c.append(n2); assigned.add(n2); break
    elif n2 in assigned and n1 not in assigned:
        for c in clusters.values():
            if n2 in c: c.append(n1); assigned.add(n1); break

print(f"  {len(clusters)} clusters found (sim > 0.995)")
for center, members in list(clusters.items())[:8]:
    print(f"  Cluster: {', '.join(members[:5])}")

# Diversity analysis
print(f"\n[DIVERSITY ANALYSIS]")
all_sims = [s for _,_,s in pairs]
print(f"  Mean similarity: {np.mean(all_sims):.3f}")
print(f"  Std: {np.std(all_sims):.3f}")
print(f"  Min: {np.min(all_sims):.3f}")
print(f"  Max: {np.max(all_sims):.3f}")
print(f"  Pairs > 0.99: {sum(1 for s in all_sims if s > 0.99)}")
print(f"  Pairs < 0.90: {sum(1 for s in all_sims if s < 0.90)}")
