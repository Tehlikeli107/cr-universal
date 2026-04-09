"""
Unified Structural Complexity Framework

Tests: Are our new complexity measures INDEPENDENT or REDUNDANT?

Measures:
  1. CR k=2 entropy (H_k2)
  2. CR k=3 entropy (H_k3)
  3. SAI (sequence assembly index of SMILES)
  4. n_atoms (trivial baseline)
  5. Graph density
  6. Clustering coefficient
  7. Spectral gap (algebraic connectivity)

For ESOL molecules:
  - Compute all 7 measures
  - Correlate them pairwise
  - How many INDEPENDENT axes of complexity exist?
  - Which measures best predict solubility?

HYPOTHESIS: SAI and CR entropy capture ORTHOGONAL complexity aspects
  SAI = sequential copy-economy (1D view: string)
  CR k=3 entropy = structural diversity (3D view: graph)
  -> Low correlation -> both needed

IF CONFIRMED: SAI + CR = new 2-axis complexity taxonomy
  Molecules classified as:
    - Low SAI, Low CR: simple symmetric (benzene)
    - Low SAI, High CR: repetitive but structurally diverse
    - High SAI, Low CR: random-like but regular topology
    - High SAI, High CR: maximally complex (caffeine, drug-like)

SECOND TEST: Do SAI and CR predict the SAME or DIFFERENT molecular properties?
  - If same property: redundant (one sufficient)
  - If different: complementary (both needed)
"""

import numpy as np
from collections import Counter
from itertools import combinations
import networkx as nx
from math import log
import time
import os
import sys

# ===================== SMILES PARSER (simplified) =====================

def smiles_to_graph(smiles):
    """
    Parse SMILES to graph with atom types.
    Simplified: handles basic organic SMILES (no stereo, no isotopes).
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), symbol=atom.GetSymbol(),
                      atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                      bond_type=int(bond.GetBondTypeAsDouble()))
        return G
    except ImportError:
        return None

# ===================== SAI (from sequence_assembly.py) =====================

def sai_repair(seq):
    """Re-Pair based SAI estimate."""
    parts = list(seq)
    vocab = set(seq)
    ops = len(vocab)
    rule_count = 0

    for iteration in range(len(seq)):
        if len(parts) <= 2: break
        bigrams = Counter()
        for i in range(len(parts)-1):
            bigrams[(parts[i], parts[i+1])] += 1
        if not bigrams: break
        most_common, freq = bigrams.most_common(1)[0]
        if freq < 2: break
        a, b = most_common
        new_sym = f"_{iteration}"
        new_parts = []
        i = 0
        while i < len(parts):
            if i < len(parts)-1 and parts[i] == a and parts[i+1] == b:
                new_parts.append(new_sym)
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1
        rule_count += 1
        parts = new_parts

    join_ops = len(parts) - 1
    return ops + rule_count + join_ops

# ===================== CR ENTROPY =====================

def cr_k3_entropy(G, max_combos=3000):
    nodes = list(G.nodes())
    n = len(nodes)
    if n < 3: return 0
    combos = list(combinations(range(n), 3))
    if len(combos) > max_combos:
        idx = np.random.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in idx]
    type_counts = Counter()
    for i,j,k in combos:
        a,b,c = nodes[i], nodes[j], nodes[k]
        ne = int(G.has_edge(a,b)) + int(G.has_edge(a,c)) + int(G.has_edge(b,c))
        type_counts[ne] += 1
    total = sum(type_counts.values())
    H = sum(-c/total*log(c/total) for c in type_counts.values() if c > 0)
    return H

def cr_k3_entropy_labeled(G, max_combos=3000):
    """H_3 with atom-type labels."""
    nodes = list(G.nodes(data=True))
    n = len(nodes)
    if n < 3: return 0
    combos = list(combinations(range(n), 3))
    if len(combos) > max_combos:
        idx = np.random.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in idx]
    type_counts = Counter()
    for i,j,k in combos:
        ai, di = nodes[i]
        aj, dj = nodes[j]
        ak, dk = nodes[k]
        si = di.get('symbol', 'C')
        sj = dj.get('symbol', 'C')
        sk = dk.get('symbol', 'C')
        ne = int(G.has_edge(ai,aj)) + int(G.has_edge(ai,ak)) + int(G.has_edge(aj,ak))
        state_tri = tuple(sorted([si, sj, sk]))
        type_counts[(state_tri, ne)] += 1
    total = sum(type_counts.values())
    H = sum(-c/total*log(c/total) for c in type_counts.values() if c > 0)
    return H

# ===================== MAIN EXPERIMENT =====================

print("=== Unified Structural Complexity Framework ===\n")

# Check if rdkit available
try:
    from rdkit import Chem
    RDKIT = True
    print("RDKit available: using molecular graphs\n")
except ImportError:
    RDKIT = False
    print("RDKit not available: using SMILES-level analysis only\n")

# Test molecules with known properties
# Format: (name, SMILES, logS)  [logS = log10 solubility in mol/L]
molecules = [
    # Very soluble (logS > -1)
    ("water", "O", 0.0),
    ("ethanol", "CCO", -0.77),
    ("acetic_acid", "CC(O)=O", -0.45),
    ("glycine", "NCC(O)=O", -0.85),
    # Moderately soluble (-3 < logS < -1)
    ("benzene", "c1ccccc1", -1.58),
    ("toluene", "Cc1ccccc1", -2.39),
    ("phenol", "Oc1ccccc1", -1.35),
    ("aniline", "Nc1ccccc1", -1.02),
    ("naphthalene", "c1ccc2ccccc2c1", -3.18),
    ("pyridine", "c1ccncc1", -0.98),
    # Poorly soluble (logS < -3)
    ("anthracene", "c1ccc2cc3ccccc3cc2c1", -5.18),
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", -6.25),
    ("biphenyl", "c1ccc(-c2ccccc2)cc1", -4.57),
    ("cholesterol", "OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C", -6.90),
    ("caffeine", "Cn1cnc2c1c(=O)n(c(=O)n2C)C", -1.61),
    ("aspirin", "CC(=O)Oc1ccccc1C(O)=O", -2.89),
    ("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O", -4.36),
    ("testosterone", "OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C", -6.54),
    # Drugs
    ("penicillin_g", "OC(=O)C1N2CC(SC2(C)C)C1NC(=O)Cc1ccccc1", -2.44),
    ("morphine", "OC1=CC2=C(CC1)CC(N(CC2)C)C1=CC=CC=C1O", -2.92),
    ("dopamine", "NCCc1ccc(O)c(O)c1", -1.12),
    ("serotonin", "NCCc1c[nH]c2ccc(O)cc12", -1.86),
]

print(f"Testing {len(molecules)} molecules\n")
print(f"Computing complexity measures...")

results = []
for name, smiles, logS in molecules:
    n = len(smiles)
    sai = sai_repair(smiles)
    sai_norm = sai / max(n, 1)

    if RDKIT:
        G = smiles_to_graph(smiles)
        if G is None:
            G = nx.path_graph(n//5 + 2)  # fallback
        n_atoms = G.number_of_nodes()
        n_bonds = G.number_of_edges()
        density = nx.density(G)
        try:
            cc = nx.average_clustering(G)
        except:
            cc = 0
        try:
            H3_unlab = cr_k3_entropy(G)
            H3_lab = cr_k3_entropy_labeled(G)
        except:
            H3_unlab = H3_lab = 0
        try:
            L = nx.laplacian_matrix(G).toarray()
            eigvals = np.sort(np.linalg.eigvalsh(L))
            spec_gap = eigvals[1] if len(eigvals) > 1 else 0
        except:
            spec_gap = 0
    else:
        # Without rdkit: estimate from SMILES
        n_atoms = smiles.count('C') + smiles.count('N') + smiles.count('O') + smiles.count('S')
        n_bonds = smiles.count('=') * 2 + smiles.count('#') * 3 + len([c for c in smiles if c.lower() == c and c.isalpha()])
        density = n_bonds / max(n_atoms*(n_atoms-1)//2, 1)
        cc = 0.1 * (smiles.count('1') + smiles.count('2'))  # ring count proxy
        H3_unlab = 0
        H3_lab = 0
        spec_gap = 0

    results.append({
        'name': name,
        'smiles': smiles,
        'logS': logS,
        'sai': sai,
        'sai_norm': sai_norm,
        'n_atoms': n_atoms,
        'n_bonds': n_bonds,
        'density': density,
        'cc': cc,
        'H3_unlab': H3_unlab,
        'H3_lab': H3_lab,
        'spec_gap': spec_gap,
        'smiles_len': n,
    })

print(f"Done.\n")

# ===================== PRINT RESULTS =====================

print(f"{'Molecule':15s}  {'SAI':4s}  {'SAI/n':6s}  {'n_at':4s}  {'H3_lab':7s}  {'logS':6s}")
print("-" * 55)
for r in results:
    print(f"{r['name']:15s}  {r['sai']:4d}  {r['sai_norm']:6.3f}  {r['n_atoms']:4d}  {r['H3_lab']:7.3f}  {r['logS']:6.2f}")

print()

# ===================== CORRELATION ANALYSIS =====================

from scipy.stats import spearmanr, pearsonr

print("=== CORRELATION WITH logS (SOLUBILITY) ===\n")

logS = np.array([r['logS'] for r in results])
sai_vals = np.array([r['sai'] for r in results])
sai_norm_vals = np.array([r['sai_norm'] for r in results])
n_atoms_vals = np.array([r['n_atoms'] for r in results])
H3_lab_vals = np.array([r['H3_lab'] for r in results])
H3_unlab_vals = np.array([r['H3_unlab'] for r in results])
density_vals = np.array([r['density'] for r in results])
cc_vals = np.array([r['cc'] for r in results])

measures = [
    ('SAI', sai_vals),
    ('SAI_norm', sai_norm_vals),
    ('n_atoms', n_atoms_vals),
    ('H3_labeled', H3_lab_vals),
    ('H3_unlabeled', H3_unlab_vals),
    ('density', density_vals),
    ('clustering', cc_vals),
]

print(f"{'Measure':15s}  {'Pearson r':10s}  {'Spearman rho':12s}  {'Interpretation'}")
print("-" * 65)
for name, vals in measures:
    if np.std(vals) < 1e-10:
        print(f"{name:15s}  {'(constant)':10s}  {'(constant)':12s}")
        continue
    pr, pp = pearsonr(vals, logS)
    sr, sp = spearmanr(vals, logS)
    interp = ""
    if abs(sr) > 0.7: interp = "STRONG"
    elif abs(sr) > 0.4: interp = "moderate"
    else: interp = "weak"
    print(f"{name:15s}  {pr:+10.3f}  {sr:+12.3f}  {interp} (p={sp:.3f})")

print()

# ===================== PAIRWISE CORRELATIONS BETWEEN MEASURES =====================

print("=== PAIRWISE CORRELATIONS (are measures redundant?) ===\n")
print("High |r| = measures are REDUNDANT (capture same info)")
print("Low |r| = measures are INDEPENDENT (capture different info)")
print()

measure_vals = {n: v for n, v in measures if np.std(v) > 1e-10}
names = list(measure_vals.keys())
vals_matrix = np.array([measure_vals[n] for n in names])

print(f"{'':15s}", end="")
for n in names:
    print(f"  {n[:6]:6s}", end="")
print()
print("-" * (15 + 8*len(names)))

for i, ni in enumerate(names):
    print(f"{ni:15s}", end="")
    for j, nj in enumerate(names):
        vi, vj = vals_matrix[i], vals_matrix[j]
        if np.std(vi) < 1e-10 or np.std(vj) < 1e-10:
            print(f"  {'N/A':6s}", end="")
        else:
            r, _ = pearsonr(vi, vj)
            print(f"  {r:+6.3f}", end="")
    print()

print()
print("=== DISCOVERY SUMMARY ===\n")

# Find most independent pairs
from itertools import combinations as icombs
print("Most INDEPENDENT measure pairs (|r| < 0.3):")
for i, j in icombs(range(len(names)), 2):
    vi, vj = vals_matrix[i], vals_matrix[j]
    if np.std(vi) < 1e-10 or np.std(vj) < 1e-10:
        continue
    r, _ = pearsonr(vi, vj)
    if abs(r) < 0.3:
        print(f"  {names[i]:15s} x {names[j]:15s}: r={r:+.3f}")

print()
print("Most REDUNDANT measure pairs (|r| > 0.8):")
for i, j in icombs(range(len(names)), 2):
    vi, vj = vals_matrix[i], vals_matrix[j]
    if np.std(vi) < 1e-10 or np.std(vj) < 1e-10 or i == j:
        continue
    r, _ = pearsonr(vi, vj)
    if abs(r) > 0.8:
        print(f"  {names[i]:15s} x {names[j]:15s}: r={r:+.3f}")

print()

# ===================== LINEAR REGRESSION =====================
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

print("=== PREDICTIVE VALUE: Which combination predicts solubility best? ===\n")

valid_measures = [(n, v) for n, v in measures if np.std(v) > 1e-10]

feature_sets = {
    'n_atoms only': ['n_atoms'],
    'SAI only': ['SAI'],
    'H3_labeled only': ['H3_labeled'],
    'SAI + n_atoms': ['SAI', 'n_atoms'],
    'H3_labeled + n_atoms': ['H3_labeled', 'n_atoms'],
    'SAI + H3_labeled': ['SAI', 'H3_labeled'],
    'SAI + H3_labeled + n_atoms': ['SAI', 'H3_labeled', 'n_atoms'],
    'all measures': names,
}

print(f"{'Feature set':35s}  {'LOO-R2':8s}  {'Pearson r':10s}")
print("-" * 60)

for fs_name, feature_names in feature_sets.items():
    available = [n for n in feature_names if n in measure_vals]
    if not available:
        continue
    X = np.column_stack([measure_vals[n] for n in available])
    y = logS

    # LOO cross-validation
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    preds = []
    for i in range(len(y)):
        train_idx = [j for j in range(len(y)) if j != i]
        clf = Ridge(alpha=1.0)
        clf.fit(X_sc[train_idx], y[train_idx])
        preds.append(clf.predict(X_sc[i:i+1])[0])

    preds = np.array(preds)
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    r, _ = pearsonr(preds, y)
    print(f"{fs_name:35s}  {r2:8.3f}  {r:10.3f}")

print()

print("=== THEORETICAL SYNTHESIS ===\n")
print("New unified picture:")
print()
print("  SAI = 'copy-economy' of the SMILES string (1D assembly)")
print("    Captures: molecular SYMMETRY and REPETITIVENESS")
print("    High SAI = unique, no reusable patterns")
print()
print("  CR H3_labeled = structural DIVERSITY at 3-node scale")
print("    Captures: variety of local chemical environments")
print("    High H3 = many different local topologies")
print()
print("  n_atoms = SIZE (crude but powerful baseline)")
print()
print("INDEPENDENCE CLAIM: If SAI and H3_labeled are weakly correlated:")
print("  -> They capture DIFFERENT aspects of molecular complexity")
print("  -> SAI = 'how built', H3 = 'what it looks like'")
print()
print("PREDICTION: SAI + H3 together explain MORE variance in properties")
print("  than either alone, because they provide complementary information")
