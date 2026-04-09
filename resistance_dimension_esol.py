"""
Resistance Dimension Validation on ESOL Dataset

KEY HYPOTHESIS:
  alpha (resistance dimension) = 2/d_eff for a d-dimensional molecular graph

  Molecules are NOT lattices, but their resistance dimension captures:
  - Linear chains (alkanes): alpha -> 1.0 (1D)
  - Rings (benzene): alpha ~ 0.577 (cyclic 2D-ish)
  - Fused rings (naphthalene, anthracene): alpha grows (higher effective dimension)
  - Complex 3D drugs: alpha -> higher (many parallel paths, hub-and-spoke)

  PREDICTION: alpha is a NOVEL PREDICTOR of molecular properties
  because it captures the "effective dimensionality" of the molecular graph
  which correlates with:
  - Surface area (higher alpha = more compact = more accessible surface)
  - LogS solubility (more compact = less soluble)
  - LogP (compact molecules = more lipophilic)

VALIDATION: ESOL dataset (1128 molecules, log solubility)
"""

import numpy as np
import networkx as nx
from itertools import combinations
import time

# ===================== RESISTANCE DIMENSION =====================

def resistance_matrix_fast(G):
    """Compute resistance matrix via Laplacian pseudoinverse."""
    n = G.number_of_nodes()
    if n < 2:
        return np.zeros((1,1)), []

    nodes = list(G.nodes())
    try:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        L_pinv = np.linalg.pinv(L)
        diag_L = np.diag(L_pinv)
        R = diag_L[:, np.newaxis] + diag_L[np.newaxis, :] - 2 * L_pinv
        np.fill_diagonal(R, 0)
        return R, nodes
    except:
        return np.zeros((n,n)), nodes

def resistance_dimension(G):
    """
    Compute alpha: R(u,v) ~ dist(u,v)^alpha
    Returns (alpha, r2, kirchhoff, mean_R)
    """
    n = G.number_of_nodes()
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
        if n < 4:
            return 0.0, 0.0, 0.0, 0.0

    R, nodes = resistance_matrix_fast(G)
    node_list = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    # Kirchhoff index
    kirchhoff = np.sum(R) / 2.0
    mean_R = kirchhoff / (n*(n-1)/2) if n > 1 else 0

    # Pairwise distances
    all_dists = dict(nx.all_pairs_shortest_path_length(G))

    dist_R = []
    for u in node_list:
        for v in node_list:
            if u < v and u in node_idx and v in node_idx:
                d = all_dists.get(u, {}).get(v, 0)
                r = R[node_idx[u], node_idx[v]]
                if d > 0 and r > 1e-10:
                    dist_R.append((d, r))

    if len(dist_R) < 3:
        return 0.0, 0.0, kirchhoff, mean_R

    dists = np.array([p[0] for p in dist_R], dtype=float)
    Rs = np.array([p[1] for p in dist_R], dtype=float)

    log_d = np.log(dists)
    log_R = np.log(Rs)

    A = np.column_stack([log_d, np.ones(len(log_d))])
    result = np.linalg.lstsq(A, log_R, rcond=None)
    alpha = result[0][0]

    predicted = A @ result[0]
    ss_res = np.sum((log_R - predicted)**2)
    ss_tot = np.sum((log_R - np.mean(log_R))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return float(alpha), float(r2), float(kirchhoff), float(mean_R)

# ===================== LOAD ESOL =====================

import os, csv

def load_esol(path="esol_cached.csv"):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                smiles = row.get('smiles', row.get('SMILES', ''))
                logS_key = [k for k in row.keys() if 'solubility' in k.lower() or 'logs' in k.lower()]
                if not logS_key or not smiles:
                    continue
                logS = float(row[logS_key[0]])
                data.append((smiles, logS))
            except:
                continue
    return data

def smiles_to_graph(smi):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       aromatic=int(atom.GetIsAromatic()))
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       bond_type=int(bond.GetBondTypeAsDouble()))
        return G, mol
    except:
        return None

# ===================== COMPUTE FEATURES =====================

print("=== Resistance Dimension Validation on ESOL ===\n")

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    RDKIT = True
except ImportError:
    RDKIT = False
    print("RDKit not available. Cannot run molecular validation.")
    exit()

data = load_esol("esol_cached.csv")
print(f"Loaded {len(data)} molecules from ESOL\n")

print("Computing resistance dimension for ESOL molecules...")
t0 = time.time()

features = []
skipped = 0
for i, (smi, logS) in enumerate(data):
    result = smiles_to_graph(smi)
    if result is None:
        skipped += 1
        continue
    G, mol = result

    if G.number_of_nodes() < 4:
        skipped += 1
        continue

    # Resistance dimension
    alpha, r2_fit, kirchhoff, mean_R = resistance_dimension(G)

    # Basic RDKit descriptors
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    except:
        skipped += 1
        continue

    n = G.number_of_nodes()
    m = G.number_of_edges()

    features.append({
        'smiles': smi,
        'logS': logS,
        'alpha': alpha,
        'r2_fit': r2_fit,
        'kirchhoff': kirchhoff,
        'mean_R': mean_R,
        'n': n, 'm': m,
        'mw': mw, 'logp': logp,
        'hbd': hbd, 'hba': hba,
        'tpsa': tpsa,
        'n_rings': n_rings,
        'n_arom': n_arom,
    })

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(data)} molecules processed ({time.time()-t0:.1f}s)")

print(f"\nDone: {len(features)} molecules, {skipped} skipped ({time.time()-t0:.1f}s total)\n")

# ===================== STATISTICAL ANALYSIS =====================

import numpy as np

X_alpha = np.array([f['alpha'] for f in features])
X_kirchhoff = np.array([f['kirchhoff'] for f in features])
X_n = np.array([f['n'] for f in features])
X_logp = np.array([f['logp'] for f in features])
X_mw = np.array([f['mw'] for f in features])
X_tpsa = np.array([f['tpsa'] for f in features])
X_rings = np.array([f['n_rings'] for f in features])
X_mean_R = np.array([f['mean_R'] for f in features])
y = np.array([f['logS'] for f in features])

print("--- Pairwise Correlations with logS ---\n")
features_to_test = {
    'alpha (resistance dim)': X_alpha,
    'kirchhoff (K index)': X_kirchhoff,
    'kirchhoff/n^2 (norm)': X_kirchhoff / np.maximum(X_n**2, 1),
    'mean_R': X_mean_R,
    'n (heavy atoms)': X_n,
    'logP': X_logp,
    'MW': X_mw,
    'TPSA': X_tpsa,
    'n_rings': X_rings,
}

correlations = {}
for fname, vals in features_to_test.items():
    mask = np.isfinite(vals) & np.isfinite(y)
    if mask.sum() < 10:
        continue
    r = np.corrcoef(vals[mask], y[mask])[0, 1]
    correlations[fname] = r
    print(f"  {fname:25s}: r = {r:+.4f}")

print()
print("Most correlated with logS:", max(correlations, key=lambda k: abs(correlations[k])))

# ===================== REGRESSION =====================

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print()
print("--- Regression (5-fold CV R^2 on logS) ---\n")

# Build feature matrices
X_basic = np.column_stack([X_n, X_logp, X_mw, X_tpsa, X_rings])
X_resistance = np.column_stack([X_alpha, X_kirchhoff/np.maximum(X_n**2,1), X_mean_R])
X_combined = np.hstack([X_basic, X_resistance])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, X in [
    ("Basic (n, logP, MW, TPSA, rings)  [5 feats]", X_basic),
    ("Resistance (alpha, K_norm, meanR) [3 feats]", X_resistance),
    ("Combined                           [8 feats]", X_combined),
]:
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_m, y_m = X[mask], y[mask]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_m)

    scores = cross_val_score(Ridge(), X_s, y_m, cv=kf, scoring='r2')
    print(f"  {name}: R^2 = {scores.mean():.4f} +/- {scores.std():.4f}")

# ===================== ALPHA DISTRIBUTION BY MOLECULE TYPE =====================

print()
print("--- Resistance Dimension by Molecule Type ---\n")
print("Testing if alpha characterizes molecular topology class:\n")

from rdkit.Chem import rdMolDescriptors

def classify_mol(mol):
    """Rough classification of molecule type."""
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_atoms = mol.GetNumHeavyAtoms()

    if n_rings == 0:
        return "acyclic"
    elif n_arom == 0:
        return "alicyclic"
    elif n_arom == 1:
        return "mono-aromatic"
    elif n_arom == 2:
        return "bi-aromatic"
    else:
        return "poly-aromatic"

type_alphas = {}
for f in features:
    result = smiles_to_graph(f['smiles'])
    if result is None: continue
    G, mol = result
    mol_type = classify_mol(mol)
    if mol_type not in type_alphas:
        type_alphas[mol_type] = []
    type_alphas[mol_type].append(f['alpha'])

print(f"{'Type':15s}  {'n':5s}  {'mean_alpha':10s}  {'std_alpha':9s}  {'range'}")
print("-" * 60)
for t in ['acyclic', 'alicyclic', 'mono-aromatic', 'bi-aromatic', 'poly-aromatic']:
    if t not in type_alphas:
        continue
    alphas = np.array(type_alphas[t])
    print(f"{t:15s}  {len(alphas):5d}  {np.mean(alphas):10.4f}  {np.std(alphas):9.4f}  "
          f"[{np.min(alphas):.3f}, {np.max(alphas):.3f}]")

print()
print("HYPOTHESIS: acyclic < alicyclic < mono-aromatic < bi-aromatic < poly-aromatic")
print("Higher rings = more parallel paths = higher effective dimension = higher alpha")

# ===================== ALPHA vs N_ATOMS =====================

print()
print("--- Is alpha Independent of Molecular Size? ---\n")
print("(If alpha correlates strongly with n_atoms, it's redundant with size)\n")

r_alpha_n = np.corrcoef(X_alpha, X_n)[0,1]
r_kirchhoff_n = np.corrcoef(X_kirchhoff, X_n)[0,1]
r_alpha_rings = np.corrcoef(X_alpha, X_rings)[0,1]

print(f"  corr(alpha, n_atoms)   = {r_alpha_n:.4f}")
print(f"  corr(kirchhoff, n_atoms) = {r_kirchhoff_n:.4f}")
print(f"  corr(alpha, n_rings)   = {r_alpha_rings:.4f}")
print()
if abs(r_alpha_n) < 0.5:
    print("  alpha is INDEPENDENT of molecular size! (genuine new descriptor)")
elif abs(r_alpha_n) < 0.8:
    print("  alpha is PARTIALLY correlated with size (some redundancy)")
else:
    print("  alpha is HIGHLY correlated with size (mostly redundant with n_atoms)")

print()
print("For comparison: kirchhoff index is highly correlated with size")
print("(Kirchhoff ~ O(n^3) for acyclic, O(n^2) for rings)")
print("alpha should be SIZE-INDEPENDENT (dimensionless by construction)")

# ===================== SCATTER ANALYSIS =====================

print()
print("--- Best Individual Predictors Ranked ---\n")

ranked = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for rank, (fname, r) in enumerate(ranked, 1):
    direction = "+" if r > 0 else "-"
    print(f"  {rank}. {fname:30s}: |r| = {abs(r):.4f} ({direction})")

print()
print("CONCLUSION:")
print("  If alpha appears in top 3: resistance dimension is a genuine novel descriptor")
print("  If alpha is weak: the tool has mathematical elegance but limited practical value")
