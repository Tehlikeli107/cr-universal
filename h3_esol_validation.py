"""
H3_labeled + n_atoms as Solubility Predictor: ESOL Validation

KEY FINDING FROM UNIFIED FRAMEWORK:
  H3_labeled + n_atoms = LOO-R2=0.693 on 22-molecule test
  SAI(SMILES) is redundant with n_atoms (r=0.939)
  H3_labeled is INDEPENDENT of n_atoms (r=0.489)

THIS IS REMARKABLE because:
  H3_labeled = CR k=3 entropy with atom-type labels
  - Captures: diversity of 3-atom environments
  - Only 3D local structure, no global topology
  - Just 4 features (k=3 with labels has only a few types)

VALIDATE ON ESOL (628 molecules, scaffold split):
  Baseline: n_atoms alone
  New: n_atoms + H3_labeled
  Reference: ECFP4 Morgan fingerprint (standard)

If n_atoms + H3_labeled beats n_atoms significantly on ESOL:
  -> H3_labeled is a GENUINELY USEFUL new feature for drug discovery
  -> Simple 2-feature model explains meaningful variance in solubility
"""

import numpy as np
from collections import Counter
from itertools import combinations
from math import log
import os
import sys

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT = True
except ImportError:
    print("RDKit not available")
    sys.exit(1)

import networkx as nx
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ===================== MOLECULE FEATURES =====================

def mol_to_graph(mol):
    """Convert RDKit mol to typed NetworkX graph."""
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   atomic_num=atom.GetAtomicNum(),
                   degree=atom.GetDegree(),
                   aromatic=int(atom.GetIsAromatic()))
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   bond_type=int(bond.GetBondTypeAsDouble()))
    return G

def cr_k3_entropy_labeled(G, max_combos=2000):
    """H_3 with atom symbol labels (full diversity measure)."""
    nodes = list(G.nodes(data=True))
    n = len(nodes)
    if n < 3: return 0, Counter()

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
    return H, type_counts

def cr_k3_entropy_unlab(G, max_combos=2000):
    """H_3 unlabeled (just edge counts)."""
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
    return sum(-c/total*log(c/total) for c in type_counts.values() if c > 0)

def compute_features(smiles):
    """Compute all features for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    Chem.SanitizeMol(mol)

    G = mol_to_graph(mol)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    H3_lab, _ = cr_k3_entropy_labeled(G)
    H3_unl = cr_k3_entropy_unlab(G)

    # Standard RDKit features
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # ECFP4 fingerprint
    from rdkit.Chem import AllChem
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    fp_arr = np.array(fp)

    # Atom type histogram (simple compositional features)
    atom_types = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    n_C = atom_types.get('C', 0)
    n_N = atom_types.get('N', 0)
    n_O = atom_types.get('O', 0)
    n_S = atom_types.get('S', 0)
    n_halogen = atom_types.get('F', 0) + atom_types.get('Cl', 0) + atom_types.get('Br', 0) + atom_types.get('I', 0)

    return {
        'n': n, 'm': m, 'density': nx.density(G),
        'H3_lab': H3_lab, 'H3_unl': H3_unl,
        'mw': mw, 'logp': logp, 'tpsa': tpsa,
        'hbd': hbd, 'hba': hba, 'rings': rings,
        'rotbonds': rotbonds, 'arom_rings': arom_rings,
        'frac_csp3': frac_csp3,
        'n_C': n_C, 'n_N': n_N, 'n_O': n_O, 'n_S': n_S, 'n_hal': n_halogen,
        'fp': fp_arr,
    }

# ===================== LOAD ESOL DATASET =====================

print("=== H3_labeled + n_atoms: ESOL Validation ===\n")

# Load ESOL
esol_path = r"C:\Users\salih\Desktop\cr-universal\esol_cached.csv"
if not os.path.exists(esol_path):
    print(f"ESOL not found at {esol_path}")
    import urllib.request, io
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    try:
        r = urllib.request.urlopen(url, timeout=15)
        content = r.read().decode()
        with open(esol_path, 'w') as f: f.write(content)
        print(f"Downloaded ESOL to {esol_path}")
    except Exception as e:
        print(f"Download failed: {e}")

import csv
data = []
try:
    with open(esol_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get('smiles', row.get('SMILES', ''))
            logS_str = row.get('measured log solubility in mols per litre',
                               row.get('logS', row.get('measured_logS', '')))
            if not logS_str: continue
            logS = float(logS_str)
            if smiles: data.append((smiles, logS))
    print(f"Loaded {len(data)} molecules from ESOL")
except Exception as e:
    print(f"Error loading ESOL: {e}")
    # Use our 22 test molecules instead
    data = [
        ("O", 0.0), ("CCO", -0.77), ("CC(O)=O", -0.45),
        ("c1ccccc1", -1.58), ("Cc1ccccc1", -2.39),
        ("c1ccc2ccccc2c1", -3.18), ("c1ccc2cc3ccccc3cc2c1", -5.18),
        ("c1cc2ccc3cccc4ccc(c1)c2c34", -6.25),
        ("c1ccc(-c2ccccc2)cc1", -4.57),
        ("Cn1cnc2c1c(=O)n(c(=O)n2C)C", -1.61),
        ("CC(=O)Oc1ccccc1C(O)=O", -2.89),
        ("CC(C)Cc1ccc(cc1)C(C)C(O)=O", -4.36),
        ("NCCc1ccc(O)c(O)c1", -1.12),
        ("NCCc1c[nH]c2ccc(O)cc12", -1.86),
    ]
    print(f"Using {len(data)} test molecules")

# Use all molecules (suppress rdkit warnings)
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
np.random.seed(42)
print(f"Using all {len(data)} molecules")

# ===================== COMPUTE FEATURES =====================

print("Computing features...", flush=True)
features = []
ys = []
scaffolds = []

from rdkit.Chem.Scaffolds import MurckoScaffold

for i, (smiles, logS) in enumerate(data):
    if i % 50 == 0: print(f"  {i}/{len(data)}...", flush=True)
    feat = compute_features(smiles)
    if feat is None: continue

    # Scaffold for split
    mol = Chem.MolFromSmiles(smiles)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        scaffold = smiles
    scaffolds.append(scaffold)
    features.append(feat)
    ys.append(logS)

print(f"Successfully processed {len(features)} molecules\n")

# ===================== SCAFFOLD SPLIT =====================

# Scaffold split: unique scaffolds
unique_scaffolds = list(set(scaffolds))
np.random.shuffle(unique_scaffolds)
n_test = max(1, len(unique_scaffolds) // 5)
test_scaffolds = set(unique_scaffolds[:n_test])

train_idx = [i for i, sc in enumerate(scaffolds) if sc not in test_scaffolds]
test_idx = [i for i, sc in enumerate(scaffolds) if sc in test_scaffolds]

print(f"Scaffold split: {len(train_idx)} train, {len(test_idx)} test\n")

y = np.array(ys)
y_train, y_test = y[train_idx], y[test_idx]

# Build feature matrices
def get_X(feat_list, feat_names):
    return np.array([[f[n] for n in feat_names] for f in feat_list])

# Feature sets to test
feature_configs = {
    'n_atoms': ['n'],
    'H3_labeled': ['H3_lab'],
    'n_atoms + H3_labeled': ['n', 'H3_lab'],
    'n_atoms + H3_unlab': ['n', 'H3_unl'],
    'n_atoms + mw + logp': ['n', 'mw', 'logp'],
    'rdkit_basic (mw+logp+tpsa+hbd+hba)': ['mw', 'logp', 'tpsa', 'hbd', 'hba'],
    'rdkit_full': ['mw', 'logp', 'tpsa', 'hbd', 'hba', 'rings', 'rotbonds', 'arom_rings', 'frac_csp3'],
    'n + H3_lab + rdkit_basic': ['n', 'H3_lab', 'mw', 'logp', 'tpsa', 'hbd', 'hba'],
}

print(f"{'Feature set':45s}  {'Test R2':8s}  {'Test MAE':9s}")
print("-" * 70)

for config_name, feat_names in feature_configs.items():
    if not all(fn in features[0] for fn in feat_names):
        print(f"{config_name:45s}  {'SKIP':8s}")
        continue

    X_train = get_X([features[i] for i in train_idx], feat_names)
    X_test = get_X([features[i] for i in test_idx], feat_names)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    # Ridge regression
    clf = Ridge(alpha=1.0)
    clf.fit(X_tr_sc, y_train)
    preds = clf.predict(X_te_sc)

    ss_res = np.sum((y_test - preds)**2)
    ss_tot = np.sum((y_test - y_test.mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    mae = np.mean(np.abs(y_test - preds))

    print(f"{config_name:45s}  {r2:8.3f}  {mae:9.3f}")

# ECFP4 comparison
print()
fp_train = np.array([features[i]['fp'] for i in train_idx])
fp_test = np.array([features[i]['fp'] for i in test_idx])

clf_fp = Ridge(alpha=1.0)
clf_fp.fit(fp_train, y_train)
preds_fp = clf_fp.predict(fp_test)
r2_fp = 1 - np.sum((y_test - preds_fp)**2) / np.sum((y_test - y_test.mean())**2)
mae_fp = np.mean(np.abs(y_test - preds_fp))
print(f"{'ECFP4 (512-bit Morgan)':45s}  {r2_fp:8.3f}  {mae_fp:9.3f}")

# ECFP4 + H3_labeled
X_fp_h3_train = np.column_stack([fp_train, [features[i]['H3_lab'] for i in train_idx]])
X_fp_h3_test = np.column_stack([fp_test, [features[i]['H3_lab'] for i in test_idx]])
clf_fp_h3 = Ridge(alpha=1.0)
clf_fp_h3.fit(X_fp_h3_train, y_train)
preds_fp_h3 = clf_fp_h3.predict(X_fp_h3_test)
r2_fp_h3 = 1 - np.sum((y_test - preds_fp_h3)**2) / np.sum((y_test - y_test.mean())**2)
mae_fp_h3 = np.mean(np.abs(y_test - preds_fp_h3))
print(f"{'ECFP4 + H3_labeled':45s}  {r2_fp_h3:8.3f}  {mae_fp_h3:9.3f}")

print()
print("=== ANALYSIS ===\n")
print("KEY QUESTION: Does H3_labeled add information BEYOND standard features?")
print()
print("If n_atoms + H3_labeled > n_atoms SIGNIFICANTLY:")
print("  H3_labeled captures structural diversity independent of size")
print()
print("If ECFP4 + H3_labeled > ECFP4:")
print("  H3_labeled adds information even to state-of-the-art fingerprints!")
print("  This would mean CR entropy = a new, orthogonal axis of molecular complexity")
