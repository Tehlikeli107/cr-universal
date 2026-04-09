"""
CR Propagation Tensor: Structural Chain Rule

CORE IDEA: The CR histogram at scale k DETERMINES the histogram at k+1
(up to correction terms). This gives a "structural chain rule":

H_{k+1}(G) = T_{k->k+1} * H_k(G) + epsilon(G)

where T is the "propagation tensor" (k-subgraph -> k+1-subgraph transitions)
and epsilon is the correction term.

THE PROPAGATION TENSOR T_{k->k+1}:
  T[s'][s] = P(seeing (k+1)-subgraph type s' | a random k-subgraph is type s)
           = fraction of (k+1)-subgraphs that contain a type-s k-subgraph
             AND are of type s'

If T is "sharp" (each row has dominant entry): H_k determines H_{k+1}
If T is "diffuse" (many non-zero entries): H_k does NOT determine H_{k+1}

THE STRUCTURAL SUFFICIENCY THEOREM:
  H_k is a SUFFICIENT STATISTIC for H_{k+1} iff T is approximately invertible.

APPLICATIONS:
  1. If T is approximately invertible for molecular graphs:
     -> H_2 determines H_3 for molecules -> k=2 IS SUFFICIENT for property prediction!
  2. If T is NOT invertible for social networks:
     -> k=2 does NOT determine k=3 for social networks -> need higher k
  3. The "entropy decay" E[k] = H_{k+1}/H_k gives the "structural scale decay rate"

TESTING:
  - Compute T_{2->3} for molecular graphs (ESOL) vs random graphs
  - Measure invertibility (rank, condition number, entropy of each row)
  - If molecules: low condition number -> deterministic propagation
  - If random: high condition number -> non-deterministic
"""
import numpy as np
from collections import Counter
from itertools import combinations, permutations
import urllib.request, io
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

def mol_to_typed_graph(mol):
    nodes = {a.GetIdx(): a.GetAtomicNum() for a in mol.GetAtoms()}
    bmap = {Chem.rdchem.BondType.SINGLE:1, Chem.rdchem.BondType.DOUBLE:2,
            Chem.rdchem.BondType.TRIPLE:3, Chem.rdchem.BondType.AROMATIC:4}
    edges = {}
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = bmap.get(b.GetBondType(), 1)
        edges[(i,j)] = bt; edges[(j,i)] = bt
    return nodes, edges

def canonical_sub(nodes_dict, edges_dict, nlist):
    """Canonical k-subgraph form."""
    n = len(nlist); nl = sorted(nlist)
    if n <= 5:
        best = None
        for perm in permutations(range(n)):
            mat = tuple(tuple(nodes_dict[nl[perm[i]]] if i==j
                              else edges_dict.get((nl[perm[i]],nl[perm[j]]),0)
                              for j in range(n)) for i in range(n))
            if best is None or mat < best: best = mat
        return best
    at = tuple(sorted(nodes_dict[u] for u in nl))
    et = tuple(sorted((nodes_dict[nl[i]],nodes_dict[nl[j]],edges_dict.get((nl[i],nl[j]),0))
               for i in range(n) for j in range(i+1,n) if edges_dict.get((nl[i],nl[j]),0)>0))
    return (at, et)

def build_propagation_matrix(mols, max_mols=200, max_combos_k3=3000):
    """
    Build the propagation matrix T[k2_type][k3_type]:
    For each 3-subgraph, what 2-subgraph types does it contain?
    T[s2][s3] = fraction of 3-subgraphs of type s3 that contain a 2-subgraph of type s2

    Alternative: For each 2-subgraph of type s2, what 3-subgraphs does it appear in?
    R[s2][s3] = P(see s3 | the 3-subgraph contains this s2 subgraph)
    """
    # Collect all (k=2 type, k=3 type) co-occurrence pairs
    print("Building propagation data...", flush=True)

    k2_vocab = {}
    k3_vocab = {}
    co_occurrences = Counter()  # (k2_type, k3_type) -> count
    k2_marginals = Counter()    # k2_type -> total count
    k3_marginals = Counter()    # k3_type -> total count

    for mol_idx, mol in enumerate(mols[:max_mols]):
        if mol_idx % 50 == 0: print(f"  Mol {mol_idx}/{min(max_mols, len(mols))}...", flush=True)

        nodes, edges = mol_to_typed_graph(mol)
        nl = list(nodes.keys())
        if len(nl) < 3: continue

        # k=2: all pairs
        k2_types_mol = {}
        for i, j in combinations(nl, 2):
            key = (min(nodes[i],nodes[j]), max(nodes[i],nodes[j]),
                   edges.get((i,j), 0))
            if key not in k2_vocab:
                k2_vocab[key] = len(k2_vocab)
            k2_types_mol[(i,j)] = k2_vocab[key]
            k2_types_mol[(j,i)] = k2_vocab[key]  # symmetric
            k2_marginals[k2_vocab[key]] += 1

        # k=3: sample 3-subgraphs
        combos3 = list(combinations(nl, 3))
        if len(combos3) > max_combos_k3:
            combos3 = [combos3[i] for i in np.random.choice(len(combos3), max_combos_k3, replace=False)]

        for a,b,c in combos3:
            s3 = canonical_sub(nodes, edges, [a,b,c])
            if s3 not in k3_vocab:
                k3_vocab[s3] = len(k3_vocab)
            k3_idx = k3_vocab[s3]
            k3_marginals[k3_idx] += 1

            # For each pair in this triple, record co-occurrence
            for (u,v) in [(a,b),(a,c),(b,c)]:
                key = (min(u,v), max(u,v))
                k2_idx = k2_types_mol.get(key) or k2_types_mol.get((max(u,v),min(u,v)))
                if k2_idx is not None:
                    co_occurrences[(k2_idx, k3_idx)] += 1

    return k2_vocab, k3_vocab, co_occurrences, k2_marginals, k3_marginals

def analyze_propagation(k2_vocab, k3_vocab, co_occur, k2_marg, k3_marg):
    """Analyze the propagation tensor T[k2|k3] = P(k2 | k3)."""
    n2 = len(k2_vocab)
    n3 = len(k3_vocab)
    print(f"k=2 types: {n2}, k=3 types: {n3}")

    # Build T[k2][k3] = co_occur(k2,k3) / k3_marginal(k3)
    T_k2_given_k3 = np.zeros((n2, n3))
    for (k2, k3), count in co_occur.items():
        if k3_marg[k3] > 0:
            T_k2_given_k3[k2, k3] = count / k3_marg[k3]

    # Build T[k3][k2] = co_occur(k2,k3) / k2_marginal(k2) (reverse: predict k3 given k2)
    T_k3_given_k2 = np.zeros((n3, n2))
    for (k2, k3), count in co_occur.items():
        if k2_marg[k2] > 0:
            T_k3_given_k2[k3, k2] = count / k2_marg[k2]

    # Normalize columns of T_k3_given_k2 (each column = P(k3|k2), should sum to 1)
    col_sums = T_k3_given_k2.sum(axis=0)
    for j in range(n2):
        if col_sums[j] > 0:
            T_k3_given_k2[:, j] /= col_sums[j]

    # Analyze T_k3_given_k2: how "deterministic" is k3 given k2?
    # Entropy of each column = H[k3 | k2 = type_j]
    col_entropies = []
    for j in range(n2):
        col = T_k3_given_k2[:, j]
        col = col[col > 0]
        H = -np.sum(col * np.log(col)) if len(col) > 0 else 0
        col_entropies.append(H)

    H_k3_given_k2 = np.mean(col_entropies)
    H_k3_marginal = -sum((v/sum(k3_marg.values())) * np.log(v/sum(k3_marg.values()))
                         for v in k3_marg.values() if v > 0)

    # Mutual information I(k2; k3) = H(k3) - H(k3|k2)
    MI = H_k3_marginal - H_k3_given_k2

    return {
        'n2': n2, 'n3': n3,
        'H_k3': H_k3_marginal,
        'H_k3_given_k2': H_k3_given_k2,
        'MI_k2_k3': MI,
        'MI_fraction': MI / H_k3_marginal,  # How much of H(k3) is explained by k2?
        'col_entropies': col_entropies,
        'T': T_k3_given_k2
    }

# Load ESOL molecules
print("=== CR Propagation Tensor Analysis ===\n")
print("KEY QUESTION: Does H_k=2 determine H_k=3?\n")
print("MEASUREMENT: I(k2; k3) / H(k3) = fraction of k=3 info explained by k=2\n")

url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
req = urllib.request.Request(url, headers={'User-Agent':'Python'})
r = urllib.request.urlopen(req, timeout=10)
df = pd.read_csv(io.StringIO(r.read().decode()))
tgt_col = [c for c in df.columns if 'measured' in c.lower()][0]
mols, ys = [], []
for _, row in df.iterrows():
    smi = row.get('smiles', row.iloc[0])
    mol = Chem.MolFromSmiles(str(smi))
    if mol:
        try:
            t = float(row[tgt_col])
            if not np.isnan(t): mols.append(mol); ys.append(t)
        except: pass

print(f"ESOL molecules: {len(mols)}\n")

# ESOL analysis
k2v, k3v, co, k2m, k3m = build_propagation_matrix(mols, max_mols=300)
res = analyze_propagation(k2v, k3v, co, k2m, k3m)
print(f"\nMolecular graphs (ESOL):")
print(f"  H(k3):         {res['H_k3']:.3f} nats ({res['n3']} types)")
print(f"  H(k3 | k2):    {res['H_k3_given_k2']:.3f} nats")
print(f"  I(k2; k3):     {res['MI_k2_k3']:.3f} nats")
print(f"  MI fraction:   {res['MI_fraction']:.3f} (= {res['MI_fraction']*100:.1f}% of k3 info explained by k2)")
print()
print("  Col entropy distribution (H[k3|k2=type]):")
ents = res['col_entropies']
print(f"  mean={np.mean(ents):.3f}, min={np.min(ents):.3f}, max={np.max(ents):.3f}")
print(f"  fraction of k2 types with H[k3|k2]<0.5 nats: {(np.array(ents)<0.5).mean():.3f}")

# Compare with random graphs
print(f"\nRandom ER graphs (same size distribution as ESOL):")
import networkx as nx
rand_mols_sim = []
# Simulate random labeled graphs of similar sizes
n_atoms_list = [mol.GetNumAtoms() for mol in mols[:100]]
for n_atoms in n_atoms_list[:100]:
    G = nx.erdos_renyi_graph(n_atoms, 0.3, seed=n_atoms)
    # Assign random labels
    class FakeMol:
        def __init__(self, G): self.G = G
        def GetAtoms(self): return [FakeAtom(self.G.nodes[n].get('l', 1)) for n in self.G.nodes()]
        def GetBonds(self): return [FakeBond(u,v,1) for u,v in self.G.edges()]
        def GetNumAtoms(self): return self.G.number_of_nodes()
    class FakeAtom:
        def __init__(self, l): self._l = l; self._idx = 0
        def GetAtomicNum(self): return self._l
        def GetIdx(self): return self._idx
    class FakeBond:
        def __init__(self,u,v,t): self._u=u; self._v=v; self._t=t
        def GetBeginAtomIdx(self): return self._u
        def GetEndAtomIdx(self): return self._v
        def GetBondType(self): return Chem.rdchem.BondType.SINGLE
    # Assign random atom types (1-10)
    for node in G.nodes():
        G.nodes[node]['l'] = np.random.randint(1, 11)

    # Just use nx graph directly for speed
    pass

# Faster: compute MI for pure graph types (no labels)
print("  (Computing for unlabeled ER graphs quickly...)", flush=True)

from itertools import combinations as comb

def graph_propagation_mi(n_nodes_list, p=0.3, n_mols=100):
    """Compute MI(k2;k3) for random unlabeled graphs."""
    co = Counter(); k2m = Counter(); k3m = Counter()
    for seed, n in enumerate(n_nodes_list[:n_mols]):
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        nodes = list(G.nodes())
        if len(nodes) < 3: continue

        # k=2: edge or no edge
        for i,j in comb(nodes, 2):
            t2 = int(G.has_edge(i,j))
            k2m[t2] += 1

        # k=3
        combos = list(comb(nodes, 3))
        if len(combos) > 500: combos = combos[:500]
        for a,b,c in combos:
            n_e = int(G.has_edge(a,b)) + int(G.has_edge(a,c)) + int(G.has_edge(b,c))
            k3m[n_e] += 1
            for (u,v) in [(a,b),(a,c),(b,c)]:
                t2 = int(G.has_edge(u,v))
                co[(t2, n_e)] += 1

    # MI
    total_k3 = sum(k3m.values())
    total_k2 = sum(k2m.values())
    H_k3 = -sum((v/total_k3)*np.log(v/total_k3) for v in k3m.values() if v > 0)

    # H(k3|k2)
    H_cond = 0
    for k2_val in k2m:
        p_k2 = k2m[k2_val] / total_k2
        col_sum = sum(co.get((k2_val, k3v), 0) for k3v in k3m)
        if col_sum == 0: continue
        h_col = 0
        for k3_val in k3m:
            c = co.get((k2_val, k3_val), 0)
            if c > 0:
                p = c / col_sum
                h_col -= p * np.log(p)
        H_cond += p_k2 * h_col

    MI = H_k3 - H_cond
    return MI, H_k3, MI/H_k3 if H_k3 > 0 else 0

rand_MI, rand_H, rand_frac = graph_propagation_mi(n_atoms_list[:100], p=0.3)
print(f"  H(k3): {rand_H:.3f}, I(k2;k3): {rand_MI:.3f}, fraction: {rand_frac:.3f} ({rand_frac*100:.1f}%)")

print()
print("=== COMPARISON ===\n")
print(f"{'':30s}  {'H(k3)':6s}  {'I(k2;k3)':8s}  {'MI%':6s}")
print("-" * 60)
print(f"  {'Molecular graphs (ESOL)':28s}  {res['H_k3']:6.3f}  {res['MI_k2_k3']:8.3f}  {res['MI_fraction']*100:5.1f}%")
print(f"  {'Random graphs (ER, p=0.3)':28s}  {rand_H:6.3f}  {rand_MI:8.3f}  {rand_frac*100:5.1f}%")

print()
mi_mol = res['MI_fraction']
if mi_mol > rand_frac + 0.05:
    print(f"CONFIRMED: Molecular graphs have HIGHER MI(k2;k3) ({mi_mol*100:.1f}% vs {rand_frac*100:.1f}%)")
    print("-> k=2 EXPLAINS MORE of k=3 in molecules than in random graphs!")
    print("-> This means molecular graphs have stronger structural correlations across scales")
    print("-> EXPLAINS WHY k=2 is sufficient for molecular property prediction:")
    print("   k=2 already encodes most of the k=3 information in molecular context")
elif mi_mol < rand_frac - 0.05:
    print(f"REVERSED: Random graphs have higher MI ({rand_frac*100:.1f}% vs {mi_mol*100:.1f}%)")
    print("-> k=2 captures LESS of k=3 in molecules than in random graphs")
else:
    print(f"Similar MI: molecules {mi_mol*100:.1f}%, random {rand_frac*100:.1f}%")

print()
print("NEW TOOL: 'CR Propagation Tensor' T_{k->k+1}")
print("  - Measures how structural patterns propagate across scales")
print("  - MI fraction = structural self-consistency measure")
print("  - High MI: structure at scale k determines scale k+1 (low-dimensional)")
print("  - Low MI: scale k and k+1 are structurally independent (high-dimensional)")
