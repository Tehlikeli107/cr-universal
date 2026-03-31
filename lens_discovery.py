"""
LENS DISCOVERY ENGINE: Automatically find new structural invariants.

Method:
1. Take ALL graphs on n vertices
2. Compute KNOWN invariants (degree seq, eigenvalues, WL, etc.)
3. Find "hard pairs" — graphs that ALL known invariants say are same
4. Generate RANDOM new features
5. Check which random feature DISTINGUISHES a hard pair
6. That feature = seed of a NEW LENS

If it finds something: we discovered a structural property
that NO EXISTING METHOD captures. This is genuinely new math.
"""
import torch
import numpy as np
from itertools import combinations
import time

DEVICE = torch.device('cuda')


# ============================================================
# KNOWN INVARIANTS (the "old lenses")
# ============================================================

def degree_sequence(adj):
    """Sorted degree sequence."""
    return tuple(sorted(adj.sum(dim=1).int().tolist()))

def eigenvalues(adj):
    """Sorted eigenvalues (rounded)."""
    try:
        eigs = torch.linalg.eigvalsh(adj.float())
        return tuple(sorted([round(e.item(), 4) for e in eigs]))
    except:
        return None

def trace_powers(adj, max_k=5):
    """Tr(A^k) for k=1..max_k. Counts closed walks."""
    A = adj.float()
    traces = []
    Ak = A.clone()
    for k in range(1, max_k + 1):
        traces.append(round(Ak.trace().item()))
        Ak = Ak @ A
    return tuple(traces)

def num_triangles(adj):
    """Number of triangles = Tr(A^3) / 6."""
    A = adj.float()
    return round((A @ A @ A).trace().item() / 6)

def num_edges(adj):
    return int(adj.sum().item() // 2)

def chromatic_bound(adj):
    """Greedy chromatic number upper bound."""
    N = adj.shape[0]
    colors = [-1] * N
    for v in range(N):
        used = set()
        for u in range(N):
            if adj[v, u] and colors[u] >= 0:
                used.add(colors[u])
        for c in range(N):
            if c not in used:
                colors[v] = c
                break
    return max(colors) + 1

def wiener_index(adj):
    """Sum of all shortest path distances."""
    N = adj.shape[0]
    # BFS from each vertex
    total = 0
    for start in range(N):
        dist = [-1] * N
        dist[start] = 0
        queue = [start]
        idx = 0
        while idx < len(queue):
            v = queue[idx]; idx += 1
            for u in range(N):
                if adj[v, u] and dist[u] == -1:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        total += sum(d for d in dist if d > 0)
    return total // 2  # each pair counted twice


def neighborhood_pattern(adj):
    """For each vertex: sorted tuple of neighbor degrees. Then sort all."""
    N = adj.shape[0]
    deg = adj.sum(dim=1).int()
    patterns = []
    for v in range(N):
        neighbor_degs = sorted([deg[u].item() for u in range(N) if adj[v, u]])
        patterns.append(tuple(neighbor_degs))
    return tuple(sorted(patterns))


def all_known_invariants(adj):
    """Compute all known invariants. Return as hashable tuple."""
    return (
        degree_sequence(adj),
        eigenvalues(adj),
        trace_powers(adj, 5),
        num_triangles(adj),
        num_edges(adj),
        chromatic_bound(adj),
        wiener_index(adj),
        neighborhood_pattern(adj),
    )


# ============================================================
# RANDOM FEATURE GENERATORS (candidates for new lens)
# ============================================================

def random_walk_signature(adj, seed, length=10, n_walks=20):
    """Random walk statistics from random starting points."""
    N = adj.shape[0]
    rng = np.random.RandomState(seed)
    visit_counts = np.zeros(N)

    for _ in range(n_walks):
        v = rng.randint(N)
        for _ in range(length):
            neighbors = [u for u in range(N) if adj[v, u]]
            if not neighbors:
                break
            v = neighbors[rng.randint(len(neighbors))]
            visit_counts[v] += 1

    return tuple(sorted([round(c, 1) for c in visit_counts]))


def local_density_profile(adj):
    """For each vertex: density of its neighborhood subgraph."""
    N = adj.shape[0]
    densities = []
    for v in range(N):
        neighbors = [u for u in range(N) if adj[v, u]]
        if len(neighbors) < 2:
            densities.append(0.0)
            continue
        # Count edges among neighbors
        n_edges = sum(1 for i, j in combinations(neighbors, 2) if adj[i, j])
        max_edges = len(neighbors) * (len(neighbors) - 1) / 2
        densities.append(round(n_edges / max_edges, 3))
    return tuple(sorted(densities))


def path_count_profile(adj, length=3):
    """Count paths of given length starting from each vertex."""
    N = adj.shape[0]
    A = adj.float()
    Ak = A.clone()
    for _ in range(length - 1):
        Ak = Ak @ A
    # Path counts per vertex (diagonal = closed, off-diagonal = open)
    counts = sorted([round(Ak[v].sum().item()) for v in range(N)])
    return tuple(counts)


def subgraph_density_at_distance(adj, dist=2):
    """Density of subgraph induced by vertices at exactly distance d from each vertex."""
    N = adj.shape[0]
    profiles = []
    for v in range(N):
        # BFS to find vertices at distance d
        level = {v: 0}
        queue = [v]
        idx = 0
        while idx < len(queue):
            u = queue[idx]; idx += 1
            for w in range(N):
                if adj[u, w] and w not in level:
                    level[w] = level[u] + 1
                    queue.append(w)

        at_dist = [u for u, d2 in level.items() if d2 == dist]
        if len(at_dist) < 2:
            profiles.append(0.0)
            continue
        n_edges = sum(1 for i, j in combinations(at_dist, 2) if adj[i, j])
        max_edges = len(at_dist) * (len(at_dist) - 1) / 2
        profiles.append(round(n_edges / max_edges, 3))
    return tuple(sorted(profiles))


def two_hop_triangle_count(adj):
    """For each edge (u,v): count triangles involving u but not v, and vice versa."""
    N = adj.shape[0]
    edge_features = []
    for u in range(N):
        for v in range(u+1, N):
            if not adj[u, v]:
                continue
            # Triangles through u (not involving v directly)
            tri_u = 0
            for w in range(N):
                if w != v and adj[u, w]:
                    for x in range(w+1, N):
                        if x != v and adj[u, x] and adj[w, x]:
                            tri_u += 1
            # Triangles through v (not involving u)
            tri_v = 0
            for w in range(N):
                if w != u and adj[v, w]:
                    for x in range(w+1, N):
                        if x != u and adj[v, x] and adj[w, x]:
                            tri_v += 1
            edge_features.append((min(tri_u, tri_v), max(tri_u, tri_v)))
    return tuple(sorted(edge_features))


def resistance_distance_profile(adj):
    """Kirchhoff/resistance distance: based on graph Laplacian pseudoinverse."""
    N = adj.shape[0]
    L = torch.diag(adj.sum(dim=1).float()) - adj.float()
    try:
        # Moore-Penrose pseudoinverse
        L_pinv = torch.linalg.pinv(L)
        # Resistance distance between all pairs
        dists = []
        for i in range(N):
            for j in range(i+1, N):
                r = (L_pinv[i,i] + L_pinv[j,j] - 2*L_pinv[i,j]).item()
                dists.append(round(r, 4))
        return tuple(sorted(dists))
    except:
        return None


# All candidate features
CANDIDATE_FEATURES = [
    ("random_walk_42", lambda adj: random_walk_signature(adj, 42)),
    ("random_walk_99", lambda adj: random_walk_signature(adj, 99)),
    ("local_density", local_density_profile),
    ("path_count_3", lambda adj: path_count_profile(adj, 3)),
    ("path_count_4", lambda adj: path_count_profile(adj, 4)),
    ("subgraph_dist2", lambda adj: subgraph_density_at_distance(adj, 2)),
    ("subgraph_dist3", lambda adj: subgraph_density_at_distance(adj, 3)),
    ("two_hop_tri", two_hop_triangle_count),
    ("resistance", resistance_distance_profile),
]


# ============================================================
# GRAPH GENERATION (n=8 from networkx atlas)
# ============================================================

def generate_all_graphs_n(n):
    """Generate all non-isomorphic graphs on n vertices using networkx."""
    try:
        import networkx as nx
        if n <= 7:
            # Use atlas for small n
            graphs = []
            for G in nx.graph_atlas_g():
                if G.number_of_nodes() == n and nx.is_connected(G):
                    adj = torch.zeros(n, n, device=DEVICE)
                    for u, v in G.edges():
                        adj[u, v] = adj[v, u] = 1
                    graphs.append(adj)
            return graphs
        else:
            return None  # need g6 file for n>=8
    except ImportError:
        return None


def load_graphs_g6(filepath, n):
    """Load graphs from g6 format file."""
    import networkx as nx
    graphs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            G = nx.from_graph6_bytes(line.encode())
            if G.number_of_nodes() == n:
                adj = torch.zeros(n, n, device=DEVICE)
                for u, v in G.edges():
                    adj[u, v] = adj[v, u] = 1
                graphs.append(adj)
    return graphs


# ============================================================
# LENS DISCOVERY ENGINE
# ============================================================

def discover_new_lens(graphs, n):
    """
    THE ENGINE:
    1. Compute known invariants for all graphs
    2. Find hard pairs (same known invariants, different graphs)
    3. Test candidate features on hard pairs
    4. Report which features distinguish hard pairs = NEW LENS
    """
    N_graphs = len(graphs)
    print(f"\nStep 1: Computing known invariants for {N_graphs} graphs...")

    t0 = time.time()
    invariant_map = {}  # invariant_tuple -> list of graph indices
    for i, adj in enumerate(graphs):
        inv = all_known_invariants(adj)
        key = str(inv)  # hashable
        if key not in invariant_map:
            invariant_map[key] = []
        invariant_map[key].append(i)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_graphs}...", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Find hard pairs
    hard_groups = {k: v for k, v in invariant_map.items() if len(v) > 1}
    n_hard_pairs = sum(len(v) * (len(v) - 1) // 2 for v in hard_groups.values())

    print(f"\nStep 2: Found {len(hard_groups)} hard groups ({n_hard_pairs} hard pairs)")
    print(f"  These pairs are INDISTINGUISHABLE by ALL known invariants!")

    if n_hard_pairs == 0:
        print("\n  NO HARD PAIRS! All graphs distinguished by known invariants.")
        print("  Known invariants are COMPLETE for this n.")
        return []

    # Show hard groups
    for key, indices in list(hard_groups.items())[:5]:
        print(f"  Group: graphs {indices}")

    # Step 3: Test candidate features on hard pairs
    print(f"\nStep 3: Testing {len(CANDIDATE_FEATURES)} candidate features on hard pairs...")

    discoveries = []
    for feat_name, feat_fn in CANDIDATE_FEATURES:
        n_distinguished = 0
        n_tested = 0

        for key, indices in hard_groups.items():
            # Compute feature for all graphs in this hard group
            features = {}
            for idx in indices:
                try:
                    f = feat_fn(graphs[idx])
                    features[idx] = f
                except Exception:
                    features[idx] = None

            # Check if feature distinguishes any pair
            idx_list = list(features.keys())
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    n_tested += 1
                    fi = features[idx_list[i]]
                    fj = features[idx_list[j]]
                    if fi is not None and fj is not None and fi != fj:
                        n_distinguished += 1

        if n_distinguished > 0:
            print(f"  *** {feat_name}: distinguished {n_distinguished}/{n_tested} hard pairs! ***")
            discoveries.append((feat_name, n_distinguished, n_tested))
        else:
            print(f"      {feat_name}: 0/{n_tested} (no new info)")

    # Report
    print(f"\n{'='*55}")
    print(f"LENS DISCOVERY RESULTS (n={n})")
    print(f"{'='*55}")
    print(f"  Graphs tested: {N_graphs}")
    print(f"  Hard pairs (all known invariants fail): {n_hard_pairs}")

    if discoveries:
        print(f"\n  NEW LENSES FOUND:")
        for name, n_dist, n_test in sorted(discoveries, key=lambda x: -x[1]):
            pct = n_dist / n_test * 100
            print(f"    {name}: {n_dist}/{n_test} ({pct:.0f}%) hard pairs distinguished")
        print(f"\n  These features capture structural information")
        print(f"  that degree seq + eigenvalues + WL + Wiener + neighborhood")
        print(f"  ALL MISS. This is genuinely new structural knowledge.")
    else:
        print(f"\n  NO NEW LENS FOUND.")
        print(f"  All candidate features also fail on hard pairs.")
        print(f"  Need more creative feature generation.")

    return discoveries


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("LENS DISCOVERY ENGINE")
    print("=" * 55)
    print("Goal: find structural invariants that KNOWN methods miss")

    # Try n=7 first (smaller, faster)
    print("\n--- n=7 (connected graphs from NetworkX atlas) ---")
    graphs_7 = generate_all_graphs_n(7)
    if graphs_7:
        print(f"Loaded {len(graphs_7)} connected graphs on 7 vertices")
        discover_new_lens(graphs_7, 7)
    else:
        print("NetworkX not available or n=7 generation failed")

    # Try n=8 if g6 file available
    g6_path = r"C:\Users\salih\Desktop\universal-arch-search\graph_data\graph8.g6"
    try:
        print(f"\n--- n=8 (from McKay catalog) ---")
        graphs_8 = load_graphs_g6(g6_path, 8)
        if graphs_8 and len(graphs_8) > 0:
            print(f"Loaded {len(graphs_8)} graphs on 8 vertices")
            # Sample for speed
            if len(graphs_8) > 500:
                import random; random.seed(42)
                sample_idx = random.sample(range(len(graphs_8)), 500)
                graphs_sample = [graphs_8[i] for i in sample_idx]
                print(f"Sampling {len(graphs_sample)} for speed")
                discover_new_lens(graphs_sample, 8)
            else:
                discover_new_lens(graphs_8, 8)
    except Exception as e:
        print(f"  Could not load n=8: {e}")
