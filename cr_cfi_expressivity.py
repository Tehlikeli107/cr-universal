"""
CR Expressivity vs WL Hierarchy: CFI Graph Family

BACKGROUND:
  - WL test (1-WL) = hash-based graph isomorphism test
  - k-WL = k-dimensional WL, equivalent to k-variable first-order logic
  - CFI(Km) construction: for each m, creates pairs of non-isomorphic graphs
    that FOOL k-WL for ALL k < m/2 (approximately)
  - k-WL equivalent to counting homomorphisms from treewidth<k graphs

  BUT: from our prior work, CFI(K3) pair (12 nodes, 6-regular) is
  distinguished by CR k=4 induced subgraph counting!

QUESTION: For CFI(Km), what is k_CR(m) = min k for CR to distinguish?

CONJECTURE: k_CR(m) = constant (independent of m!)
  - k_WL(m) grows with m -> CR is STRICTLY stronger than all finite k-WL

If true: CR is NOT equivalent to any k-WL. This is a major GNN expressivity result.
  -> CR-based GNNs are more expressive than any k-MPNN architecture

We test:
  - CFI(K3): 12 nodes, 6-regular (known: k-WL fails for k<=3, CR k=4 succeeds)
  - CFI(K4): 16 nodes, 8-regular (k-WL fails for k<=4, CR k=?)
  - CFI(K5): 20 nodes, 10-regular (k-WL fails for k<=5, CR k=?)
  - Pattern: If k_CR stays at 4 for all m, that's the separationresult.
"""
import numpy as np
from collections import Counter
from itertools import combinations, permutations
import networkx as nx

def build_cfi_k3():
    """
    CFI(K3): Cai-Furer-Immerman graph based on K3.
    K3 has 3 edges: {0-1, 0-2, 1-2}
    CFI construction: for each vertex v in K3, create a gadget.
    For K3, the result has 12 nodes.

    Standard construction: vertex gadgets connect via edge gadgets.
    Here we use the direct construction.
    """
    # CFI(K3) construction:
    # For K3 with vertices {a,b,c} and edges {e12=(a,b), e13=(a,c), e23=(b,c)}
    # Each vertex v with degree d gets a gadget of 2^(d-1) vertices
    # For K3: each vertex has degree 2, so each gadget has 2 vertices -> 6 vertices per "side"
    # But the standard CFI pair is over 12 nodes

    # We use the concrete construction from the literature:
    # Vertices: for each edge e=(u,v) in K3, create 2 vertices: e_uv and e_vu
    # Plus for each vertex v: one "center" vertex
    # Total: 2*|E| + |V| = 2*3 + 3 = 9? No that's not right.

    # Let me use the construction directly by NetworkX or explicitly
    # The 12-node CFI(K3) I previously verified in the memory

    # Use explicit adjacency from prior verified work
    # CFI(K3): 12 vertices, 6-regular
    # G0 and G1 are non-isomorphic but k-WL equivalent for all k

    # Reimplement from scratch following Arvind et al.:
    # For K3 = ({0,1,2}, {01,02,12}):
    # For each vertex v in K3 with incident edges E_v:
    #   Create 2^|E_v|-1 = 2^2-1 = 3 binary vectors of length |E_v|=2
    #   But this is getting complex. Let me use a known explicit construction.

    # From the memory: CFI(K3) pair verified at n=12 with 4-subgraph distributions
    # Let me just build it from the known edge list

    # Using the construction from Grohe's book:
    # K3 vertices: 0,1,2
    # For each vertex v, create 4 "split" vertices (one for each parity of incident edges)
    # Vertex 0 has edges to 1,2: splits 00a,00b,01a,01b (incident = b values at positions)
    # This gets to 4*3=12 vertices

    # Simpler: use the known explicit edge lists from prior work
    # (My prior work showed CFI(K3) has 12 vertices, 6-regular, WL fails k=1..4)

    # Let me use a different approach: generate from the base K3 directly

    n_base = 3  # K3
    base_edges = [(0,1), (0,2), (1,2)]

    # CFI construction for m-regular base graph:
    # Each vertex v in base gets a gadget of size 2^(deg-1)
    # For K3: deg=2, gadget size = 2
    # Total: 3*2 = 6 nodes (each half of CFI pair)
    # Total with two "copies": we create two variants differing by one twist

    # Standard CFI for K3 (I'll use the 12-node explicit version verified before)
    # Building from first principles:

    # Phase 1 version (standard twist):
    # For each edge (u,v): create 4 nodes: (u,v,0), (u,v,1), (v,u,0), (v,u,1)
    # For each vertex v: for each parity p in {0,1}: create node (v, p)
    # Edges within edge gadgets and between gadgets...

    # This is getting complex. Let me just hardcode the known 12-node CFI(K3).
    # From memory: CFI(K3) is a 12-node, 6-regular pair.

    # I'll use the clean mathematical construction directly:
    # Nodes: {(i, S) : i in K3, S subset of edges incident to i with |S| even}
    # Edges: connect (i,S) -- (j,T) if (i,j) in E and S XOR (i,j) = some condition

    # For K3, vertex i has 2 incident edges, so |S| even means S={} or S={both edges}
    # -> 2 nodes per vertex = 6 nodes total
    # Two copies: G_0 and G_1 (twisted version)

    # K3 CFI (6-node version, G_0):
    # Vertex 0 has incident edges (0,1) and (0,2)
    # Node (0, {}) = 0, Node (0, {01,02}) = 1
    # Vertex 1 has incident edges (0,1) and (1,2)
    # Node (1, {}) = 2, Node (1, {01,12}) = 3
    # Vertex 2 has incident edges (0,2) and (1,2)
    # Node (2, {}) = 4, Node (2, {02,12}) = 5

    # Wait, I need to think about which edges exist between these nodes.
    # The CFI construction connects (i,S) -- (j,T) if (i,j) in E_K3 and
    # S symmetric_difference {edge (i,j) in S} = same parity condition

    # Let me just use a direct adjacency matrix for the 6-node pair

    # G_0 (no twist):
    # (0,0)--(1,0), (0,0)--(2,0)
    # (0,1)--(1,1), (0,1)--(2,1)
    # (1,0)--(2,0), (1,1)--(2,1)... no this doesn't give 6-regular

    # I'll use the 12-node version explicitly from Wikipedia/literature
    # K3 has 3 vertices, each with degree 2
    # In CFI(K3), each vertex expands to 4 nodes (2^(deg) = 4), giving 12 total

    # Let me implement a generic CFI builder
    return build_cfi_from_base_graph(n_base, base_edges)

def build_cfi_from_base_graph(n_base, base_edges):
    """
    Build CFI graph pair from a base graph.
    Returns (G0, G1) as networkx graphs.

    Construction (Cai-Furer-Immerman 1992):
    For base graph H with vertices V and edges E:
    - For each vertex v in V: create 2^(deg(v)-1) vertices
    - The 'gadget' at v is a complete bipartite graph K_{2^(deg-1), 2^(deg-1)}
    - Edges between gadgets encode the base edges

    For K_m (complete graph on m vertices):
    - Each vertex has degree m-1
    - Gadget size: 2^(m-2) vertices per parity class -> 2^(m-1) total per vertex
    """
    from itertools import product

    # Compute degree for each vertex
    deg = [0] * n_base
    for u, v in base_edges:
        deg[u] += 1; deg[v] += 1

    # For each vertex v, create 2^(deg[v]) bit vectors
    # Split into even and odd parity classes
    # "Even" = even number of 1s in the bits
    # These form two parity classes

    # node id = (vertex, bit_tuple)
    # Total nodes per vertex = 2^(deg[v])

    # For a vertex v with incident edges e_1, e_2, ..., e_d:
    # A node (v, S) where S is a subset of incident edges
    # Two nodes (v, S) and (v, S') are in the SAME parity class iff |S|=|S'| (mod 2)
    # CFI gadget at v: complete bipartite between even and odd parity subsets

    # Build node index mapping
    node_ids = {}
    node_counter = 0

    for v in range(n_base):
        d = deg[v]
        # All subsets of incident edges
        for bits in range(2**d):
            node_ids[(v, bits)] = node_counter
            node_counter += 1

    n_total = node_counter
    adj_G0 = set()
    adj_G1 = set()

    # Get incident edges for each vertex (as ordered list)
    incident = [[] for _ in range(n_base)]
    for e_idx, (u, v) in enumerate(base_edges):
        incident[u].append(e_idx)
        incident[v].append(e_idx)

    # Gadget edges at each vertex v:
    # Connect (v, S) -- (v, S') iff S and S' differ in exactly one incident edge bit
    # AND the parity of |S| != parity of |S'| (cross parity)
    # This makes a bipartite graph between even/odd parity subsets
    for v in range(n_base):
        d = deg[v]
        for s1 in range(2**d):
            for s2 in range(2**d):
                if s1 == s2: continue
                # Same parity -> no edge in gadget
                if bin(s1).count('1') % 2 == bin(s2).count('1') % 2: continue
                # Different parity -> edge if they differ in exactly 1 bit
                if bin(s1 ^ s2).count('1') == 1:
                    u_id = node_ids[(v, s1)]
                    v_id = node_ids[(v, s2)]
                    if u_id < v_id:
                        adj_G0.add((u_id, v_id))
                        adj_G1.add((u_id, v_id))

    # Base edge gadgets: for each edge (u,v) in H:
    # Find the position of (u,v) in incident[u] and in incident[v]
    # For each (u, S_u) and (v, S_v):
    #   Connect them in G0 iff S_u has the same bit for (u,v) as S_v has for (v,u)
    #   G1: twist one specific gadget (flip the connection for one edge)

    for e_idx, (u, v) in enumerate(base_edges):
        pos_u = incident[u].index(e_idx)
        pos_v = incident[v].index(e_idx)

        du, dv = deg[u], deg[v]

        for s_u in range(2**du):
            for s_v in range(2**dv):
                # bit of edge e_idx in s_u
                bit_u = (s_u >> pos_u) & 1
                bit_v = (s_v >> pos_v) & 1

                u_id = node_ids[(u, s_u)]
                v_id = node_ids[(v, s_v)]
                key = (min(u_id, v_id), max(u_id, v_id))

                # G0: connect if bit_u == bit_v (same parity match)
                if bit_u == bit_v:
                    adj_G0.add(key)

                # G1: twist the first edge
                if e_idx == 0:
                    # Flip: connect if bit_u != bit_v
                    if bit_u != bit_v:
                        adj_G1.add(key)
                else:
                    if bit_u == bit_v:
                        adj_G1.add(key)

    G0 = nx.Graph(); G0.add_nodes_from(range(n_total))
    G1 = nx.Graph(); G1.add_nodes_from(range(n_total))
    G0.add_edges_from(adj_G0)
    G1.add_edges_from(adj_G1)

    return G0, G1

def canonical_sub(nodes, nlist, G):
    """Canonical form of induced k-subgraph in G (unlabeled)."""
    n = len(nlist); nl = sorted(nlist)
    best = None
    for perm in permutations(range(n)):
        mat = tuple(tuple(1 if i!=j and G.has_edge(nl[perm[i]], nl[perm[j]]) else 0 for j in range(n)) for i in range(n))
        if best is None or mat < best: best = mat
    return best

def cr_fingerprint(G, k, max_combos=10000):
    """CR k-subgraph fingerprint."""
    nodes = list(G.nodes())
    c = Counter()
    combos = list(combinations(nodes, k))
    if len(combos) > max_combos:
        np.random.shuffle(combos)
        combos = combos[:max_combos]
    for sub in combos:
        canon = canonical_sub(None, list(sub), G)
        c[canon] += 1
    return frozenset(c.items())

# Build CFI pairs for K3, K4, K5
print("=== CR Expressivity: CFI Graph Family ===\n")
print("Testing: What is k_CR(m) for CFI(Km)?")
print("If k_CR(m) = constant -> CR strictly stronger than ALL finite k-WL\n")

for m in [3, 4, 5]:
    print(f"CFI(K{m}):", flush=True)
    n_base = m
    # K_m: complete graph on m vertices
    base_edges = [(i,j) for i in range(m) for j in range(i+1, m)]

    G0, G1 = build_cfi_from_base_graph(n_base, base_edges)
    n_nodes = G0.number_of_nodes()

    print(f"  n={n_nodes} nodes, deg: {[d for n,d in G0.degree()[:3]]}...", flush=True)

    # Verify non-isomorphic
    is_iso = nx.is_isomorphic(G0, G1)
    print(f"  G0 ~= G1: {is_iso} (should be False)", flush=True)

    # Test k-WL failure (simplified: check if degree sequences match)
    deg0 = sorted(dict(G0.degree()).values())
    deg1 = sorted(dict(G1.degree()).values())
    print(f"  Degree seqs equal: {deg0 == deg1}", flush=True)

    # Test CR fingerprints for k=2,3,4,5,6
    found_k = None
    for k in range(2, min(7, n_nodes)):
        if n_nodes > 20 and k > 5:
            print(f"  k={k}: skipping (too large)")
            continue
        max_c = 5000 if n_nodes > 15 else None
        fp0 = cr_fingerprint(G0, k, max_combos=max_c or 100000)
        fp1 = cr_fingerprint(G1, k, max_combos=max_c or 100000)
        same = (fp0 == fp1)
        print(f"  k={k}: CR fps {'SAME' if same else 'DIFFERENT'}", flush=True)
        if not same and found_k is None:
            found_k = k
            print(f"  -> k_CR(K{m}) = {k} !!", flush=True)
            break

    if found_k is None:
        print(f"  k_CR(K{m}) > tested range")
    print()

print("=== THEORETICAL IMPLICATIONS ===")
print()
print("If k_CR(m) = constant C for all m:")
print("  CR is STRICTLY stronger than k-WL for all finite k")
print("  -> CR-based GNNs are more expressive than any k-MPNN")
print("  -> Opens new GNN expressivity class beyond WL hierarchy")
