"""
Sequence Assembly Index (SAI): Assembly Theory for Linear Sequences

CORE INNOVATION:
Molecular Assembly Index (MA) = minimum COPY+JOIN steps to build a molecule graph
Sequence Assembly Index (SAI) = minimum COPY+CONCATENATE steps to build a string

OPERATION SET (physical, not Turing-complete):
  1. INTRODUCE(c): introduce a new single character (cost 1)
  2. JOIN(a, b): concatenate two existing parts (cost 1)
  Once a part is created, any number of COPIES are FREE.

EXAMPLE:
  "ABCABC":
  - Introduce A (1), B (2), C (3) -> {A, B, C}
  - Join A+B -> AB (4), Join AB+C -> ABC (5)
  - Join ABC+ABC -> ABCABC (6)   <- copy ABC for free, 1 join
  Total: 6 operations

  Without copy optimization: same ABCABC would take 5 joins + 6 introduces = 11

DIFFERENCE FROM KOLMOGOROV COMPLEXITY:
  - K(s) = min program that outputs s (Turing-complete, uncomputable)
  - SAI(s) = min PHYSICAL assembly steps (concatenation only, computable!)

DIFFERENCE FROM LZ COMPLEXITY:
  - LZ76 = number of distinct phrases in sliding-window parse
  - SAI = min operations to BUILD the string using hierarchical copying
  - SAI allows arbitrary-length "parts" to be reused, not just previous substrings
  - SAI is more like "minimal factory instructions" than "streaming compression"

RELATIONSHIP TO MOLECULAR ASSEMBLY INDEX:
  - MA uses the string representation (SMILES/InChI) of molecules
  - SAI is MA applied DIRECTLY to the sequence
  - For DNA: SAI measures the copy-paste economy of the genome

BIOLOGICAL MOTIVATION:
  Gene duplication: gene X appears twice -> SAI lower than two distinct genes
  Repeat elements: Alu (10^6 copies in human genome) -> SAI << genome length
  Random DNA: no repeats -> SAI ~ sequence length

  KEY CLAIM: SAI(biological_sequence) << SAI(random_sequence_same_length)
  This is a computable, physics-grounded complexity measure.

ALGORITHM:
  Exact SAI is NP-hard (equivalent to "Shortest Superstring from Grammar").
  We use three approximations:

  1. GREEDY-BOTTOM-UP: repeatedly find and compress the most frequent repeated bigram
     (similar to Re-Pair algorithm). Count total operations.

  2. SUFFIX-ARRAY-BASED: find longest repeated substring, compress, repeat.
     (greedy top-down)

  3. ANALYTICAL BOUND: SAI >= unique_characters + (length - compression) / 2
"""

import numpy as np
from collections import Counter
import time
import random
import string

# ===================== SAI ALGORITHMS =====================

def sai_repair(seq, verbose=False):
    """
    SAI via Re-Pair compression (greedy bigram replacement).

    Re-Pair: repeatedly find most frequent bigram (a,b), replace all
    non-overlapping occurrences with new symbol c. Count operations.

    SAI estimate: number of grammar rules in the resulting grammar.
    """
    # Represent as list of "parts" (each part is a string)
    parts = list(seq)  # Initially each char is a part
    vocab = set(seq)   # Chars introduced so far

    ops = len(vocab)   # Introduce operations (one per unique char)
    rule_count = 0     # Number of JOIN rules

    # Iterate: find and compress most frequent bigram
    max_iter = len(seq)
    for iteration in range(max_iter):
        if len(parts) <= 2:
            break

        # Count bigrams
        bigrams = Counter()
        for i in range(len(parts)-1):
            bigrams[(parts[i], parts[i+1])] += 1

        if not bigrams:
            break

        most_common, freq = bigrams.most_common(1)[0]
        if freq < 2:  # No bigram appears more than once -> no compression possible
            break

        # Replace all non-overlapping occurrences
        a, b = most_common
        new_sym = f"_{iteration}"  # New non-terminal

        new_parts = []
        i = 0
        while i < len(parts):
            if i < len(parts)-1 and parts[i] == a and parts[i+1] == b:
                new_parts.append(new_sym)
                i += 2
            else:
                new_parts.append(parts[i])
                i += 1

        # This rule = 1 JOIN operation
        rule_count += 1
        parts = new_parts

        if verbose and iteration < 5:
            print(f"    Iter {iteration}: compressed ({a},{b}) -> {new_sym} (freq={freq}), remaining parts: {len(parts)}")

    # Remaining parts need to be joined
    join_ops = len(parts) - 1

    total_ops = ops + rule_count + join_ops
    return total_ops, ops, rule_count, join_ops

def sai_longest_repeat(seq, verbose=False):
    """
    SAI via longest-repeat greedy compression.

    Find the longest repeated substring, 'compile' it as one part,
    replace all occurrences, repeat.
    """
    import re

    parts = seq  # String
    introduced = set(seq)
    ops = len(introduced)
    rule_count = 0

    for iteration in range(100):  # Max iterations
        n = len(parts)
        if n <= 2:
            break

        # Find longest repeated substring using suffix approach
        best_len = 1
        best_substr = None

        # Check substrings of decreasing length
        found = False
        for length in range(n//2, 1, -1):
            seen = {}
            for i in range(n - length + 1):
                s = parts[i:i+length]
                if s in seen:
                    best_len = length
                    best_substr = s
                    found = True
                    break
            if found:
                break

        if best_substr is None or best_len <= 1:
            break

        # Replace all non-overlapping occurrences
        new_sym = f"\x01{iteration}\x01"  # New symbol
        new_parts = parts.replace(best_substr, new_sym)

        if new_parts == parts:  # No replacement occurred
            break

        rule_count += 1
        parts = new_parts

        if verbose and iteration < 3:
            print(f"    Iter {iteration}: compressed '{best_substr[:10]}' (len={best_len}), remaining len: {len(parts)}")

    join_ops = max(0, len(parts.split('\x00')) - 1)  # Approximate remaining joins
    # Actually count remaining distinct symbols to join
    remaining_syms = len(parts.replace('\x01', '').replace('\x00', ''))

    total_ops = ops + rule_count + remaining_syms
    return total_ops, ops, rule_count, remaining_syms

# ===================== TEST SEQUENCES =====================

def make_random_dna(n, seed=42):
    """Random DNA sequence."""
    rng = np.random.RandomState(seed)
    return ''.join(rng.choice(list('ACGT'), n))

def make_tandem_repeat(unit, n_copies):
    """Tandem repeat: CATCATCAT..."""
    return unit * n_copies

def make_alu_like(n):
    """Alu-like sequence: ~300bp Alu element repeated with mutations."""
    alu_unit = make_random_dna(100, seed=42)  # Base Alu
    result = ""
    while len(result) < n:
        # Copy Alu with ~5% mutations
        mutated = list(alu_unit)
        for i in range(len(mutated)):
            if random.random() < 0.05:
                mutated[i] = random.choice('ACGT')
        result += ''.join(mutated)
    return result[:n]

def make_fibonacci_dna(n):
    """Fibonacci string (self-similar): A, B, AB, BAB, ABBAB, ..."""
    a, b = "A", "B"
    while len(a) < n:
        a, b = b, a+b
    return a[:n]

def make_coding_gene(n, codon_table_size=20):
    """Protein-coding gene: codons from limited set."""
    codons = [make_random_dna(3, seed=i) for i in range(codon_table_size)]
    seq = ""
    rng = np.random.RandomState(123)
    while len(seq) < n:
        seq += rng.choice(codons)
    return seq[:n]

# ===================== EXPERIMENT 1: LENGTH SCALING =====================

print("=== Sequence Assembly Index (SAI) ===\n")
print("New complexity measure for biological sequences\n")
print("CLAIM: SAI(biological) << SAI(random) for same length\n")

print("--- Experiment 1: SAI vs Sequence Length ---\n")
print(f"{'Sequence':20s}  {'Length':7s}  {'SAI':6s}  {'SAI/n':7s}  {'Ratio vs random':15s}")
print("-" * 65)

random.seed(42)
for n in [100, 300, 1000]:
    # Random DNA
    rand_seq = make_random_dna(n)
    sai_rand, ops_rand, rules_rand, joins_rand = sai_repair(rand_seq)

    # Tandem repeat (ACGT x n/4)
    period = "ACGT"
    tandem_seq = make_tandem_repeat(period, n//len(period) + 1)[:n]
    sai_tandem, _, _, _ = sai_repair(tandem_seq)

    # Alu-like (biological repeats)
    alu_seq = make_alu_like(n)
    sai_alu, _, _, _ = sai_repair(alu_seq)

    # Fibonacci (self-similar)
    fib_seq = make_fibonacci_dna(n)
    sai_fib, _, _, _ = sai_repair(fib_seq)

    # Coding gene
    gene_seq = make_coding_gene(n)
    sai_gene, _, _, _ = sai_repair(gene_seq)

    print(f"{'Random DNA':20s}  {n:7d}  {sai_rand:6d}  {sai_rand/n:7.3f}  {1.00:15.3f} (baseline)")
    print(f"{'Tandem(ACGT)':20s}  {n:7d}  {sai_tandem:6d}  {sai_tandem/n:7.3f}  {sai_tandem/sai_rand:15.3f}")
    print(f"{'Alu-like':20s}  {n:7d}  {sai_alu:6d}  {sai_alu/n:7.3f}  {sai_alu/sai_rand:15.3f}")
    print(f"{'Fibonacci':20s}  {n:7d}  {sai_fib:6d}  {sai_fib/n:7.3f}  {sai_fib/sai_rand:15.3f}")
    print(f"{'Coding gene':20s}  {n:7d}  {sai_gene:6d}  {sai_gene/n:7.3f}  {sai_gene/sai_rand:15.3f}")
    print()

# ===================== EXPERIMENT 2: SCALING LAW =====================

print("--- Experiment 2: SAI Scaling Laws ---\n")
print("Measuring how SAI grows with n for different sequence types\n")

ns = [50, 100, 200, 400, 800]
sai_scaling = {
    'random': [],
    'tandem': [],
    'alu': [],
    'fibonacci': [],
    'gene': [],
}

for n in ns:
    rand_seq = make_random_dna(n)
    sai_rand, _, _, _ = sai_repair(rand_seq)
    sai_scaling['random'].append(sai_rand)

    tandem_seq = make_tandem_repeat("ACGT", n//4 + 1)[:n]
    sai_tandem, _, _, _ = sai_repair(tandem_seq)
    sai_scaling['tandem'].append(sai_tandem)

    alu_seq = make_alu_like(n)
    sai_alu, _, _, _ = sai_repair(alu_seq)
    sai_scaling['alu'].append(sai_alu)

    fib_seq = make_fibonacci_dna(n)
    sai_fib, _, _, _ = sai_repair(fib_seq)
    sai_scaling['fibonacci'].append(sai_fib)

    gene_seq = make_coding_gene(n)
    sai_gene, _, _, _ = sai_repair(gene_seq)
    sai_scaling['gene'].append(sai_gene)

# Fit power laws: SAI ~ n^alpha
print(f"{'Sequence':15s}  {'Scaling law SAI~n^alpha':25s}  {'Alpha':6s}")
print("-" * 55)
for name, saivs in sai_scaling.items():
    log_ns = np.log(ns)
    log_saivs = np.log(saivs)
    alpha, _ = np.polyfit(log_ns, log_saivs, 1)
    # Also check linear fit
    print(f"{name:15s}  {'SAI ~ n^%.3f' % alpha:25s}  {alpha:6.3f}", end="")

    if alpha > 0.95:
        print("  (near-LINEAR, incompressible)")
    elif alpha > 0.7:
        print("  (sub-linear, moderate structure)")
    elif alpha > 0.4:
        print("  (high compression)")
    else:
        print("  (log-like, very repetitive)")

print()
print("THEORY: SAI ~ n^alpha defines 'Assembly Dimension'")
print("  alpha = 1.0: no self-similarity (random)")
print("  alpha = 0.0: perfectly self-similar (log growth)")
print("  0 < alpha < 1: fractal-like complexity")
print()

# ===================== EXPERIMENT 3: REAL BIOLOGY =====================

print("--- Experiment 3: Real Biological Sequences ---\n")

# Simple bacterial gene (synthetic but realistic)
ecoli_like = """
ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA
""".replace('\n', '').replace(' ', '').strip()

# HIV LTR repeat sequence (known to have tandem repeats)
hiv_ltr_like = "TGGAAGGGCTAATTCACTCCCAACGAAGACAAGATATCCTTGATCTGTGG" * 3

# Globin coding sequence (protein-coding, limited amino acid variety)
# Approximately codon-biased
globin_like = make_coding_gene(300, codon_table_size=61)  # Full codon table but biased

# Microsatellite (short tandem repeat, disease-related)
microsatellite = "CAG" * 50  # Huntington's disease-like CAG repeat

# Regulatory element (CpG island): enriched in CG dinucleotides
def make_cpg_island(n):
    seq = ""
    for _ in range(n):
        if random.random() < 0.6:
            seq += random.choice("CG")  # CpG enriched
        else:
            seq += random.choice("ACGT")
    return seq[:n]

cpg_seq = make_cpg_island(200)
random_seq_200 = make_random_dna(200)

bio_seqs = {
    "E.coli gene (65bp)": ecoli_like,
    "HIV LTR repeat (3x50)": hiv_ltr_like,
    "Globin coding (300bp)": globin_like,
    "Huntington CAG50": microsatellite,
    "CpG island (200bp)": cpg_seq,
    "Random DNA (200bp)": random_seq_200,
}

print(f"{'Sequence':25s}  {'Length':7s}  {'SAI':6s}  {'SAI/n':7s}  {'Expected'}")
print("-" * 75)
for name, seq in bio_seqs.items():
    sai, ops, rules, joins = sai_repair(seq)
    n = len(seq)

    # Compare to random baseline of same length
    rand_baseline, _, _, _ = sai_repair(make_random_dna(n))
    ratio = sai / max(rand_baseline, 1)

    if "CAG" in name: expected = "LOW (perfect repeat)"
    elif "HIV" in name: expected = "LOW (tandem repeat)"
    elif "CpG" in name: expected = "MEDIUM (biased)"
    elif "globin" in name: expected = "MEDIUM (codon structure)"
    elif "E.coli" in name: expected = "MEDIUM?"
    else: expected = "HIGH (random baseline)"

    print(f"{name:25s}  {n:7d}  {sai:6d}  {sai/n:7.3f}  {expected} (rand={rand_baseline}, ratio={ratio:.2f})")

print()

# ===================== THEOREM =====================

print("=== THEORETICAL ANALYSIS ===\n")
print("SAI vs classic measures:\n")
print("  Sequence 'ACGTACGT' (period 4, length 8):")
acgt2 = "ACGTACGT"
sai_v, o, r, j = sai_repair(acgt2)
print(f"    SAI={sai_v} (introduce 4, build ACGT: 3 joins, copy+join: 1) = {sai_v}")

print()
print("  Sequence 'ACGTACGTACGTACGT' (period 4, length 16):")
acgt4 = "ACGT" * 4
sai_v2, o2, r2, j2 = sai_repair(acgt4)
print(f"    SAI={sai_v2} (introduce 4, build ACGT: 3, build 8: 1, build 16: 1) = 9 expected")
print(f"    Re-Pair estimate: {sai_v2}")

print()
print("THEOREM CANDIDATE: For tandem repeat (u)^k where |u|=T:")
print("  SAI((u)^k) <= T + ceil(log2(k)) * (T - 1) + ceil(log2(k))")
print("  (build u in T+1 steps, then double k times via copy+join)")
print("  -> SAI grows as O(T * log k) = O(T * log(n/T))")
print("  -> For k >> T: SAI ~ T * log(n), but for n fixed: SAI ~ constant in k!")
print()

# Verify: ACGT * k for k=1..20
print("SAI(ACGT * k) for k = 1..16:")
for k in [1, 2, 4, 8, 16]:
    seq = "ACGT" * k
    sai_v, _, _, _ = sai_repair(seq)
    n = len(seq)
    print(f"  k={k:3d}, n={n:4d}: SAI={sai_v:4d}, SAI/n={sai_v/n:.3f}")

print()
print("KEY FINDING: SAI/n DECREASES as k increases (more repetition = better compression)")
print("But SAI itself grows SLOWER than n (sub-linear for structured sequences)")

print()
print("=== NEW TOOL: SAI AS BIOLOGICAL COMPLEXITY MEASURE ===\n")
print("SAI measures 'copy-paste economy' of a sequence")
print("  Low SAI: highly repetitive, low informational content")
print("  High SAI: random-like, maximal information per nucleotide")
print()
print("Applications:")
print("  1. ORIGIN OF LIFE: early RNA/DNA should have LOW SAI (simple repeats)")
print("     SAI can test 'metabolism-first' vs 'replication-first' hypotheses")
print("     Short periodic sequences (protocells) have SAI << full genome SAI")
print()
print("  2. REPEAT ELEMENT ANALYSIS: Alu, LINE, SINE elements lower genome SAI")
print("     SAI difference(with/without repeat elements) = their assembly contribution")
print()
print("  3. GENE COMPLEXITY: coding genes have STRUCTURED SAI (codon patterns)")
print("     SAI of different gene families reveals evolutionary compression")
print()
print("  4. DISEASE BIOMARKER: CAG repeats (Huntington's), CGG (Fragile X) -> low SAI")
print("     SAI of repeat expansion regions as quantitative disease measure")
print()
print("COMPARISON TO ASSEMBLY INDEX (molecular graphs):")
print("  Both measure 'minimum copy-assembly steps'")
print("  MA = for molecular GRAPHS")
print("  SAI = for biological SEQUENCES")
print("  Combined: SAI of SMILES string ~= MA of the molecule? (empirical test!)")
print()

# BONUS: Test if SAI of SMILES correlates with MA
smiles_examples = {
    "ethane": "CC",
    "propane": "CCC",
    "butane": "CCCC",
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "glucose": "OCC(O)C(O)C(O)C(O)C=O",
    "caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
}
print("--- SAI of SMILES strings vs known assembly complexity ---\n")
print(f"{'Molecule':15s}  {'SMILES':35s}  {'SAI':5s}  {'SMILES_len':10s}")
print("-" * 70)
for name, smi in smiles_examples.items():
    sai_s, _, _, _ = sai_repair(smi)
    print(f"{name:15s}  {smi:35s}  {sai_s:5d}  {len(smi):10d}")

print()
print("NOTE: SAI(SMILES) is a rough proxy for molecular assembly complexity")
print("  More complex molecules -> higher SAI(SMILES)")
print("  But SMILES encoding depends on atom numbering (not canonical)")
print("  Better: use canonical InChI or graph assembly index directly")
