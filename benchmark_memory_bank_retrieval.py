"""
Memory-Bank Retrieval Scalability Benchmark (Reviewer R2.8)

Measures top-k retrieval latency over the HERO memory bank as the bank grows,
comparing:
  - exact search (full similarity matmul, as currently deployed)
  - IVF-style approximate search (k-means coarse quantizer, probe top-p
    clusters) implemented with numpy only - a stand-in for FAISS IVF

Bank sizes: the real bank (reports/fleet_rad/fleet_memory_bank.pt, 64-d keys)
plus synthetic extensions to 10k / 100k / 1M entries.

Outputs: reports/memory_bank_retrieval_benchmark.json
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

TOP_K = 16
N_QUERIES = 50
BANK_SIZES = [3979, 10_000, 100_000, 1_000_000]
N_CLUSTERS = 256
N_PROBE = 8

OUT = Path('reports/memory_bank_retrieval_benchmark.json')


def load_real_keys():
    mb = torch.load('reports/fleet_rad/fleet_memory_bank.pt',
                    map_location='cpu', weights_only=False)
    return mb['keys'].numpy().astype(np.float32)


def make_bank(real_keys, n):
    if n <= len(real_keys):
        return real_keys[:n]
    rng = np.random.default_rng(0)
    extra = real_keys[rng.integers(0, len(real_keys), n - len(real_keys))]
    extra = extra + rng.normal(0, 0.05, extra.shape).astype(np.float32)
    return np.vstack([real_keys, extra])


def kmeans(x, k, iters=10, seed=0):
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ centroids.T, axis=1)
        for c in range(k):
            m = assign == c
            if m.any():
                centroids[c] = x[m].mean(axis=0)
    assign = np.argmax(x @ centroids.T, axis=1)
    return centroids, assign


def bench_exact(bank, queries):
    times = []
    for q in queries:
        t0 = time.perf_counter()
        sims = bank @ q
        np.argpartition(-sims, TOP_K)[:TOP_K]
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def bench_ivf(bank, queries, centroids, assign):
    # Pre-bucket the bank by cluster (index build cost excluded, done offline)
    buckets = [np.where(assign == c)[0] for c in range(len(centroids))]
    times, recalls = [], []
    for q in queries:
        true_top = set(np.argpartition(-(bank @ q), TOP_K)[:TOP_K].tolist())
        t0 = time.perf_counter()
        cl = np.argpartition(-(centroids @ q), N_PROBE)[:N_PROBE]
        cand = np.concatenate([buckets[c] for c in cl])
        sims = bank[cand] @ q
        top = cand[np.argpartition(-sims, min(TOP_K, len(sims) - 1))[:TOP_K]]
        times.append(time.perf_counter() - t0)
        recalls.append(len(true_top & set(top.tolist())) / TOP_K)
    return float(np.median(times) * 1000), float(np.mean(recalls))


def main():
    real = load_real_keys()
    dim = real.shape[1]
    rng = np.random.default_rng(1)
    queries = real[rng.integers(0, len(real), N_QUERIES)] \
        + rng.normal(0, 0.05, (N_QUERIES, dim)).astype(np.float32)

    results = {'top_k': TOP_K, 'dim': dim, 'n_clusters': N_CLUSTERS,
               'n_probe': N_PROBE, 'sizes': {}}
    print(f"{'bank size':>10} {'exact (ms)':>11} {'IVF (ms)':>9} {'IVF recall':>11}")
    for n in BANK_SIZES:
        bank = make_bank(real, n)
        exact_ms = bench_exact(bank, queries)
        centroids, assign = kmeans(bank, N_CLUSTERS)
        ivf_ms, recall = bench_ivf(bank, queries, centroids, assign)
        results['sizes'][str(n)] = {
            'exact_median_ms': round(exact_ms, 3),
            'ivf_median_ms': round(ivf_ms, 3),
            'ivf_recall_at_k': round(recall, 3),
            'speedup': round(exact_ms / ivf_ms, 1) if ivf_ms > 0 else None,
        }
        print(f"{n:>10} {exact_ms:>11.3f} {ivf_ms:>9.3f} {recall:>11.3f}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()
