#!/usr/bin/env python3
"""
Benchmarking Merge Sort vs Quick Sort (Divide-and-Conquer)

Usage:
  python benchmark.py

Outputs:
  - divide_and_conquer_benchmarks.csv
  - time_chart_nmax.png
  - memory_chart_nmax.png
"""
import random
import time
import tracemalloc
import sys
from statistics import mean
from typing import List, Tuple, Callable, Dict
import pandas as pd
import matplotlib.pyplot as plt

sys.setrecursionlimit(1000000)

def merge_sort(arr: List[int]) -> List[int]:
    """Classic top-down merge sort; returns a new sorted list. O(n log n) time, O(n) space."""
    n = len(arr)
    if n <= 1:
        return arr[:]
    mid = n // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    i = j = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])
    return merged

def _median_of_three(a: int, b: int, c: int) -> int:
    return sorted([a, b, c])[1]

def _hoare_partition(a: List[int], low: int, high: int) -> int:
    """Hoare partition with median-of-three pivot selection."""
    mid = (low + high) // 2
    pivot_val = _median_of_three(a[low], a[mid], a[high])
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while a[i] < pivot_val:
            i += 1
        j -= 1
        while a[j] > pivot_val:
            j -= 1
        if i >= j:
            return j
        a[i], a[j] = a[j], a[i]

def quick_sort_inplace(a: List[int], low: int, high: int) -> None:
    """In-place quicksort using Hoare partition + median-of-three pivot."""
    if low < high:
        p = _hoare_partition(a, low, high)
        quick_sort_inplace(a, low, p)
        quick_sort_inplace(a, p + 1, high)

def quick_sort(arr: List[int]) -> List[int]:
    """Wrapper returning a sorted copy to align interface with merge_sort."""
    a = arr[:]
    if len(a) > 1:
        quick_sort_inplace(a, 0, len(a) - 1)
    return a

def dataset_sorted(n: int):
    return list(range(n))

def dataset_reverse(n: int):
    return list(range(n, 0, -1))

def dataset_random(n: int, seed: int = None):
    rnd = random.Random(seed)
    return [rnd.randint(0, 10**9) for _ in range(n)]

def measure(alg_fn, data):
    """Return (time_ms, peak_kb)."""
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    out = alg_fn(data)
    t1 = time.perf_counter_ns()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert out == sorted(data), "Sorting algorithm returned incorrect result."
    time_ms = (t1 - t0) / 1e6
    peak_kb = peak // 1024
    return time_ms, peak_kb

ALGORITHMS: Dict[str, Callable] = {
    "Merge Sort": merge_sort,
    "Quick Sort (Hoare, median-of-three)": quick_sort,
}

DATASETS = {
    "sorted": dataset_sorted,
    "reverse": dataset_reverse,
    "random": dataset_random,
}

N_SIZES = [2000, 5000, 10000]
TRIALS = 3

def main():
    records = []
    for n in N_SIZES:
        for dtype, gen in DATASETS.items():
            base_seed = 42
            for alg_name, alg_fn in ALGORITHMS.items():
                times, peaks = [], []
                for trial in range(TRIALS):
                    data = gen(n, seed=base_seed + trial) if dtype == "random" else gen(n)
                    t_ms, p_kb = measure(alg_fn, data)
                    times.append(t_ms); peaks.append(p_kb)
                records.append({
                    "algorithm": alg_name,
                    "n": n,
                    "dataset": dtype,
                    "time_ms_avg": mean(times),
                    "peak_kb_avg": mean(peaks),
                    "trials": TRIALS,
                })
    import pandas as pd
    df = pd.DataFrame(records)
    csv_path = "divide_and_conquer_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    # Charts for largest n
    nmax = max(N_SIZES)
    df_n = df[df["n"] == nmax]
    import matplotlib.pyplot as plt
    labels = [f'{d}\\n{a.split(" (")[0]}' for d, a in zip(df_n["dataset"], df_n["algorithm"])]
    plt.figure(figsize=(9,6))
    plt.bar(labels, df_n["time_ms_avg"])
    plt.title(f"Execution Time (ms) for n={nmax}")
    plt.ylabel("Time (ms)"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig("time_chart_nmax.png"); plt.close()
    plt.figure(figsize=(9,6))
    plt.bar(labels, df_n["peak_kb_avg"])
    plt.title(f"Peak Memory (KB) for n={nmax}")
    plt.ylabel("Peak Memory (KB)"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig("memory_chart_nmax.png"); plt.close()
    print("Wrote:", csv_path, "time_chart_nmax.png", "memory_chart_nmax.png")

if __name__ == "__main__":
    main()
