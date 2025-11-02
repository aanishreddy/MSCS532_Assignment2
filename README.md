# Divide-and-Conquer: Merge Sort vs Quick Sort

This repo contains Python implementations of **Merge Sort** and **Quick Sort** (Hoare partition, median-of-three pivot) plus a small benchmark harness.

## How to Run

```bash
python benchmark.py
```

This produces:
- `divide_and_conquer_benchmarks.csv` (results table)
- `time_chart_nmax.png` (execution time for largest n)
- `memory_chart_nmax.png` (peak memory for largest n)

## What’s Inside

- `benchmark.py` – implementations + benchmark
- Results CSV + charts (you can regenerate anytime by running the script)
- A Word/Markdown **report template** to complete your write-up

## Notes

- Dependencies: `pandas`, `matplotlib`
- Python ≥ 3.9 recommended
- Algorithms verified against Python’s `sorted` for correctness

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley.
