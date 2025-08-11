#!/usr/bin/env python3
# verify_spectrograms.py
import numpy as np
import json, os, glob
from pathlib import Path
import matplotlib.pyplot as plt

# --- Config (edit if you keep data elsewhere) ---
BASE_DIR = Path("/home/jakelove/Documents/2025/Thesis")
# Use the focused set by default (what you trained/clustered with)
SPECTROGRAM_DIR = BASE_DIR / "spectrograms_focused"

# We’ll try these names in order:
SPEC_CANDIDATES = [
    "spectrograms_enhanced.npy",
    "spectrograms_original.npy",
    "spectrograms.npy",
]
FREQ_FILE = "frequencies.npy"   # optional
TIME_FILE = "times.npy"         # optional
META_FILE = "metadata.json"     # optional

OUTPUT_DIR = BASE_DIR / "verify_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLES_PER_CLUSTER = 12
CMAP = "hot"

def _load_first_existing(base: Path, names):
    for name in names:
        p = base / name
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"Could not find any spectrogram array in {base} "
                            f"(tried: {', '.join(names)})")

def _safe_load(path: Path):
    try:
        return np.load(path)
    except Exception:
        return None

def _fallback_freqs(specs):
    # If (N, 129, T), assume 0–500 Hz
    if specs.ndim == 3 and specs.shape[1] == 129:
        return np.linspace(0, 500, 129)
    return None

def _fallback_times(specs):
    # If (N, F, 64), assume ~26 ms/bin for ~1.66s total
    if specs.ndim == 3 and specs.shape[2] == 64:
        return np.arange(64) * 0.026
    return None

def _find_latest_labels(results_dir: Path):
    paths = sorted(results_dir.glob("spectral_cluster_labels_*.npy"))
    if not paths:
        return None
    # choose most recently modified
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]

def _plot_spec(ax, spec, freqs=None, times=None, title=None):
    im = ax.imshow(spec, origin="lower", aspect="auto", cmap=CMAP,
                   interpolation="bilinear")
    if title: ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Freq")
    # Light ticks if we have axes
    if freqs is not None:
        yticks = np.linspace(0, spec.shape[0]-1, 5).astype(int)
        yhz = np.interp(yticks, np.arange(len(freqs)), freqs)
        ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.0f}" for v in yhz])
    if times is not None:
        xticks = np.linspace(0, spec.shape[1]-1, 5).astype(int)
        xt = np.interp(xticks, np.arange(len(times)), times)
        ax.set_xticks(xticks); ax.set_xticklabels([f"{v:.2f}" for v in xt])
    return im

def main():
    # ---- Load core arrays ----
    specs = _load_first_existing(SPECTROGRAM_DIR, SPEC_CANDIDATES)
    freqs = _safe_load(SPECTROGRAM_DIR / FREQ_FILE)
    times = _safe_load(SPECTROGRAM_DIR / TIME_FILE)

    if freqs is None: freqs = _fallback_freqs(specs)
    if times is None: times = _fallback_times(specs)

    meta = {}
    meta_path = SPECTROGRAM_DIR / META_FILE
    if meta_path.exists():
        try:
            meta = json.load(open(meta_path, "r"))
        except Exception:
            meta = {}

    print("=== Dataset Summary ===")
    print(f"Folder: {SPECTROGRAM_DIR}")
    print(f"Spectrogram array shape: {specs.shape}")
    print(f"Value range: [{specs.min():.2f}, {specs.max():.2f}]  mean={specs.mean():.2f}, std={specs.std():.2f}")

    if freqs is not None:
        print(f"Frequency axis: {freqs[0]:.1f} – {freqs[-1]:.1f} Hz  (Δ≈{(freqs[1]-freqs[0]):.2f} Hz)")
    else:
        print("Frequency axis: not found (no freqs.npy and no 129-bin fallback)")

    if times is not None and len(times) > 1:
        print(f"Time axis: 0 – {times[-1]:.3f} s  (Δ≈{(times[1]-times[0])*1000:.1f} ms)")
    else:
        print("Time axis: not found (no times.npy and no 64-bin fallback)")

    # ---- Try to load latest cluster labels automatically ----
    labels_path = _find_latest_labels(BASE_DIR / "spectral_clustering_results")
    if labels_path is None:
        print("\nNo cluster labels found in spectral_clustering_results/. Skipping cluster analysis.")
        return
    labels = np.load(labels_path)
    print(f"\nLoaded cluster labels: {labels_path.name}  (n={len(labels)})")

    # Align lengths if needed (truncate to min)
    n = len(specs)
    m = len(labels)
    if n != m:
        print(f"WARNING: spectrograms (N={n}) and labels (N={m}) differ. Truncating to min(N)={min(n,m)}.")
        N = min(n, m)
        specs = specs[:N]
        labels = labels[:N]
        if freqs is not None: freqs = freqs  # unchanged; per-bin axis
        if times is not None: times = times  # unchanged

    # ---- Cluster counts ----
    uniq, counts = np.unique(labels, return_counts=True)
    pct = counts / len(labels) * 100.0
    print("\nCluster sizes:")
    for c, ct, p in zip(uniq, counts, pct):
        print(f"  cluster {int(c)}: {ct} ({p:.1f}%)")

    # Save counts CSV
    with open(OUTPUT_DIR / "cluster_counts.csv", "w") as f:
        f.write("cluster_id,count,percentage\n")
        for c, ct, p in zip(uniq, counts, pct):
            f.write(f"{int(c)},{int(ct)},{p:.4f}\n")

    # ---- Mean spectrogram per cluster ----
    import matplotlib.pyplot as plt
    n_clusters = len(uniq)
    cols = min(4, n_clusters)
    rows = int(np.ceil(n_clusters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows), squeeze=False)
    for ax in axes.ravel(): ax.axis('off')

    for i, c in enumerate(uniq):
        idx = np.where(labels == c)[0]
        mean_spec = specs[idx].mean(axis=0)
        r, k = divmod(i, cols)
        ax = axes[r, k]
        _plot_spec(ax, mean_spec, freqs=freqs, times=times, title=f"Cluster {int(c)} (n={len(idx)})")
        ax.axis('on')

    fig.suptitle("Mean spectrogram per cluster", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cluster_means.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR/'cluster_means.png'}")

    # ---- Sample grids per cluster ----
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        take = idx[:SAMPLES_PER_CLUSTER] if len(idx) >= SAMPLES_PER_CLUSTER else idx
        k = len(take)
        cols = min(4, k)
        rows = int(np.ceil(k / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows), squeeze=False)
        for ax in axes.ravel(): ax.axis('off')
        for j, ii in enumerate(take):
            r, cc = divmod(j, cols)
            ax = axes[r, cc]
            _plot_spec(ax, specs[ii], freqs=freqs, times=times, title=f"idx {ii}")
            ax.axis('on')
        fig.suptitle(f"Cluster {int(c)}: {k} sample(s)", fontsize=12)
        fig.tight_layout()
        out = OUTPUT_DIR / f"cluster_{int(c)}_samples.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"Saved: {out}")

    # ---- Rough dominant-frequency summary per cluster ----
    # For each sample, average over time → F vector; take argmax bin → Hz (or bin index).
    if specs.ndim == 3:
        F = specs.shape[1]
        if freqs is None:
            # convert bin index to pseudo-Hz
            hz_axis = np.arange(F)
        else:
            hz_axis = freqs
        domfreqs_per_cluster = []
        for c in uniq:
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                domfreqs_per_cluster.append([int(c), 0, 0, 0, 0])
                continue
            dom_vals = []
            for ii in idx:
                spec = specs[ii]
                # mean over time axis -> shape (F,)
                prof = spec.mean(axis=1)
                b = int(np.argmax(prof))
                dom_vals.append(hz_axis[b])
            dom_vals = np.array(dom_vals, dtype=float)
            domfreqs_per_cluster.append([
                int(c),
                float(np.median(dom_vals)),
                float(np.mean(dom_vals)),
                float(np.percentile(dom_vals, 25)),
                float(np.percentile(dom_vals, 75)),
            ])
        with open(OUTPUT_DIR / "cluster_dominant_freqs.csv", "w") as f:
            f.write("cluster_id,median_hz,mean_hz,q25_hz,q75_hz\n")
            for row in domfreqs_per_cluster:
                f.write(",".join(map(str, row)) + "\n")
        print(f"Saved: {OUTPUT_DIR/'cluster_dominant_freqs.csv'}")

    print("\n✓ Verification complete. See:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
