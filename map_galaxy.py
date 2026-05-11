"""
Build a top-down map of the Milky Way from SALSA HI spectra.

For every spectrum (l, b=0):
  - Savitzky-Golay smooth, baseline-subtract
  - find every significant peak (>= 5 sigma, prominence)
  - convert each peak (l, v_r) -> (R, distance r, x, y) assuming a flat rotation
    curve  V(R) = V0:
        R = R0 * V0 * sin(l) / (V0 * sin(l) + v_r)
        r = R0 cos(l) ± sqrt(R^2 - R0^2 sin^2(l))     (Q1/Q4: two solutions)
                       single solution in Q2/Q3.
  - plot (x, y) with the Sun at the origin and the Galactic center at (0, R0).
"""
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.signal import find_peaks, savgol_filter

R0_KPC = 8.5
V0_KMS = 220.0
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Rough galactocentric radii of named arms, for colouring
ARM_BINS = [
    (0.0,  4.0,  "Inner / 3-kpc arm",  "#d62728"),
    (4.0,  5.5,  "Norma arm",          "#ff7f0e"),
    (5.5,  7.0,  "Scutum-Centaurus",   "#bcbd22"),
    (7.0,  8.2,  "Sagittarius-Carina", "#2ca02c"),
    (8.2,  9.5,  "Local (Orion spur)", "#17becf"),
    (9.5, 12.0,  "Perseus arm",        "#1f77b4"),
    (12.0, 25.0, "Outer arm",          "#9467bd"),
]


def arm_for_R(R):
    for lo, hi, name, color in ARM_BINS:
        if lo <= R < hi:
            return name, color
    return "unknown", "grey"


def read_salsa_csv(path):
    lon = lat = None
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith("# Target:"):
                m = re.search(r"# Target:\s*([-\d.]+),\s*([-\d.]+)", line)
                lon, lat = float(m.group(1)), float(m.group(2))
                continue
            if line.startswith("#") or line.startswith("frequency_hz"):
                continue
            parts = line.strip().split(",")
            if len(parts) == 3:
                try:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    arr = np.array(data)
    v = arr[:, 2] / 1000.0
    a = arr[:, 1]
    order = np.argsort(v)
    return lon, lat, v[order], a[order]


def clean_and_smooth(amp):
    med = np.median(amp)
    mad = np.median(np.abs(amp - med)) + 1e-9
    bad = amp < med - 6 * 1.4826 * mad
    cleaned = amp.copy()
    cleaned[bad] = med
    return savgol_filter(cleaned, window_length=21, polyorder=3)


def estimate_noise(smoothed, v):
    # Galactic HI sits in |v| < ~200; sample 200-290 km/s as off-line baseline.
    # Skip the bandpass edges (last 20 channels) which can have rolloff.
    mask = (np.abs(v) > 200) & (np.abs(v) < 280)
    if mask.sum() < 30:
        n = len(smoothed)
        mask = np.zeros(n, dtype=bool)
        mask[20:n // 10] = True
        mask[-n // 10:-20] = True
    s = smoothed[mask]
    return float(np.std(s - np.median(s)))


def cloud_distances(R, l_deg):
    """Return list of (r, x, y) tuples for clouds at galactocentric radius R.

    Convention matches the NASA artist roadmap:
        Galactic centre at (0, 0).
        Sun at (0, -R0) (i.e., below the GC).
        l = 0 points from Sun toward GC (+y).
        l increases counterclockwise, so l = 90 is -x (left).
        l = 180 is -y (anticenter, below Sun).
    """
    l = np.deg2rad(l_deg)
    disc = R ** 2 - (R0_KPC * np.sin(l)) ** 2
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    base = R0_KPC * np.cos(l)
    rs = [base + sq, base - sq]
    out = []
    for r in rs:
        if r <= 0:
            continue
        x = -r * np.sin(l)
        y = -R0_KPC + r * np.cos(l)
        out.append((r, x, y))
    return out


def find_clouds(v, smoothed, noise, l_deg):
    """Find every emission peak and back out R, distance, position."""
    baseline = np.median(smoothed)
    sig = smoothed - baseline
    in_band = (v > -200) & (v < 200)
    sig_band = np.where(in_band, sig, -np.inf)
    peaks, _ = find_peaks(
        sig_band,
        height=3 * noise,
        prominence=1.5 * noise,
        distance=10,
    )
    clouds = []
    sin_l = np.sin(np.deg2rad(l_deg))
    for p in peaks:
        v_r = v[p]
        denom = V0_KMS * sin_l + v_r
        if abs(denom) < 1e-6:
            continue
        R = R0_KPC * V0_KMS * sin_l / denom
        if R <= 0 or R > 20:
            continue
        for r, x, y in cloud_distances(R, l_deg):
            arm_name, color = arm_for_R(R)
            clouds.append({
                "l_deg": l_deg,
                "v_r_kms": v_r,
                "amp": sig[p],
                "R_kpc": R,
                "r_kpc": r,
                "x_kpc": x,
                "y_kpc": y,
                "arm": arm_name,
                "color": color,
            })
    return clouds


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    all_clouds = []
    for path in files:
        l, b, v, a = read_salsa_csv(path)
        smoothed = clean_and_smooth(a)
        noise = estimate_noise(smoothed, v)
        clouds = find_clouds(v, smoothed, noise, l)
        all_clouds.extend(clouds)

    # write CSV
    out_csv = os.path.join(OUT_DIR, "clouds.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["l_deg", "v_r_kms", "amp", "R_kpc",
                           "r_kpc", "x_kpc", "y_kpc", "arm"],
            extrasaction="ignore",
        )
        w.writeheader()
        for c in all_clouds:
            w.writerow(c)

    # ---- plot (GC at centre, Sun at (0, -R0))
    fig, ax = plt.subplots(figsize=(9, 9))

    # faint solar circle
    ax.add_patch(Circle((0, 0), R0_KPC, fill=False, ls="--",
                        color="0.55", lw=0.8, label="solar circle"))
    # reference radii every 2 kpc — keep them subtle
    for radius in (2, 4, 6, 10, 12, 14):
        ax.add_patch(Circle((0, 0), radius, fill=False, color="0.85",
                            lw=0.5, zorder=0))

    # longitude rays from the Sun (only the surveyed half of the sky)
    sun = np.array([0.0, -R0_KPC])
    PLOT_HALF = 13.0  # plot extends ±13 kpc; clamp labels inside
    for l_ray in (0, 30, 60, 90, 120, 150, 180):
        lr = np.deg2rad(l_ray)
        direction = np.array([-np.sin(lr), np.cos(lr)])
        # ray length: clip to plot bounds
        t_x = (np.sign(direction[0]) * PLOT_HALF - sun[0]) / direction[0] \
            if abs(direction[0]) > 1e-6 else np.inf
        t_y = (np.sign(direction[1]) * PLOT_HALF - sun[1]) / direction[1] \
            if abs(direction[1]) > 1e-6 else np.inf
        t = min(abs(t_x), abs(t_y))
        end = sun + t * direction
        ax.plot([sun[0], end[0]], [sun[1], end[1]],
                color="0.8", lw=0.5, zorder=0)
        # put label ~85% of the way along the ray, inside the plot
        lbl = sun + 0.85 * t * direction
        ax.text(lbl[0], lbl[1], f"l = {l_ray}°", color="0.4",
                fontsize=8, ha="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=1))

    # cloud points — colour-coded by spiral arm
    plotted_arms = set()
    # plot in a consistent order so the legend follows R outward
    for lo, hi, arm_name, _ in ARM_BINS:
        members = [c for c in all_clouds if c["arm"] == arm_name]
        if not members:
            continue
        xs = [c["x_kpc"] for c in members]
        ys = [c["y_kpc"] for c in members]
        amps = [c["amp"] for c in members]
        ax.scatter(xs, ys,
                   s=[25 + 2 * a for a in amps],
                   color=members[0]["color"],
                   edgecolors="0.2", linewidths=0.5,
                   alpha=0.85, label=arm_name, zorder=4)

    # Sun and Galactic centre
    ax.scatter([0], [-R0_KPC], marker="o", s=70, color="gold",
               edgecolors="black", linewidths=1.2, zorder=6, label="Sun")
    ax.scatter([0], [0], marker="+", s=180, color="black",
               linewidths=2, zorder=6, label="Galactic centre")

    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_aspect("equal")
    ax.set_xlabel("x (kpc)")
    ax.set_ylabel("y (kpc)")
    ax.set_title("Top-down map of the Milky Way — SALSA HI 21 cm\n"
                 f"{len(all_clouds)} clouds, {len(files)} pointings  "
                 "(near and far solutions both plotted in Q1)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "galaxy_map.png"), dpi=140)
    plt.close(fig)

    # summary
    arm_counts = {}
    for c in all_clouds:
        arm_counts[c["arm"]] = arm_counts.get(c["arm"], 0) + 1
    print(f"{len(all_clouds)} cloud detections across {len(files)} spectra")
    print()
    print(f"{'arm':25} {'count':>6}")
    for lo, hi, name, _ in ARM_BINS:
        print(f"{name:25} {arm_counts.get(name, 0):>6}")
    print()
    print(f"Outputs:")
    print(f"  {os.path.join(OUT_DIR, 'galaxy_map.png')}")
    print(f"  {out_csv}")


if __name__ == "__main__":
    main()
