"""
Data import + spectrum plotting + math helpers for the SALSA HI 21 cm
observations.

You decide what counts as a peak / V_r,max / an outlier — these helpers just
give you the canonical formulas and the loaded data.

Data + plotting:
    read_salsa_csv(path)   -> (l, b, v_kms, amplitude)
    clean_and_smooth(amp)  -> Savitzky-Golay smoothed amplitude
    estimate_noise(s, v)   -> RMS noise sampled from 200-280 km/s off-line
    plot_spectra(...)      -> 4-column grid of all spectra

Math (tangent-point method, kinematic distance, mapping, mass):
    tangent_R(l)               -> R0 sin(l)                              [kpc]
    tangent_V(v_rmax, l)       -> v_rmax + V0 sin(l)                   [km/s]
    kinematic_R(v_r, l)        -> R0 V0 sin(l) / (V0 sin(l) + v_r)      [kpc]
    cloud_distances(R, l)      -> [(r, x, y), ...]                      [kpc]
    enclosed_mass(R, V)        -> M(<R) in solar masses
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Galactic parameters (IAU recommended values)
R0_KPC = 8.5     # Sun -> Galactic centre distance
V0_KMS = 220.0   # Sun's circular speed

# Unit conversions
KPC_TO_M = 3.086e19
M_SUN_KG = 1.989e30
G_SI    = 6.674e-11


# ---------------------------------------------------------------- math
def tangent_R(l_deg):
    """Galactocentric radius of the tangent point along longitude l.

    R = R0 * sin(l)        — valid for Quadrants I and IV (R < R0)
    """
    return R0_KPC * np.sin(np.deg2rad(l_deg))


def tangent_V(v_rmax_kms, l_deg):
    """Rotational velocity at the tangent point from the observed terminal
    velocity V_r,max.

    V = V_r,max + V0 * sin(l)
    """
    return v_rmax_kms + V0_KMS * np.sin(np.deg2rad(l_deg))


def kinematic_R(v_r_kms, l_deg):
    """Galactocentric radius of a cloud at radial velocity v_r along
    longitude l, assuming a flat rotation curve V(R) = V0.

    R = R0 * V0 * sin(l) / (V0 * sin(l) + v_r)
    """
    sin_l = np.sin(np.deg2rad(l_deg))
    return R0_KPC * V0_KMS * sin_l / (V0_KMS * sin_l + v_r_kms)


def cloud_distances(R_kpc, l_deg):
    """Heliocentric distance(s) and (x, y) for a cloud at galactocentric R
    seen at longitude l.

    Convention: Galactic centre at (0, 0), Sun at (0, -R0). l = 0 points
    toward the GC, l = 90 toward -x, l = 180 toward -y (anticenter).

    Returns a list of (r, x, y) tuples. In Q1/Q4 there are two solutions
    (near and far). In Q2/Q3 there is one.
    """
    l = np.deg2rad(l_deg)
    disc = R_kpc ** 2 - (R0_KPC * np.sin(l)) ** 2
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    base = R0_KPC * np.cos(l)
    out = []
    for r in (base + sq, base - sq):
        if r <= 0:
            continue
        x = -r * np.sin(l)
        y = -R0_KPC + r * np.cos(l)
        out.append((r, x, y))
    return out


def enclosed_mass(R_kpc, V_kms):
    """Mass interior to galactocentric radius R, in solar masses.

    Newton + circular orbit:  M(<R) = V^2 R / G
    """
    R_m = R_kpc * KPC_TO_M
    V_ms = V_kms * 1000.0
    return V_ms ** 2 * R_m / G_SI / M_SUN_KG



def read_salsa_csv(path):
    """Return (longitude_deg, latitude_deg, v_kms, amplitude), sorted by v."""
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
                    data.append([float(parts[0]), float(parts[1]),
                                 float(parts[2])])
                except ValueError:
                    pass
    arr = np.array(data)
    v_kms = arr[:, 2] / 1000.0
    amp = arr[:, 1]
    order = np.argsort(v_kms)
    return lon, lat, v_kms[order], amp[order]


def clean_and_smooth(amp):
    """Replace large negative spikes with the median; Savitzky-Golay smooth."""
    med = np.median(amp)
    mad = np.median(np.abs(amp - med)) + 1e-9
    bad = amp < med - 6 * 1.4826 * mad
    cleaned = amp.copy()
    cleaned[bad] = med
    return savgol_filter(cleaned, window_length=8, polyorder=3)


def estimate_noise(smoothed, v_kms):
    """RMS in the off-line window 200 < |v| < 280 km/s."""
    mask = (np.abs(v_kms) > 200) & (np.abs(v_kms) < 280)
    if mask.sum() < 30:
        n = len(smoothed)
        mask = np.zeros(n, dtype=bool)
        mask[20:n // 10] = True
        mask[-n // 10:-20] = True
    sample = smoothed[mask]
    return float(np.std(sample - np.median(sample)))


def find_terminal_peak(v_kms, smoothed, l_deg, noise,
                       v_clip=150.0, snr=3.0):
    """Pick the terminal-velocity (V_r,max) peak of an HI spectrum.

    The terminal velocity sits at the extreme-velocity edge of the
    emission: most positive in Quadrant I (l <= 90), most negative in
    Quadrant II (l > 90). Anything outside +/- v_clip km/s is ignored
    (RFI, local-arm sidelobes, non-Galactic emission).

    Returns (v_peak_kms, amp_peak) or (None, None) if no peak qualifies.
    """
    s = smoothed - np.median(smoothed)
    window = (v_kms > -v_clip) & (v_kms < v_clip)
    if not window.any():
        return None, None

    idx_window = np.where(window)[0]
    peaks_local, _ = find_peaks(s[window], height=snr * noise)
    if peaks_local.size == 0:
        return None, None

    peaks = idx_window[peaks_local]
    pick = peaks[np.argmax(v_kms[peaks])] if l_deg <= 90.0 \
        else peaks[np.argmin(v_kms[peaks])]
    return float(v_kms[pick]), float(s[pick])


def plot_spectra(spectra, out_path):
    """spectra: list of dicts with keys l_deg, v_kms, smoothed, noise."""
    spectra = sorted(spectra, key=lambda r: r["l_deg"])
    n = len(spectra)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, spectra):
        v, s, l = r["v_kms"], r["smoothed"], r["l_deg"]
        color = "tab:blue" if l <= 90.0 else "tab:green"
        baseline = np.median(s)
        ax.plot(v, s - baseline, lw=1.0, color=color)
        ax.fill_between(v, 0, s - baseline,
                        where=((s - baseline > 2.5 * r["noise"]) &
                               (v > -200) & (v < 200)),
                        alpha=0.25, color=color)
        ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
        ax.axhline(2.5 * r["noise"], color="grey", lw=0.5, ls=":", alpha=0.7)
        if r.get("v_rmax") is not None:
            ax.axvline(r["v_rmax"], color="red", lw=0.8, ls="--", alpha=0.8)
            ax.plot(r["v_rmax"], r["amp_rmax"], "rx", ms=7)
        quadrant = "I" if l <= 90.0 else "II"
        title = f"l = {l:.0f}°  ({quadrant})"
        if r.get("v_rmax") is not None:
            title += f"   V$_r$$_,$$_m$$_a$$_x$ = {r['v_rmax']:.0f} km/s"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-250, 250)
        ax.grid(alpha=0.3)

    for ax in axes[len(spectra):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("V$_{LSR}$ (km/s)")

    fig.suptitle("SALSA HI spectra — Quadrant I (blue) + Quadrant II (green)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    spectra = []
    for path in files:
        l, b, v, a = read_salsa_csv(path)
        s = clean_and_smooth(a)
        noise = estimate_noise(s, v)
        v_rmax, amp_rmax = find_terminal_peak(v, s, l, noise)
        spectra.append({
            "file": os.path.basename(path),
            "l_deg": l, "b_deg": b,
            "v_kms": v, "amplitude": a,
            "smoothed": s,
            "noise": noise,
            "v_rmax": v_rmax,
            "amp_rmax": amp_rmax,
        })

    out_path = os.path.join(OUT_DIR, "spectra.png")
    plot_spectra(spectra, out_path)
    print(f"Plotted {len(spectra)} spectra → {out_path}")

    print(f"\n{'l (deg)':>8}  {'V_r,max (km/s)':>15}  {'R (kpc)':>8}  {'V (km/s)':>9}")
    for r in sorted(spectra, key=lambda x: x["l_deg"]):
        if r["v_rmax"] is None:
            print(f"{r['l_deg']:>8.1f}  {'—':>15}")
            continue
        R = tangent_R(r["l_deg"])
        V = tangent_V(r["v_rmax"], r["l_deg"])
        print(f"{r['l_deg']:>8.1f}  {r['v_rmax']:>15.1f}  {R:>8.2f}  {V:>9.1f}")


if __name__ == "__main__":
    main()
