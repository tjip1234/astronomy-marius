import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")

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
def main():
    path = os.path.join(DATA_DIR, "SALSA-vale-20260511T082607.csv")
    lon, lat, v_kms, amp = read_salsa_csv(path)
    print(f"Target: {lon:.2f}, {lat:.2f}")
    amp_smooth = savgol_filter(amp, window_length=8, polyorder=3)
    #take the derivative of the smoothed amplitude with respect to velocity
    d_amp_d_v = np.gradient(amp_smooth, v_kms)
    #find the velocity where the derivative is maximum
    max_deriv_index = np.argmax(d_amp_d_v)
    v_max_deriv = v_kms[max_deriv_index]
    print(f"Velocity of maximum derivative: {v_max_deriv:.2f} km/s")
    plt.plot(v_kms, amp, label="Raw")
    plt.plot(v_kms, amp_smooth, label="Smoothed")
    #smooth the derivative with a Savitzky-Golay filter to reduce noise
    d_amp_d_v_smooth = savgol_filter(d_amp_d_v, window_length=8, polyorder=3)
    reverse_d_amp_d_v_smooth = -d_amp_d_v_smooth
    peaks, _ = find_peaks(reverse_d_amp_d_v_smooth, height=5)
    plt.plot(v_kms, d_amp_d_v_smooth, label="Smoothed Derivative")
    plt.plot(v_kms[peaks], d_amp_d_v_smooth[peaks], "x", label="Peaks")
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig("salsa_spectrum.png")

if __name__ == "__main__":
    main()