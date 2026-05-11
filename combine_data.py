"""Combine all SALSA CSVs in Data/ into one long-format file.

Output columns: longitude_deg, latitude_deg, date_utc, integration_s,
                frequency_hz, amplitude, vlsr_kms
"""
import csv
import glob
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "salsa_combined.csv")


def parse_header(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if (m := re.search(r"# Target:\s*([-\d.]+),\s*([-\d.]+)", line)):
                meta["longitude_deg"] = float(m.group(1))
                meta["latitude_deg"] = float(m.group(2))
            elif (m := re.search(r"# Date:\s*(\S+)", line)):
                meta["date_utc"] = m.group(1)
            elif (m := re.search(r"# Integration time:\s*(\d+)", line)):
                meta["integration_s"] = int(m.group(1))
    return meta


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    n_rows = 0
    with open(OUT_PATH, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow([
            "longitude_deg", "latitude_deg", "date_utc", "integration_s",
            "frequency_hz", "amplitude", "vlsr_kms",
        ])
        for path in files:
            meta = parse_header(path)
            with open(path) as f:
                for line in f:
                    if line.startswith("#") or line.startswith("frequency_hz"):
                        continue
                    parts = line.strip().split(",")
                    if len(parts) != 3:
                        continue
                    try:
                        freq = float(parts[0])
                        amp = float(parts[1])
                        vlsr_kms = float(parts[2]) / 1000.0
                    except ValueError:
                        continue
                    w.writerow([
                        meta["longitude_deg"], meta["latitude_deg"],
                        meta["date_utc"], meta["integration_s"],
                        freq, amp, vlsr_kms,
                    ])
                    n_rows += 1

    print(f"Wrote {n_rows} rows from {len(files)} spectra → {OUT_PATH}")


if __name__ == "__main__":
    main()
