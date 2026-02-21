#!/usr/bin/env python3
"""
Plot SPANet purity vs version directly from testlog text files.

- Input:
    spanet_output/version_*/testlog.txt
    spanet_output/version_*/testlog_last.txt

- Behavior:
    * If a version is missing either log file, it is skipped
    * Uses Full row only
    * Metrics: Event / H / Ht / Lt purity
    * best = solid, last = dashed

- Output:
    purity_vs_version_full.png / .pdf
    + inline imgcat display if available
"""

from pathlib import Path
import shutil
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import os

def imgCat(imgpath):
    imgcat = Path.home() / ".iterm2" / "imgcat"

    if not imgcat.exists():
        print("\033[1;33m[INFO]\033[0m imgcat not found — skipping inline display")
        return

    print(f"\033[1;32m▶ imgcat {imgpath}\033[0m")
    subprocess.run(
        [str(imgcat), str(imgpath)],
        check=False,
    )
# ------------------------
# config
# ------------------------
BASE_DIR = Path("spanet_output")

METRICS = ["Event", "H", "Ht", "Lt"]
METRIC_IDX = {
    "Event": 3,
    "H": 4,
    "Ht": 5,
    "Lt": 6,
}

COLORS = {
    "Event": "black",
    "H": "tab:red",
    "Ht": "tab:blue",
    "Lt": "tab:green",
}
LINESTYLES = {"best": "-", "last": "--"}
MARKERS = {"best": "o", "last": "s"}

PNG_OUT = "purity_vs_version_full.png"
PDF_OUT = "purity_vs_version_full.pdf"


# ------------------------
# helpers
# ------------------------
def extract_full_row(logfile: Path):
    """
    Extract purity values from the 'Full' row.
    Returns dict or None if file / row is missing.
    """
    if not logfile.exists():
        return None

    with logfile.open() as f:
        for line in f:
            if line.strip().startswith("Full"):
                parts = [p.strip() for p in line.split("|")]
                return {
                    m: float(parts[METRIC_IDX[m]])
                    for m in METRICS
                }

    return None


# ------------------------
# collect data
# ------------------------
versions = []
data = {
    "best": {m: [] for m in METRICS},
    "last": {m: [] for m in METRICS},
}

version_dirs = sorted(
    BASE_DIR.glob("version_*"),
    key=lambda p: int(p.name.split("_")[-1])
)

if not version_dirs:
    raise RuntimeError(f"No version_* directories found under {BASE_DIR}")

for vdir in version_dirs:
    version = int(vdir.name.split("_")[-1])

    best_log = vdir / "testlog.txt"
    last_log = vdir / "testlog_last.txt"

    best_vals = extract_full_row(best_log)
    last_vals = extract_full_row(last_log)

    # skip incomplete versions
    if best_vals is None or last_vals is None:
        print(f"\033[1;33m[INFO]\033[0m Skipping version_{version} (missing log or Full row)")
        continue

    versions.append(version)

    for m in METRICS:
        data["best"][m].append(best_vals[m])
        data["last"][m].append(last_vals[m])


if not versions:
    raise RuntimeError("No valid versions found with complete logs")


# ------------------------
# plot
# ------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for m in METRICS:
    ax.plot(
        versions,
        data["best"][m],
        label=f"{m} (best)",
        color=COLORS[m],
        linestyle=LINESTYLES["best"],
        marker=MARKERS["best"],
        linewidth=2,
    )
    ax.plot(
        versions,
        data["last"][m],
        label=f"{m} (last)",
        color=COLORS[m],
        linestyle=LINESTYLES["last"],
        marker=MARKERS["last"],
        linewidth=2,
    )

ax.set_xlabel("SPANet version", fontsize=13)
ax.set_ylabel("Purity", fontsize=13)
ax.set_ylim(0.0, 1.0)

ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=10)

ax.set_title(
    "Full-bin Purity vs SPANet Version\n"
    "(solid: best checkpoint, dashed: last checkpoint)",
    fontsize=14,
)

plt.tight_layout()
plt.savefig(PDF_OUT)
plt.savefig(PNG_OUT, dpi=150)
plt.close()


# ------------------------
# imgcat
# ------------------------
imgCat(PNG_OUT)

