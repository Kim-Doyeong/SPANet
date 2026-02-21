#!/usr/bin/env python3
"""
Plot SPANet purity vs version directly from testlog text files.

- Input:
    spanet_output/version_*/testlog.txt
    spanet_output/version_*/testlog_last.txt

- Behavior:
    * If a version is missing either log file, it is skipped
    * Extracts purity from TWO different "Full" rows:
        (1) first "Full" row in the file  -> usually Event Type: *lt*ht*h
        (2) last  "Full" row in the file  -> usually most specific Event Type block
    * Metrics: Event / H / Ht / Lt purity
    * best = solid, last = dashed (overlaid in the same plot)

- Output:
    purity_vs_version_first_full.png / .pdf
    purity_vs_version_last_full.png  / .pdf
    + inline imgcat display if available
"""

from pathlib import Path
import subprocess
import matplotlib.pyplot as plt


# ------------------------
# helpers
# ------------------------
def imgCat(imgpath: str | Path):
    imgcat = Path.home() / ".iterm2" / "imgcat"

    if not imgcat.exists():
        print("\033[1;33m[INFO]\033[0m imgcat not found — skipping inline display")
        return

    print(f"\033[1;32m▶ imgcat {imgpath}\033[0m")
    subprocess.run([str(imgcat), str(imgpath)], check=False)


# ------------------------
# config
# ------------------------
BASE_DIR = Path("spanet_output")

METRICS = ["Event", "H", "Ht", "Lt"]
# 'Full | EventProp | JetProp | EventPurity | HPurity | HtPurity | LtPurity'
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


# ------------------------
# parsing
# ------------------------
def extract_full_row(logfile: Path, which: str = "first"):
    """
    Extract purity values from a 'Full' row and keep its Event Type.

    The testlog structure is:
        Event Type: <something>
        ...
        Full | ... | ... | ...

    We track the most recent 'Event Type:' and attach it to each Full row.

    Args:
        logfile: path to testlog*.txt
        which: "first" or "last" Full row in the whole file

    Returns:
        dict like {"event_type": str, "values": {metric: float, ...}}
        or None if file / row is missing.
    """
    if not logfile.exists():
        return None
    if which not in ("first", "last"):
        raise ValueError("which must be 'first' or 'last'")

    current_event_type = None
    full_rows = []

    with logfile.open() as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("Event Type:"):
                current_event_type = line.split("Event Type:")[1].strip()
                continue

            if line.startswith("Full") and current_event_type is not None:
                parts = [p.strip() for p in line.split("|")]
                try:
                    values = {m: float(parts[METRIC_IDX[m]]) for m in METRICS}
                except (IndexError, ValueError):
                    # if formatting changed or N/A appears, just skip this Full row
                    continue

                full_rows.append(
                    {
                        "event_type": current_event_type,
                        "values": values,
                    }
                )

    if not full_rows:
        return None

    return full_rows[0] if which == "first" else full_rows[-1]


# ------------------------
# collect data
# ------------------------
versions: list[int] = []

data = {
    "first": {  # first Full row in file
        "best": {m: [] for m in METRICS},
        "last": {m: [] for m in METRICS},
    },
    "last": {   # last Full row in file
        "best": {m: [] for m in METRICS},
        "last": {m: [] for m in METRICS},
    },
}

event_type_label = {
    "first": {"best": None, "last": None},
    "last": {"best": None, "last": None},
}

version_dirs = sorted(
    BASE_DIR.glob("version_*"),
    key=lambda p: int(p.name.split("_")[-1]),
)

if not version_dirs:
    raise RuntimeError(f"No version_* directories found under {BASE_DIR}")

for vdir in version_dirs:
    version = int(vdir.name.split("_")[-1])

    best_log = vdir / "testlog.txt"
    last_log = vdir / "testlog_last.txt"

    # first Full
    best_first = extract_full_row(best_log, which="first")
    last_first = extract_full_row(last_log, which="first")

    # last Full
    best_last = extract_full_row(best_log, which="last")
    last_last = extract_full_row(last_log, which="last")

    # skip incomplete versions
    if (best_first is None or last_first is None or
        best_last is None or last_last is None):
        print(f"\033[1;33m[INFO]\033[0m Skipping version_{version} (missing log or Full row)")
        continue

    versions.append(version)

    for m in METRICS:
        data["first"]["best"][m].append(best_first["values"][m])
        data["first"]["last"][m].append(last_first["values"][m])

        data["last"]["best"][m].append(best_last["values"][m])
        data["last"]["last"][m].append(last_last["values"][m])

    # store event type label (assume stable-ish; keep last seen)
    event_type_label["first"]["best"] = best_first["event_type"]
    event_type_label["first"]["last"] = last_first["event_type"]
    event_type_label["last"]["best"] = best_last["event_type"]
    event_type_label["last"]["last"] = last_last["event_type"]

if not versions:
    raise RuntimeError("No valid versions found with complete logs")


# ------------------------
# plot
# ------------------------
def make_plot(which_full: str, png_out: str, pdf_out: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for m in METRICS:
        ax.plot(
            versions,
            data[which_full]["best"][m],
            label=f"{m} (best)",
            color=COLORS[m],
            linestyle=LINESTYLES["best"],
            marker=MARKERS["best"],
            linewidth=2,
        )
        ax.plot(
            versions,
            data[which_full]["last"][m],
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

    et_best = event_type_label[which_full]["best"]
    et_last = event_type_label[which_full]["last"]

    if et_best == et_last:
        et_line = f"Event Type: {et_best}"
    else:
        et_line = f"Event Type (best): {et_best} | (last): {et_last}"

    ax.set_title(
        f"{which_full.capitalize()} Full-bin Purity vs SPANet Version\n"
        f"(solid: best checkpoint, dashed: last checkpoint)\n"
        f"{et_line}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(pdf_out)
    plt.savefig(png_out, dpi=150)
    plt.close()

    imgCat(png_out)


make_plot(
    which_full="first",
    png_out="purity_vs_version_first_full.png",
    pdf_out="purity_vs_version_first_full.pdf",
)

make_plot(
    which_full="last",
    png_out="purity_vs_version_last_full.png",
    pdf_out="purity_vs_version_last_full.pdf",
)
