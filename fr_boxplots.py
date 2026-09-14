#!/usr/bin/env python3
"""Box-and-whisker plots of firing rate (FR) per channel from a MEA-NAP style CSV.

Each row of the CSV holds one (FileName, Grp, Channel, FR) observation. A box is
drawn for every channel; the values inside a box are the FR readings for that
channel taken from every recording (FileName) in the selected group. Choosing an
organoid does not change the boxes: the group's boxes stay as they are and only
that organoid's readings are overlaid on top, coloured by slice, so a single
organoid can be compared against its group's channel distribution.
The FileName / Channel / FR triple is always taken from the same row, so the
association between a recording, its channel and that channel's firing rate is
never broken.

The "All groups" view is a channel overview: one box per channel built from
every recording in the file regardless of group. Use it to spot channels that
record low firing rates throughout.

Organoid identity is parsed from the file name. The segment right after the
group marker ("CT" for BCTL, "MO" for BMOS, "MT" for BMUT) is a number-letter
pair, e.g. "R250929CT7A_DIV250" -> organoid "CT7", slice "CT7A".

Usage
-----
Interactive viewer (default):
    python3 fr_boxplots.py NeuronalActivity_NodeLevel.csv

    Two drop-downs select the group ("All groups" gives the channel overview)
    and, within it, "All organoids" or a single organoid whose readings are
    highlighted. Hovering over any dot (an outlier or an individual reading)
    shows the recording, slice, channel and FR it came from. The toolbar under
    the plot saves the current figure at 300 dpi.

List what is in the file (groups -> organoids -> slices, with recording counts):
    python3 fr_boxplots.py data.csv --list

Save a figure without opening a window:
    python3 fr_boxplots.py data.csv --grp all -o channel_overview.png
    python3 fr_boxplots.py data.csv --grp BCTL -o bctl_all.png
    python3 fr_boxplots.py data.csv --grp BCTL --organoid CT7 -o bctl_ct7.png

Requires matplotlib (pip install matplotlib). Tkinter ships with python.org
builds of Python.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Column names are matched case-insensitively (the user's file may say "grp").
COL_FILENAME = "filename"
COL_GRP = "grp"
COL_CHANNEL = "channel"
COL_FR = "fr"

# Group -> marker that precedes the organoid number-letter pair in the file name.
GROUP_MARKERS = {"BCTL": "CT", "BMOS": "MO", "BMUT": "MT"}
ALL_MARKERS = tuple(GROUP_MARKERS.values())

ALL_ORGANOIDS = "All organoids"
ALL_GROUPS = "All groups"          # channel overview across every group
UNKNOWN = "unknown"

# Fixed-order categorical palette (identity colours for organoids / slices).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_COLOR = "#9a9a96"      # everything past the palette folds into "Other"
BOX_COLOR = "#52514e"        # neutral ink for boxes / whiskers / medians
GRID_COLOR = "#e4e3df"
MAX_SERIES = len(PALETTE)
SAVE_DPI = 300               # resolution of figures saved from the viewer / CLI


@dataclass(frozen=True)
class Record:
    filename: str
    grp: str
    channel: int
    fr: float
    organoid: str   # e.g. CT7
    slice: str      # e.g. CT7A


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_organoid(filename: str, grp: str) -> tuple[str, str]:
    """Return (organoid, slice) parsed from a file name, or (UNKNOWN, UNKNOWN)."""
    marker = GROUP_MARKERS.get(grp.upper())
    markers = (marker,) if marker else ALL_MARKERS
    pattern = r"(?<![A-Za-z])(%s)(\d+)([A-Za-z])(?![A-Za-z])" % "|".join(markers)
    m = re.search(pattern, filename)
    if not m:
        return UNKNOWN, UNKNOWN
    prefix, number, letter = m.group(1), m.group(2), m.group(3).upper()
    return f"{prefix}{number}", f"{prefix}{number}{letter}"


def _find_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map the canonical names to the actual header spellings (case-insensitive)."""
    lookup = {name.strip().lstrip("﻿").lower(): name for name in fieldnames}
    missing = [c for c in (COL_FILENAME, COL_GRP, COL_CHANNEL, COL_FR) if c not in lookup]
    if missing:
        sys.exit(f"Error: CSV is missing required column(s): {', '.join(missing)}\n"
                 f"       Found columns: {', '.join(fieldnames)}")
    return {c: lookup[c] for c in (COL_FILENAME, COL_GRP, COL_CHANNEL, COL_FR)}


def load_records(csv_path: Path) -> list[Record]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = _find_columns(reader.fieldnames or [])
        rows = list(reader)

    records: list[Record] = []
    skipped_fr = 0
    skipped_channel = 0
    unknown_files: set[str] = set()

    for row in rows:
        filename = (row[cols[COL_FILENAME]] or "").strip()
        grp = (row[cols[COL_GRP]] or "").strip()
        try:
            fr = float(row[cols[COL_FR]])
            if math.isnan(fr):
                raise ValueError
        except (TypeError, ValueError):
            skipped_fr += 1
            continue
        try:
            channel = int(float(row[cols[COL_CHANNEL]]))
        except (TypeError, ValueError):
            skipped_channel += 1
            continue

        organoid, slc = parse_organoid(filename, grp)
        if organoid == UNKNOWN:
            unknown_files.add(filename)
        records.append(Record(filename, grp, channel, fr, organoid, slc))

    if skipped_fr:
        print(f"Note: skipped {skipped_fr} row(s) with missing/non-numeric FR.", file=sys.stderr)
    if skipped_channel:
        print(f"Note: skipped {skipped_channel} row(s) with non-numeric Channel.", file=sys.stderr)
    if unknown_files:
        print(f"Warning: could not parse an organoid ID from {len(unknown_files)} file name(s); "
              f"they are grouped under '{UNKNOWN}':", file=sys.stderr)
        for name in sorted(unknown_files):
            print(f"  {name}", file=sys.stderr)
    if not records:
        sys.exit("Error: no usable rows found in the CSV.")
    return records


# --------------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------------- #

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def groups(records: list[Record]) -> list[str]:
    return sorted({r.grp for r in records}, key=_natural_key)


def organoids(records: list[Record], grp: str) -> list[str]:
    return sorted({r.organoid for r in records if r.grp == grp}, key=_natural_key)


def select(records: list[Record], grp: str, organoid: str | None) -> list[Record]:
    """Records used for the *points*: the group, narrowed to one organoid if given."""
    sel = records if grp == ALL_GROUPS else [r for r in records if r.grp == grp]
    if organoid and organoid != ALL_ORGANOIDS:
        sel = [r for r in sel if r.organoid == organoid]
    return sel


def box_records(records: list[Record], grp: str) -> list[Record]:
    """Records used for the *boxes*: always the whole group (or the whole file)."""
    return select(records, grp, None)


def describe(records: list[Record]) -> None:
    for grp in groups(records):
        in_grp = [r for r in records if r.grp == grp]
        print(f"{grp}: {len({r.filename for r in in_grp})} recordings, "
              f"{len({r.channel for r in in_grp})} channels")
        for org in organoids(in_grp, grp):
            in_org = [r for r in in_grp if r.organoid == org]
            slices = sorted({r.slice for r in in_org}, key=_natural_key)
            print(f"  {org}: {len({r.filename for r in in_org})} recordings; "
                  f"slices: {', '.join(slices)}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _style_axes(ax, channels: list[int], positions: list[int]) -> None:
    ax.set_xlabel("Channel")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in channels],
                       rotation=90 if len(channels) > 30 else 0, fontsize=8)
    ax.set_xlim(-0.7, len(channels) - 0.3)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=BOX_COLOR)


def _legend(ax, title: str) -> None:
    ax.legend(title=title, frameon=False, fontsize=8, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))


class HoverTooltip:
    """Shows which recording a plotted point came from when the mouse is over it.

    `draw()` registers every scatter artist it creates together with the
    records behind its points (same order). The viewer connects `on_move` to
    the canvas' motion events.
    """

    def __init__(self, ax):
        self.ax = ax
        self.targets: list[tuple[object, list[Record]]] = []
        self.annot = None
        self._current: tuple[int, int] | None = None

    def reset(self) -> None:
        """Call after `ax.clear()`: the old annotation is gone with the axes."""
        self.targets.clear()
        self._current = None
        self.annot = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            fontsize=8, color="#0b0b0b", zorder=10, annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BOX_COLOR, lw=0.8),
        )
        self.annot.set_visible(False)

    def add(self, artist, recs: list[Record]) -> None:
        self.targets.append((artist, recs))

    @staticmethod
    def text_for(r: Record) -> str:
        return (f"{r.grp} \u00b7 {r.slice}\n"
                f"Channel {r.channel}\nFR = {r.fr:.4g} Hz")

    def on_move(self, event) -> None:
        if self.annot is None:
            return
        hit = None
        if event.inaxes is self.ax:
            for t_idx, (artist, recs) in enumerate(self.targets):
                found, info = artist.contains(event)
                if found and len(info["ind"]):
                    hit = (t_idx, int(info["ind"][0]))
                    break
        if hit == self._current:
            return
        self._current = hit
        if hit is None:
            self.annot.set_visible(False)
        else:
            artist, recs = self.targets[hit[0]]
            x, y = artist.get_offsets()[hit[1]]
            self.annot.xy = (x, y)
            self.annot.set_text(self.text_for(recs[hit[1]]))
            # Flip the label to the left near the right edge so it stays readable.
            lo, hi = self.ax.get_xlim()
            on_right = x > lo + 0.7 * (hi - lo)
            self.annot.set_x(-12 if on_right else 12)
            self.annot.set_ha("right" if on_right else "left")
            self.annot.set_visible(True)
        event.canvas.draw_idle()


def _outliers(recs: list[Record], by_channel: dict[int, list[float]]) -> list[Record]:
    """Records outside the whiskers, using the same rule as ax.boxplot (1.5 x IQR)."""
    from matplotlib import cbook
    bounds = {}
    for c, vals in by_channel.items():
        st = cbook.boxplot_stats(vals, whis=1.5)[0]
        bounds[c] = (st["whislo"], st["whishi"])
    return [r for r in recs if not bounds[r.channel][0] <= r.fr <= bounds[r.channel][1]]


def _draw_boxes(ax, by_channel: dict[int, list[float]], channels: list[int],
                positions: list[int]) -> None:
    """Boxes and whiskers only. Outliers are drawn separately (see _draw_outliers)
    so that each dot stays tied to its source row for the hover tooltip."""
    ax.boxplot(
        [by_channel[c] for c in channels], positions=positions, widths=0.55,
        showfliers=False,
        boxprops=dict(color=BOX_COLOR, linewidth=1),
        whiskerprops=dict(color=BOX_COLOR, linewidth=1),
        capprops=dict(color=BOX_COLOR, linewidth=1),
        medianprops=dict(color=BOX_COLOR, linewidth=1.6),
    )


def _draw_outliers(ax, recs: list[Record], by_channel: dict[int, list[float]],
                   pos_of: dict[int, int], hover: HoverTooltip | None) -> None:
    """Grey dots for readings beyond 1.5 x IQR of their channel's box."""
    outs = _outliers(recs, by_channel)
    if not outs:
        return
    sc = ax.scatter([pos_of[r.channel] for r in outs], [r.fr for r in outs],
                    s=10, color=BOX_COLOR, alpha=0.6, linewidths=0, zorder=3)
    if hover:
        hover.add(sc, outs)


def draw(ax, records: list[Record], grp: str, organoid: str | None,
         show_points: bool = True, hover: HoverTooltip | None = None) -> None:
    """Draw per-channel FR box plots for the selection onto `ax`.

    Boxes always summarise the whole group (`grp`), or the whole file when
    `grp` is ALL_GROUPS. Selecting an organoid only changes which readings are
    overlaid as points (that organoid's, coloured by slice). When `hover` is
    given, every plotted point is registered with it for the tooltip.
    """
    ax.clear()
    ax.set_axis_on()
    if hover:
        hover.reset()
    at_organoid_level = bool(organoid) and organoid != ALL_ORGANOIDS

    boxes = box_records(records, grp)
    points = select(records, grp, organoid)
    if not boxes:
        ax.set_title(f"Firing rate per channel: {grp}")
        ax.text(0.5, 0.5, "No data for this selection", ha="center", va="center",
                transform=ax.transAxes, color=BOX_COLOR)
        ax.set_axis_off()
        return

    by_channel: dict[int, list[float]] = defaultdict(list)
    for r in boxes:
        by_channel[r.channel].append(r.fr)
    channels = sorted(by_channel)
    positions = list(range(len(channels)))
    pos_of = {c: p for c, p in zip(channels, positions)}
    n_files = len({r.filename for r in boxes})
    n_label = f"n = {n_files} recording{'s' if n_files != 1 else ''}"

    # ---- Channel overview: all groups, one box per channel ------------------
    # Grey dots are the box plot's outliers (readings more than 1.5 x IQR
    # beyond the box edges); hover over one in the viewer to see its recording.
    if grp == ALL_GROUPS:
        _draw_boxes(ax, by_channel, channels, positions)
        _draw_outliers(ax, boxes, by_channel, pos_of, hover)
        ax.set_title(f"Firing rate per channel: all groups   ({n_label})",
                     fontsize=11, loc="left", color="#0b0b0b")
        _style_axes(ax, channels, positions)
        return

    # ---- Group view (optionally highlighting one organoid) ------------------
    _draw_boxes(ax, by_channel, channels, positions)
    if not show_points:
        _draw_outliers(ax, boxes, by_channel, pos_of, hover)

    if show_points:
        # Points are coloured by organoid when the whole group is shown and by
        # slice when one organoid is highlighted. Series beyond the palette
        # fold into "Other". A legend is always drawn, even for one series.
        key = (lambda r: r.slice) if at_organoid_level else (lambda r: r.organoid)
        series = sorted({key(r) for r in points}, key=_natural_key)
        if len(series) > MAX_SERIES:
            named = series[:MAX_SERIES - 1]
            colors = {s: PALETTE[i] for i, s in enumerate(named)}
            label_of = lambda s: s if s in colors else "Other"
            color_of = lambda s: colors.get(s, OTHER_COLOR)
        else:
            colors = {s: PALETTE[i] for i, s in enumerate(series)}
            label_of = lambda s: s
            color_of = colors.__getitem__

        rng = random.Random(0)
        seen_labels: set[str] = set()
        for s in series:
            pts = [r for r in points if key(r) == s]
            xs = [pos_of[r.channel] + rng.uniform(-0.18, 0.18) for r in pts]
            ys = [r.fr for r in pts]
            label = label_of(s)
            sc = ax.scatter(xs, ys, s=14, color=color_of(s), alpha=0.75, linewidths=0,
                            zorder=3, label=None if label in seen_labels else label)
            seen_labels.add(label)
            if hover:
                hover.add(sc, pts)
        if seen_labels:
            _legend(ax, "Slice" if at_organoid_level else "Organoid")

    if at_organoid_level:
        title = f"Firing rate per channel: {grp} boxes, {organoid} readings highlighted"
    else:
        title = f"Firing rate per channel: {grp}"
    ax.set_title(f"{title}   ({n_label})", fontsize=11, loc="left", color="#0b0b0b")
    _style_axes(ax, channels, positions)


def figure_size(records: list[Record], grp: str, organoid: str | None) -> tuple[float, float]:
    n = len({r.channel for r in box_records(records, grp)})
    return (max(8.0, min(0.22 * n + 2.5, 20.0)), 5.0)


# --------------------------------------------------------------------------- #
# Interactive viewer
# --------------------------------------------------------------------------- #

def run_gui(records: list[Record], csv_path: Path, show_points: bool) -> None:
    import tkinter as tk
    from tkinter import ttk

    import matplotlib
    matplotlib.use("TkAgg")
    # Figures saved from the toolbar's save button come out at SAVE_DPI.
    # On-screen rendering stays at the display's own pixel density.
    matplotlib.rcParams["savefig.dpi"] = SAVE_DPI
    matplotlib.rcParams["savefig.bbox"] = "tight"
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    grps = [ALL_GROUPS] + groups(records)

    root = tk.Tk()
    root.title(f"Firing rate per channel: {csv_path.name}")

    controls = ttk.Frame(root, padding=(10, 8))
    controls.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(controls, text="Group:").pack(side=tk.LEFT)
    grp_var = tk.StringVar(value=grps[0])
    grp_box = ttk.Combobox(controls, textvariable=grp_var, values=grps, state="readonly", width=12)
    grp_box.pack(side=tk.LEFT, padx=(4, 16))

    ttk.Label(controls, text="Organoid:").pack(side=tk.LEFT)
    org_var = tk.StringVar(value=ALL_ORGANOIDS)
    org_box = ttk.Combobox(controls, textvariable=org_var, state="readonly", width=16)
    org_box.pack(side=tk.LEFT, padx=(4, 16))

    points_var = tk.BooleanVar(value=show_points)
    ttk.Checkbutton(controls, text="Show individual readings", variable=points_var).pack(side=tk.LEFT)

    status_var = tk.StringVar()
    ttk.Label(controls, textvariable=status_var, foreground="#52514e").pack(side=tk.RIGHT)

    fig = Figure(figsize=(12, 5.5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=root)
    NavigationToolbar2Tk(canvas, root).update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    hover = HoverTooltip(ax)
    canvas.mpl_connect("motion_notify_event", hover.on_move)

    def refresh(*_):
        grp, org = grp_var.get(), org_var.get()
        draw(ax, records, grp, org, points_var.get(), hover)
        fig.subplots_adjust(left=0.06, right=0.86, top=0.9, bottom=0.16)
        canvas.draw_idle()
        boxes, pts = box_records(records, grp), select(records, grp, org)
        msg = (f"boxes: {len({r.filename for r in boxes})} recordings, "
               f"{len({r.channel for r in boxes})} channels")
        if org != ALL_ORGANOIDS:
            msg += f"   |   {org}: {len({r.filename for r in pts})} recordings, {len(pts)} readings"
        status_var.set(msg)

    def on_group_change(*_):
        grp = grp_var.get()
        if grp == ALL_GROUPS:
            org_box["values"] = [ALL_ORGANOIDS]
            org_box.state(["disabled"])
        else:
            org_box["values"] = [ALL_ORGANOIDS] + organoids(records, grp)
            org_box.state(["!disabled"])
        org_var.set(ALL_ORGANOIDS)
        refresh()

    grp_box.bind("<<ComboboxSelected>>", on_group_change)
    org_box.bind("<<ComboboxSelected>>", refresh)
    points_var.trace_add("write", refresh)

    on_group_change()
    root.mainloop()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, help="CSV with FileName, grp, Channel and FR columns.")
    parser.add_argument("--list", action="store_true",
                        help="Print groups, organoids and slices found in the file, then exit.")
    parser.add_argument("--grp", help="Group to plot (e.g. BCTL), or 'all' for the channel overview "
                                      "across every group. Required with -o.")
    parser.add_argument("--organoid", help="Organoid whose readings to highlight on the group's boxes "
                                           "(e.g. CT7). Default: all organoids.")
    parser.add_argument("-o", "--output", type=Path,
                        help="Save the figure to this path (png/pdf/svg) instead of opening the viewer.")
    parser.add_argument("--no-points", action="store_true",
                        help="Do not overlay individual readings on the boxes (show outliers instead).")
    parser.add_argument("--dpi", type=int, default=SAVE_DPI,
                        help=f"Resolution for saved figures (default {SAVE_DPI}).")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        sys.exit(f"Error: input file '{args.input_csv}' does not exist.")
    records = load_records(args.input_csv)

    if args.list:
        describe(records)
        return

    if args.output is None and args.grp is None:
        run_gui(records, args.input_csv, show_points=not args.no_points)
        return

    grps = groups(records)
    if args.grp.lower() == "all":
        args.grp = ALL_GROUPS
    elif args.grp not in grps:
        sys.exit(f"Error: group '{args.grp}' not found. Available: all, {', '.join(grps)}")
    if args.grp == ALL_GROUPS and args.organoid:
        sys.exit("Error: --organoid cannot be combined with --grp all.")
    if args.organoid and args.organoid not in organoids(records, args.grp):
        sys.exit(f"Error: organoid '{args.organoid}' not found in {args.grp}. "
                 f"Available: {', '.join(organoids(records, args.grp))}")
    if args.output is None:
        sys.exit("Error: --grp/--organoid without -o has nothing to do; add -o OUTPUT to save a figure "
                 "or omit them to open the interactive viewer.")

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=figure_size(records, args.grp, args.organoid))
    ax = fig.add_subplot(111)
    draw(ax, records, args.grp, args.organoid, show_points=not args.no_points)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved '{args.output}'.")


if __name__ == "__main__":
    main()
