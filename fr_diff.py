#!/usr/bin/env python3
"""Percentage change in firing rate (FR) from baseline, per channel, from a MEA-NAP CSV.

Each row of the CSV holds one (FileName, Grp, Channel, FR) observation. A slice
recorded under stimulation is compared against the same slice's own
pre-stimulation recording, channel by channel:

    percentage difference = 100 x (stim FR - baseline FR) / baseline FR

Pairing
-------
The condition is the token after "DIV<n>_" in the file name. The baseline is
"prestim"; the stimulation patterns are "stim1" (spatial 1), "stim3" (spatial
3), "stimLR" and "stimRL" (temporal). Any other condition, including the older
"base" recordings, is not part of this analysis and is ignored with a note.

A stim recording pairs with a prestim recording when the run ID *and* the slice
match, so "R250929CT1A_DIV250_stim1" pairs with "R250929CT1A_DIV250_prestim"
but never with "R250929CT1B_DIV250_prestim" (different slice) or with
"R250930CT1A_DIV250_prestim" (different run). The slice is the token fused onto
the run ID ("CT1A"); Grp plays no part in pairing.

Only complete experiments are plotted: a slice needs its prestim recording and
all four stimulation patterns. A slice missing any of them gets no panel. A
slice with a condition recorded twice (e.g. at two DIVs) gets no panel either,
since there is no telling which recording a reading belongs with; merge_csv.py
refuses to produce such a file in the first place. Use --list to see which
slices were plotted and why the rest were not.

Excluded channels
-----------------
Some readings say nothing about the organoid. A channel listed in
GROUNDED_CHANNELS is left out of every recording, and one listed under a pattern
in STIMULATED_CHANNELS is left out of that pattern only. A channel whose
baseline is 0 Hz has no percentage to compute and is left out too. A channel
left with no point to plot has its number printed in red along the x-axis of
that slice's panel, and --list names it with the reason.

A channel that falls silent under stimulation is *not* assumed to be the
stimulating electrode: it is plotted at -100%.

Usage
-----
Interactive viewer (default) opens in the web browser:
    python3 fr_diff.py NeuronalActivity_NodeLevel.csv

    One panel per slice, laid out as a grid so every slice can be scanned at
    once, with the "Slice" drop-down zooming into a single slice full width.
    Each panel plots the percentage difference (y) against channel (x), one
    colour per stimulation pattern. Clicking a legend entry hides that pattern
    in every panel at once. Hovering a dot shows the channel, the percentage and
    the two firing rates it came from. The camera button in the plot's toolbar
    saves the current view as a PNG at (about) 300 dpi.

    The viewer is one HTML file written to the system temp directory. It loads
    the Plotly.js charting library from cdn.plot.ly, so the first run on a
    machine needs an internet connection.

Write the viewer to a file (to share, or on a headless machine):
    python3 fr_diff.py data.csv -o diff.html
    python3 fr_diff.py data.csv --slice CT1A -o diff.html    # initial selection

List what was plotted and what was not, with per-slice channel counts:
    python3 fr_diff.py data.csv --list

Exits 1 if no slice has a complete set of recordings.

Needs fr_boxplots.py beside it (it shares that script's CSV reader and file-name
grammar). Unlike fr_boxplots.py it does not need matplotlib.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# The CSV reader, the file-name grammar and the browser plumbing are shared with
# fr_boxplots.py rather than copied: both scripts read the same export, and two
# drifting copies of parse_organoid() would silently mis-pair slices. Importing
# it costs nothing -- fr_boxplots imports matplotlib lazily, inside the two
# functions that need it, so none of it is pulled in here.
from fr_boxplots import (
    PALETTE,
    PLOTLY_JS_URL,
    SAVE_DPI,
    UNKNOWN,
    Record,
    _json_for_html,
    _natural_key,
    _open_in_browser,
    load_records,
    winpath_to_wsl,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASELINE = "prestim"               # the condition every stim is measured against

# Condition token -> the name shown in the legend. A slice needs the baseline
# and every one of these to be plotted; any other condition is ignored.
STIM_LABELS = {"stim1": "Spatial 1", "stim3": "Spatial 3",
               "stimLR": "Temporal LR", "stimRL": "Temporal RL"}
REQUIRED_STIMS = tuple(STIM_LABELS)
STIM_COLORS = dict(zip(STIM_LABELS, PALETTE))

# Electrodes whose readings are not the organoid's. A grounded channel is left
# out of every recording; a stimulating channel only out of the pattern that
# drove it (condition token -> channels). Channel 15 is grounded in every
# experiment; channels 21, 31, 41, 51, 61 and 71 were stimulated and left out of
# spike counting, so they read 0 Hz by design. They are listed under every
# pattern for now; a pattern that drives only some of them can list fewer.
GROUNDED_CHANNELS: frozenset[int] = frozenset({15})
_STIM_ELECTRODES = frozenset({21, 31, 41, 51, 61, 71})
STIMULATED_CHANNELS: dict[str, frozenset[int]] = {
    "stim1": _STIM_ELECTRODES, "stim3": _STIM_ELECTRODES,
    "stimLR": _STIM_ELECTRODES, "stimRL": _STIM_ELECTRODES,
}

EXCLUDED_COLOR = "#e34948"         # excluded channel numbers on the x axis, nothing else
GRID_COLOR = "#e4e3df"
INK_COLOR = "#52514e"
ZERO_LINE_COLOR = "#bdbcb8"        # the 0 % reference line inside each panel

ALL_SLICES = "All slices"

MAX_EXAMPLES = 5                   # file names listed per stderr note


@dataclass(frozen=True)
class Diff:
    """One channel's percentage change under one stimulation pattern."""
    channel: int
    stim: str            # condition token, e.g. stim1
    pct: float           # 100 * (stim_fr - base_fr) / base_fr
    base_fr: float
    stim_fr: float


EXCLUDED_GROUNDED = "grounded"
EXCLUDED_STIMULATING = "stimulating"
EXCLUDED_ZERO_BASE = "0 Hz baseline"


@dataclass(frozen=True)
class Excluded:
    """Readings of one channel left out of the plot, and why.

    `readings` holds what each stimulation pattern recorded on that channel and
    was set aside: every pattern for a grounded channel or a 0 Hz baseline, only
    the driving patterns for a stimulating one.
    """
    channel: int
    reason: str                                 # one of the EXCLUDED_* constants
    base_fr: float
    readings: tuple[tuple[str, float], ...]     # ((token, stim_fr), ...), never empty


@dataclass
class Panel:
    """One slice with a baseline and every stimulation pattern."""
    run: str             # e.g. R250929
    slice: str           # e.g. CT1A
    organoid: str        # e.g. CT1
    grp: str             # e.g. CTL, whatever prefix the export gave it
    base_file: str
    stim_files: dict[str, str]          # condition token -> FileName
    diffs: list[Diff] = field(default_factory=list)
    excluded: list[Excluded] = field(default_factory=list)
    missing_base: list[int] = field(default_factory=list)   # in a stim, absent from baseline
    missing_stim: list[int] = field(default_factory=list)   # in baseline, absent from a stim

    @property
    def id(self) -> str:
        return f"{self.run}/{self.slice}"

    @property
    def stims(self) -> list[str]:
        return sorted(self.stim_files, key=_natural_key)

    @property
    def channels(self) -> list[int]:
        return sorted({d.channel for d in self.diffs} | {e.channel for e in self.excluded})

    @property
    def unplotted(self) -> list[int]:
        """Excluded channels left with no point at all: the ones marked in red."""
        plotted = {d.channel for d in self.diffs}
        return sorted({e.channel for e in self.excluded} - plotted)


@dataclass(frozen=True)
class Unpaired:
    """A slice that produced no panel, and why."""
    run: str
    slice: str
    conditions: tuple[str, ...]
    reason: str          # see the REASON_* constants
    missing: tuple[str, ...] = ()       # the patterns an incomplete slice lacks


REASON_NO_STIM = "no stim recording"
REASON_NO_BASE = f"no {BASELINE} recording"
REASON_INCOMPLETE = "incomplete"
REASON_TWICE = "a condition recorded twice"
REASON_UNPARSED = "condition not parsed"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_run(filename: str) -> str | None:
    """Return the run ID "R250929" from "R250929CT1A_DIV250_prestim", or None.

    The twin of merge_csv.parse_run. Half of the pairing key; the other half is
    the slice, which fr_boxplots.parse_organoid supplies.
    """
    m = re.match(r"(R\d+)", filename)
    return m.group(1) if m else None


def group_of(grp: str) -> str:
    """The experimental group from a Grp value: its last three characters.

    Some exports prefix the group with a letter (BCTL, BMOS, BMUT), so "BCTL"
    and "CTL" are the same group.
    """
    return grp.strip().upper()[-3:]


def pct_diff(base_fr: float, stim_fr: float) -> float:
    """Percentage change from `base_fr` to `stim_fr`. `base_fr` must be above 0."""
    return 100.0 * (stim_fr - base_fr) / base_fr


def stim_label(token: str) -> str:
    """The legend name for a condition token."""
    return STIM_LABELS[token]


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def _note(message: str, examples: list[str] | None = None) -> None:
    """A stderr note, optionally naming a few of the file names it is about."""
    print(f"Note: {message}", file=sys.stderr)
    for name in sorted(examples or [])[:MAX_EXAMPLES]:
        print(f"  {name}", file=sys.stderr)
    if examples and len(examples) > MAX_EXAMPLES:
        print(f"  ... and {len(examples) - MAX_EXAMPLES} more", file=sys.stderr)


def _div(filename: str) -> str | None:
    m = re.search(r"(DIV\d+)_", filename)
    return m.group(1) if m else None


def _bucket(records: list[Record]) -> tuple[dict, dict, list[Unpaired]]:
    """Group usable records by (run, slice) -> condition -> channel -> (fr, filename).

    Also returns (run, slice, condition) -> the FileNames seen for it, so a
    condition recorded twice can be caught. Records whose run, condition or slice
    could not be parsed cannot be paired, and records of a condition outside this
    analysis are not wanted; both are reported and left out.
    """
    by_key: dict[tuple[str, str], dict[str, dict[int, tuple[float, str]]]] = defaultdict(
        lambda: defaultdict(dict))
    files_of: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    unpaired: list[Unpaired] = []
    wanted = {BASELINE, *REQUIRED_STIMS}

    no_run: list[str] = []
    no_stim: list[str] = []
    no_slice: list[str] = []
    ignored: dict[str, set[str]] = defaultdict(set)     # token -> FileNames
    duplicates: list[str] = []

    for r in records:
        run = parse_run(r.filename)
        if run is None:
            no_run.append(r.filename)
            continue
        if r.stim is None:
            no_stim.append(r.filename)
            unpaired.append(Unpaired(run, r.slice, (r.filename,), REASON_UNPARSED))
            continue
        if r.stim not in wanted:
            ignored[r.stim].add(r.filename)
            continue
        # Every unparseable name shares the slice UNKNOWN, so keeping them would
        # let two unrelated recordings pair with each other.
        if r.slice == UNKNOWN:
            no_slice.append(r.filename)
            continue
        files_of[(run, r.slice, r.stim)].add(r.filename)
        channels = by_key[(run, r.slice)][r.stim]
        if r.channel in channels:
            duplicates.append(r.filename)
            continue
        channels[r.channel] = (r.fr, r.filename)

    if no_run:
        _note(f"no run ID in {len(set(no_run))} file name(s); they cannot be paired:",
              list(set(no_run)))
    if no_stim:
        _note(f"no condition after 'DIV<n>_' in {len(set(no_stim))} file name(s); "
              f"they cannot be paired:", list(set(no_stim)))
    if ignored:
        names = [n for files in ignored.values() for n in files]
        _note(f"{len(names)} recording(s) of condition(s) {', '.join(sorted(ignored))} "
              f"were ignored; this analysis uses {BASELINE} and "
              f"{', '.join(REQUIRED_STIMS)} only:", names)
    if no_slice:
        _note(f"no organoid slice in {len(set(no_slice))} file name(s); "
              f"they cannot be paired:", list(set(no_slice)))
    if duplicates:
        _note(f"{len(duplicates)} channel(s) appear on more than one row of the same "
              f"file name; the first value was kept:", list(set(duplicates)))

    # Collapse the per-file Unpaired rows so one recording is named once.
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for u in unpaired:
        key = (u.run, u.slice, u.conditions[0])
        if key not in seen:
            seen.add(key)
            deduped.append(u)
    return by_key, files_of, deduped


def _classify(panel: Panel, channel: int, base_fr: float,
              readings: tuple[tuple[str, float], ...]) -> None:
    """Add one channel's percentages, or its exclusion, to `panel`."""
    if channel in GROUNDED_CHANNELS:
        panel.excluded.append(Excluded(channel, EXCLUDED_GROUNDED, base_fr, readings))
        return
    # A known stimulating electrode is named as such before its baseline is
    # looked at: it reads 0 Hz by design, and "0 Hz baseline" is kept for the
    # zeros nothing explains.
    driving = tuple((t, fr) for t, fr in readings
                    if channel in STIMULATED_CHANNELS.get(t, ()))
    if driving:
        panel.excluded.append(Excluded(channel, EXCLUDED_STIMULATING, base_fr, driving))
    driven_by = {t for t, _ in driving}
    rest = tuple((t, fr) for t, fr in readings if t not in driven_by)
    if not rest:
        return
    # TODO: a 0 Hz baseline is taken to mean an electrode that recorded nothing;
    # whether any are really silent-then-recruited channels is still open (#3).
    if base_fr == 0:
        panel.excluded.append(Excluded(channel, EXCLUDED_ZERO_BASE, base_fr, rest))
        return
    for token, stim_fr in rest:
        panel.diffs.append(Diff(channel, token, pct_diff(base_fr, stim_fr),
                                base_fr, stim_fr))


def build_panels(records: list[Record]) -> tuple[list[Panel], list[Unpaired]]:
    """Pair each complete slice's stim recordings against its own baseline.

    Returns the panels that can be plotted and, alongside them, the slices that
    could not be so --list can explain the gaps.
    """
    by_key, files_of, unpaired = _bucket(records)
    grps_of: dict[tuple[str | None, str], set[str]] = defaultdict(set)
    organoid_of: dict[tuple[str | None, str], str] = {}
    for r in records:
        grps_of[(parse_run(r.filename), r.slice)].add(group_of(r.grp))
        organoid_of[(parse_run(r.filename), r.slice)] = r.organoid

    panels: list[Panel] = []
    negative_base: list[str] = []
    mixed_div: list[str] = []
    mixed_grp: list[str] = []

    for (run, slc) in sorted(by_key, key=lambda k: (_natural_key(k[0]), _natural_key(k[1]))):
        conditions = by_key[(run, slc)]
        present = [t for t in REQUIRED_STIMS if t in conditions]
        missing = tuple(t for t in REQUIRED_STIMS if t not in conditions)
        twice = [t for t in conditions if len(files_of[(run, slc, t)]) > 1]

        if twice:
            unpaired.append(Unpaired(run, slc, tuple(sorted(twice, key=_natural_key)),
                                     REASON_TWICE))
            continue
        if BASELINE not in conditions:
            unpaired.append(Unpaired(run, slc, tuple(present), REASON_NO_BASE))
            continue
        if not present:
            unpaired.append(Unpaired(run, slc, (BASELINE,), REASON_NO_STIM))
            continue
        if missing:
            unpaired.append(Unpaired(run, slc, (BASELINE, *present), REASON_INCOMPLETE,
                                     missing))
            continue

        base_channels = conditions[BASELINE]
        base_file = next(iter(files_of[(run, slc, BASELINE)]))
        stim_files = {t: next(iter(files_of[(run, slc, t)])) for t in REQUIRED_STIMS}

        # The pairing rule is run + slice, so a pair whose DIV tokens differ is
        # still a pair -- but it is worth saying out loud.
        base_div = _div(base_file)
        for token, name in stim_files.items():
            if base_div and _div(name) and _div(name) != base_div:
                mixed_div.append(f"{run} {slc}: {base_div}_{BASELINE} with {_div(name)}_{token}")

        grps = grps_of.get((run, slc), set())
        if len(grps) > 1:
            mixed_grp.append(f"{run} {slc}: {', '.join(sorted(grps))}")

        panel = Panel(run=run, slice=slc,
                      organoid=organoid_of.get((run, slc), UNKNOWN),
                      grp="/".join(sorted(grps)),
                      base_file=base_file, stim_files=stim_files)

        # Channel-major: whether a channel is excluded is a fact about the
        # channel across every condition, not about one stim recording.
        stim_channels = {c for t in REQUIRED_STIMS for c in conditions[t]}
        for channel in sorted(stim_channels | set(base_channels)):
            if channel not in base_channels:
                panel.missing_base.append(channel)
                continue
            readings = tuple((t, conditions[t][channel][0])
                             for t in REQUIRED_STIMS if channel in conditions[t])
            if len(readings) < len(REQUIRED_STIMS):
                panel.missing_stim.append(channel)
            if not readings:
                continue
            base_fr = base_channels[channel][0]
            if base_fr < 0:
                negative_base.append(base_channels[channel][1])
                continue
            _classify(panel, channel, base_fr, readings)

        panels.append(panel)

    if negative_base:
        _note(f"{len(negative_base)} channel(s) have a negative baseline FR, which is not "
              f"a firing rate; they were skipped:", list(set(negative_base)))
    if mixed_div:
        _note("a pair spans two DIVs (run ID and slice still match, so it is a pair):",
              list(set(mixed_div)))
    if mixed_grp:
        _note("a slice's recordings disagree on its group (compared on the last three "
              "characters of Grp):", mixed_grp)
    return panels, unpaired


def pct_range(panels: list[Panel]) -> tuple[float, float]:
    """The smallest and largest percentage across every panel (0 included)."""
    values = [d.pct for p in panels for d in p.diffs]
    if not values:
        return (-100.0, 100.0)
    return (min(0.0, min(values)), max(0.0, max(values)))


def multi_run(panels: list[Panel]) -> bool:
    return len({p.run for p in panels}) > 1


def panel_title(panel: Panel, multi: bool) -> str:
    """"CT1A", or "R250929 CT1A" when the file holds more than one run."""
    return f"{panel.run} {panel.slice}" if multi else panel.slice


def describe(panels: list[Panel], unpaired: list[Unpaired]) -> None:
    """Print what paired, what did not, and why.

    This is a diagnostic listing, so it always names the run: the whole point is
    to explain why two recordings that look alike did not pair.
    """
    for p in panels:
        counts = [f"{len({d.channel for d in p.diffs})} channels paired"]
        if p.excluded:
            named = ", ".join(f"{e.channel} ({e.reason}"
                              + (f": {', '.join(t for t, _ in e.readings)})"
                                 if e.reason == EXCLUDED_STIMULATING else ")")
                              for e in p.excluded)
            counts.append(f"{len(p.excluded)} excluded: {named}")
        if p.missing_base:
            counts.append(f"{len(p.missing_base)} stim-only")
        if p.missing_stim:
            counts.append(f"{len(p.missing_stim)} missing from a stim recording")
        print(f"{p.run} {p.slice}: {BASELINE} + {', '.join(p.stims)}"
              f"   |   {', '.join(counts)}")

    if unpaired:
        print("Not plotted:")
        for u in sorted(unpaired, key=lambda u: (_natural_key(u.run), _natural_key(u.slice))):
            if u.reason == REASON_UNPARSED:
                print(f"  {u.run}: condition not parsed from the file name "
                      f"({u.conditions[0]})")
            elif u.reason == REASON_INCOMPLETE:
                print(f"  {u.run} {u.slice}: {', '.join(u.conditions)} "
                      f"(incomplete: no {', '.join(u.missing)})")
            elif u.reason == REASON_TWICE:
                print(f"  {u.run} {u.slice}: {', '.join(u.conditions)} recorded more than "
                      f"once (e.g. at two DIVs)")
            else:
                print(f"  {u.run} {u.slice}: {', '.join(u.conditions)} only ({u.reason})")

    total = len(panels) + len(unpaired)
    print(f"{len(panels)} of {total} slice(s) plotted.")


# --------------------------------------------------------------------------- #
# Interactive viewer
# --------------------------------------------------------------------------- #

# The viewer is one HTML page: the paired percentages computed above are inlined
# as JSON and the script in the page template only lays them out. The templates
# for every script live together in viewers/.
VIEWER_HTML_PATH = Path(__file__).parent / "viewers" / "fr_diff_viewer.html"


def build_payload(panels: list[Panel], csv_name: str) -> dict:
    multi = multi_run(panels)
    lo, hi = pct_range(panels)
    return {
        "csv": csv_name,
        "runs": sorted({p.run for p in panels}, key=_natural_key),
        "multiRun": multi,
        "stims": list(REQUIRED_STIMS),
        "labels": {t: stim_label(t) for t in REQUIRED_STIMS},
        "colors": {
            "stim": dict(STIM_COLORS),
            "excluded": EXCLUDED_COLOR, "grid": GRID_COLOR, "ink": INK_COLOR,
            "zeroLine": ZERO_LINE_COLOR,
        },
        "range": {"min": lo, "max": hi},
        "text": {"allSlices": ALL_SLICES,
                 "y": "Firing rate difference from baseline (%)",
                 "x": "Channel"},
        "panels": [{
            "id": p.id, "title": panel_title(p, multi),
            "run": p.run, "slice": p.slice, "organoid": p.organoid, "grp": p.grp,
            "baseFile": p.base_file,
            "stims": p.stims,
            "channels": p.channels,
            "points": [{"c": d.channel, "t": d.stim, "p": d.pct,
                        "b": d.base_fr, "s": d.stim_fr} for d in p.diffs],
            "excluded": p.unplotted,
            "missingBase": p.missing_base,
            "n": {"paired": len({d.channel for d in p.diffs}),
                  "excluded": len(p.unplotted),
                  "missingBase": len(p.missing_base)},
        } for p in panels],
    }


def initial_state(slice_: str | None = None, same_y: bool = True,
                  dpi: int = SAVE_DPI) -> dict:
    # Plotly's export scale multiplies CSS pixels, which browsers lay out at 96/inch.
    return {"slice": slice_ or ALL_SLICES, "sameY": same_y,
            "scale": round(dpi / 96, 2)}


def render_html(payload: dict, initial: dict) -> str:
    template = VIEWER_HTML_PATH.read_text(encoding="utf-8")
    return (template
            .replace("__PLOTLY_JS_URL__", PLOTLY_JS_URL)
            .replace("__PAYLOAD__", _json_for_html(payload))
            .replace("__INITIAL__", _json_for_html(initial)))


def open_viewer(panels: list[Panel], csv_path: Path, initial: dict) -> None:
    """Write one viewer holding every slice and every condition, and open it."""
    out = Path(tempfile.gettempdir()) / f"fr_diff_{csv_path.stem}.html"
    out.write_text(render_html(build_payload(panels, csv_path.name), initial),
                   encoding="utf-8")
    print(f"Viewer written to '{out}'; opening it in your browser.")
    _open_in_browser(out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def resolve_slice(panels: list[Panel], wanted: str) -> str:
    """Return the panel id for --slice, accepting "CT1A" or "R250929/CT1A"."""
    ids = [p.id for p in panels]
    if wanted in ids:
        return wanted
    matches = [p for p in panels if p.slice == wanted]
    if len(matches) == 1:
        return matches[0].id
    if len(matches) > 1:
        sys.exit(f"Error: slice '{wanted}' is in more than one run. "
                 f"Use one of: {', '.join(m.id for m in matches)}")
    sys.exit(f"Error: slice '{wanted}' has no plotted panel. "
             f"Available: {', '.join(ids)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path,
                        help="CSV with FileName, Grp, Channel and FR columns, holding the "
                             f"_{BASELINE} and every _stim recording.")
    parser.add_argument("--winpath", action="store_true",
                        help="Treat input_csv (and -o/--output, if given) as Windows paths, e.g. "
                             "'C:\\Users\\...' as pasted from File Explorer, and convert them to "
                             "their WSL equivalent ('/mnt/c/Users/...'). Requires running under WSL.")
    parser.add_argument("--list", action="store_true",
                        help="Print which slices were plotted and why the rest were not, "
                             "then exit.")
    parser.add_argument("--slice",
                        help="Slice to show on its own instead of the grid (e.g. CT1A, or "
                             "R250929/CT1A when the file holds several runs).")
    parser.add_argument("-o", "--output", type=Path,
                        help="Write the viewer to this .html path instead of opening it.")
    parser.add_argument("--per-panel-y", action="store_true",
                        help="Scale each panel's y-axis to its own data instead of sharing one "
                             "axis across every panel.")
    parser.add_argument("--dpi", type=int, default=SAVE_DPI,
                        help=f"Resolution of the viewer's PNG export (default {SAVE_DPI}).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.winpath:
        args.input_csv = winpath_to_wsl(args.input_csv)
        if args.output is not None:
            args.output = winpath_to_wsl(args.output)
    if not args.input_csv.is_file():
        sys.exit(f"Error: input file '{args.input_csv}' does not exist.")

    records = load_records(args.input_csv)
    panels, unpaired = build_panels(records)

    # --list is how an unpairable file gets diagnosed, so it stays usable when
    # nothing paired; rendering has nothing to draw and writes no file.
    if args.list:
        describe(panels, unpaired)
        return
    if not panels:
        sys.exit(f"Error: no slice has a _{BASELINE} recording and all of "
                 f"{', '.join('_' + t for t in REQUIRED_STIMS)},\n"
                 f"       so there is nothing to plot. A set needs the same run ID and "
                 f"the same slice, e.g.\n       R250929CT1A_DIV250_{BASELINE} and "
                 f"R250929CT1A_DIV250_stim1.\n       Run with --list to see what the "
                 f"file holds.")

    selected = resolve_slice(panels, args.slice) if args.slice else None
    initial = initial_state(selected, not args.per_panel_y, args.dpi)

    if args.output is None:
        open_viewer(panels, args.input_csv, initial)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(build_payload(panels, args.input_csv.name), initial), encoding="utf-8")
    print(f"Saved '{args.output}'.")


if __name__ == "__main__":
    main()
