#!/usr/bin/env python3
"""Percentage change in firing rate (FR) from baseline, per channel, from a MEA-NAP CSV.

Each row of the CSV holds one (FileName, Grp, Channel, FR) observation. A slice
recorded under stimulation is compared against the same slice's own baseline
recording, channel by channel:

    percentage difference = 100 x (stim FR - base FR) / base FR

Pairing
-------
The condition is the token after "DIV<n>_" in the file name: "base" for the
baseline and "stim1", "stim3", "stimLR", "stimRL" for the stimulation patterns.
A stim recording pairs with a base recording when the run ID *and* the slice
match, so "R250929CT1A_DIV250_stim1" pairs with "R250929CT1A_DIV250_base" but
never with "R250929CT1B_DIV250_base" (different slice) or with
"R250930CT1A_DIV250_base" (different run).

Not every slice was recorded under stimulation. A slice with only a baseline
(or only stimulation) has nothing to compare, so no percentage is computed and
it gets no panel. Use --list to see which slices paired and which did not.

Grounded and stimulated channels
--------------------------------
Some channels report 0 Hz for reasons that have nothing to do with the organoid:
the electrode was grounded, or it was the one delivering the stimulation and so
recorded nothing while it fired. Both show up the same way in the export, either
as a baseline FR of 0, which leaves no percentage to compute, or as a baseline
that is fine while every stimulation recording reads 0, which comes out as a flat
-100%. Neither is a firing-rate change, so the channel is not plotted; its number
is printed in red along the x-axis of that slice's panel instead, and --list
names it.

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

List what paired and what did not, with per-slice channel counts:
    python3 fr_diff.py data.csv --list

Exits 1 if no slice has both a baseline and a stimulation recording.

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

BASE = "base"                      # the condition token every stim is measured against

# Condition token -> the name shown in the legend.
STIM_LABELS = {"stim1": "Stim 1", "stim3": "Stim 3",
               "stimLR": "Stim LR", "stimRL": "Stim RL"}

# Identity colours for the four known patterns; anything else the file happens
# to hold takes the remaining palette entries in the order it is encountered.
STIM_COLORS = dict(zip(STIM_LABELS, PALETTE))
EXTRA_COLORS = PALETTE[len(STIM_LABELS):]

GROUNDED_COLOR = "#e34948"         # grounded channel numbers on the x axis, nothing else
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


@dataclass(frozen=True)
class Grounded:
    """A channel whose electrode recorded nothing. See is_grounded().

    `readings` holds what each stimulation pattern recorded on that channel. The
    viewer plots none of it; --list names the channel so the numbers can be
    looked at when they matter.
    """
    channel: int
    base_fr: float                              # 0 when that is what grounded it
    readings: tuple[tuple[str, float], ...]     # ((token, stim_fr), ...), never empty


@dataclass
class Panel:
    """One slice that has a baseline *and* at least one stimulation recording."""
    run: str             # e.g. R250929
    slice: str           # e.g. CT1A
    organoid: str        # e.g. CT1
    grp: str
    base_file: str
    stim_files: dict[str, str]          # condition token -> FileName
    diffs: list[Diff] = field(default_factory=list)
    grounded: list[Grounded] = field(default_factory=list)
    missing_base: list[int] = field(default_factory=list)   # in a stim, absent from base

    @property
    def id(self) -> str:
        return f"{self.run}/{self.slice}"

    @property
    def stims(self) -> list[str]:
        return sorted(self.stim_files, key=_natural_key)

    @property
    def channels(self) -> list[int]:
        return sorted({d.channel for d in self.diffs} | {g.channel for g in self.grounded})


@dataclass(frozen=True)
class Unpaired:
    """A slice that produced no panel, and why."""
    run: str
    slice: str
    conditions: tuple[str, ...]
    reason: str          # see the REASON_* constants


REASON_NO_STIM = "no stim recording"
REASON_NO_BASE = "no base recording"
REASON_UNPARSED = "condition not parsed"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_run(filename: str) -> str | None:
    """Return the run ID "R250929" from "R250929CT1A_DIV250_base", or None.

    The twin of merge_csv.parse_run. Half of the pairing key; the other half is
    the slice, which fr_boxplots.parse_organoid supplies.
    """
    m = re.match(r"(R\d+)", filename)
    return m.group(1) if m else None


def pct_diff(base_fr: float, stim_fr: float) -> float | None:
    """Percentage change from `base_fr` to `stim_fr`, or None if there is no answer.

    A baseline of 0 has no percentage to give: every change from it is infinite.
    Those channels are reported as Grounded instead of being dropped.
    """
    if base_fr <= 0:
        return None
    return 100.0 * (stim_fr - base_fr) / base_fr


def is_grounded(base_fr: float, readings: tuple[tuple[str, float], ...]) -> bool:
    """Whether a channel recorded nothing: it was grounded, or it stimulated.

    Two ways that shows up. The baseline is 0, which leaves no percentage to
    compute; or the baseline is fine but every stimulation recording reads 0,
    because that electrode was the one delivering the stimulation. Either way the
    number says more about the electrode than about the organoid, so the channel
    is reddened on the axis rather than plotted.
    """
    if base_fr == 0:
        return True
    return bool(readings) and all(fr == 0 for _, fr in readings)


def stim_label(token: str) -> str:
    """The legend name for a condition token; an unknown token names itself."""
    return STIM_LABELS.get(token, token)


def stim_color(token: str, index: int) -> str:
    """The colour for a condition token, fixed for the four known patterns."""
    if token in STIM_COLORS:
        return STIM_COLORS[token]
    return EXTRA_COLORS[index % len(EXTRA_COLORS)] if EXTRA_COLORS else GROUNDED_COLOR


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


def _bucket(records: list[Record]) -> tuple[dict, list[Unpaired]]:
    """Group usable records by (run, slice) -> condition -> channel -> (fr, filename).

    Records whose run, condition or slice could not be parsed cannot be paired;
    they are reported and left out.
    """
    by_key: dict[tuple[str, str], dict[str, dict[int, tuple[float, str]]]] = defaultdict(
        lambda: defaultdict(dict))
    unpaired: list[Unpaired] = []

    no_run: list[str] = []
    no_stim: list[str] = []
    no_slice: list[str] = []
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
        # Every unparseable name shares the slice UNKNOWN, so keeping them would
        # let two unrelated recordings pair with each other.
        if r.slice == UNKNOWN:
            no_slice.append(r.filename)
            continue
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
    if no_slice:
        _note(f"no organoid slice in {len(set(no_slice))} file name(s); "
              f"they cannot be paired:", list(set(no_slice)))
    if duplicates:
        _note(f"{len(duplicates)} duplicate row(s) for a channel already seen in the "
              f"same recording; the first value was kept:", list(set(duplicates)))

    # Collapse the per-file Unpaired rows so one recording is named once.
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for u in unpaired:
        key = (u.run, u.slice, u.conditions[0])
        if key not in seen:
            seen.add(key)
            deduped.append(u)
    return by_key, deduped


def build_panels(records: list[Record]) -> tuple[list[Panel], list[Unpaired]]:
    """Pair each slice's stim recordings against its own baseline.

    Returns the panels that can be plotted and, alongside them, the slices that
    could not be paired so --list can explain the gaps.
    """
    by_key, unpaired = _bucket(records)
    grp_of = {(parse_run(r.filename), r.slice): r.grp for r in records}
    organoid_of = {(parse_run(r.filename), r.slice): r.organoid for r in records}

    panels: list[Panel] = []
    unknown_tokens: set[str] = set()
    negative_base: list[str] = []
    mixed_div: list[str] = []

    for (run, slc) in sorted(by_key, key=lambda k: (_natural_key(k[0]), _natural_key(k[1]))):
        conditions = by_key[(run, slc)]
        stim_tokens = sorted((t for t in conditions if t != BASE), key=_natural_key)

        if BASE not in conditions:
            unpaired.append(Unpaired(run, slc, tuple(stim_tokens), REASON_NO_BASE))
            continue
        if not stim_tokens:
            unpaired.append(Unpaired(run, slc, (BASE,), REASON_NO_STIM))
            continue

        base_channels = conditions[BASE]
        base_file = next(iter(base_channels.values()))[1] if base_channels else ""
        stim_files = {t: next(iter(conditions[t].values()))[1] for t in stim_tokens}

        # The pairing rule the user specified is run + slice, so a pair whose DIV
        # tokens differ is still a pair -- but it is worth saying out loud.
        base_div = _div(base_file)
        for token, name in stim_files.items():
            if base_div and _div(name) and _div(name) != base_div:
                mixed_div.append(f"{run} {slc}: {base_div}_{BASE} with {_div(name)}_{token}")

        panel = Panel(run=run, slice=slc,
                      organoid=organoid_of.get((run, slc), UNKNOWN),
                      grp=grp_of.get((run, slc), ""),
                      base_file=base_file, stim_files=stim_files)

        unknown_tokens.update(t for t in stim_tokens if t not in STIM_LABELS)

        # Channel-major: whether a channel was grounded is a fact about the
        # channel across every condition, not about one stim recording.
        for channel in sorted({c for t in stim_tokens for c in conditions[t]}):
            readings = tuple((t, conditions[t][channel][0])
                             for t in stim_tokens if channel in conditions[t])
            if channel not in base_channels:
                panel.missing_base.append(channel)
                continue
            base_fr = base_channels[channel][0]
            if base_fr < 0:
                negative_base.append(base_channels[channel][1])
                continue
            if is_grounded(base_fr, readings):
                panel.grounded.append(Grounded(channel, base_fr, readings))
                continue
            for token, stim_fr in readings:
                panel.diffs.append(Diff(channel, token, pct_diff(base_fr, stim_fr),
                                        base_fr, stim_fr))

        panels.append(panel)

    if unknown_tokens:
        _note(f"condition(s) {', '.join(sorted(unknown_tokens))} are not one of "
              f"{', '.join(STIM_LABELS)}; they are plotted under their own name.")
    if negative_base:
        _note(f"{len(negative_base)} channel(s) have a negative baseline FR, which is not "
              f"a firing rate; they were skipped:", list(set(negative_base)))
    if mixed_div:
        _note("a pair spans two DIVs (run ID and slice still match, so it is a pair):",
              list(set(mixed_div)))
    return panels, unpaired


def all_stims(panels: list[Panel]) -> list[str]:
    """Every condition token present across the panels, in natural order."""
    return sorted({t for p in panels for t in p.stim_files}, key=_natural_key)


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
        if p.grounded:
            channels = ", ".join(str(g.channel) for g in p.grounded)
            counts.append(f"{len(p.grounded)} grounded or stimulated ({channels})")
        if p.missing_base:
            counts.append(f"{len(p.missing_base)} stim-only")
        print(f"{p.run} {p.slice}: {BASE} + {', '.join(p.stims)}"
              f"   |   {', '.join(counts)}")

    if unpaired:
        print("Not paired:")
        for u in sorted(unpaired, key=lambda u: (_natural_key(u.run), _natural_key(u.slice))):
            if u.reason == REASON_UNPARSED:
                print(f"  {u.run}: condition not parsed from the file name "
                      f"({u.conditions[0]})")
            else:
                print(f"  {u.run} {u.slice}: {', '.join(u.conditions)} only ({u.reason})")

    total = len(panels) + len(unpaired)
    print(f"{len(panels)} of {total} slice(s) paired.")


# --------------------------------------------------------------------------- #
# Interactive viewer
# --------------------------------------------------------------------------- #

# The viewer is one HTML page: the paired percentages computed above are inlined
# as JSON and the script in the page template only lays them out. The templates
# for every script live together in viewers/.
VIEWER_HTML_PATH = Path(__file__).parent / "viewers" / "fr_diff_viewer.html"


def build_payload(panels: list[Panel], csv_name: str) -> dict:
    multi = multi_run(panels)
    tokens = all_stims(panels)
    lo, hi = pct_range(panels)
    return {
        "csv": csv_name,
        "runs": sorted({p.run for p in panels}, key=_natural_key),
        "multiRun": multi,
        "stims": tokens,
        "labels": {t: stim_label(t) for t in tokens},
        "colors": {
            "stim": {t: stim_color(t, i) for i, t in enumerate(tokens)},
            "grounded": GROUNDED_COLOR, "grid": GRID_COLOR, "ink": INK_COLOR,
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
            "grounded": [g.channel for g in p.grounded],
            "missingBase": p.missing_base,
            "n": {"paired": len({d.channel for d in p.diffs}),
                  "grounded": len(p.grounded),
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
    sys.exit(f"Error: slice '{wanted}' has no paired panel. "
             f"Available: {', '.join(ids)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path,
                        help="CSV with FileName, Grp, Channel and FR columns, holding both "
                             "the _base and the _stim recordings.")
    parser.add_argument("--winpath", action="store_true",
                        help="Treat input_csv (and -o/--output, if given) as Windows paths, e.g. "
                             "'C:\\Users\\...' as pasted from File Explorer, and convert them to "
                             "their WSL equivalent ('/mnt/c/Users/...'). Requires running under WSL.")
    parser.add_argument("--list", action="store_true",
                        help="Print which slices paired and which did not, then exit.")
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
        sys.exit("Error: no slice has both a _base and a _stim recording, so there is "
                 "nothing to\n       compare. A pair needs the same run ID and the same "
                 "slice, e.g.\n       R250929CT1A_DIV250_base and R250929CT1A_DIV250_stim1."
                 "\n       Run with --list to see what the file holds.")

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
