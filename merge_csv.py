#!/usr/bin/env python3
"""Merge MEA-NAP CSV exports that share a column layout and a recording run.

Written for combining the per-condition node-level exports of one run

Any number of CSVs can be given. Nothing is written until four checks pass:

    Columns     Every file must carry the same set of columns, compared without
                regard to case or order. The merged file uses the first file's
                spelling and column order.

    Run         Every row of every file must come from the same run, identified
                by the leading R<digits> token of its FileName, e.g. "R250929" in
                "R250929CT7A_DIV250_stim1". A file holding two runs, or a file
                from a different run than the others, stops the merge.

    Repeats     No recording may appear in more than one file. A FileName repeats
                within a file once per channel, which is expected, but the same
                FileName in two files means the inputs overlap and that
                recording's rows would be doubled.

    Recordings  No recording may be present twice under any guise: the same
                FileName and Channel on two rows (checked when there is a Channel
                column), or two FileNames that differ only in their DIV<n> token,
                e.g. "R250929CT1A_DIV250_base" and "R250929CT1A_DIV251_base" --
                the same slice and condition recorded twice, which leaves no way
                to tell which one a reading belongs with.

Rows are otherwise passed through untouched and in the order given: no
de-duplication, no reordering, no rewriting of values.

Usage:
    python3 merge_csv.py base.csv stim1.csv stim3.csv
    python3 merge_csv.py stimLR.csv stimRL.csv -o combined.csv

OUTPUT may be:
    omitted                   -> <first_input_dir>/NeuronalActivity_NodeLevel_base_stim_merged.csv
    a bare filename (x.csv)   -> <first_input_dir>/x.csv
    a directory               -> <dir>/NeuronalActivity_NodeLevel_base_stim_merged.csv
    a path ending in .csv     -> used as given

Exits 1 if fewer than two inputs are given, an input is missing or empty, the
columns disagree, the runs disagree, a recording appears in more than one
input, a recording is present twice, or the output would overwrite an input.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NAME = "NeuronalActivity_NodeLevel_base_stim_merged.csv"

# Column names are matched case-insensitively (the user's file may say "filename").
COL_FILENAME = "filename"
COL_CHANNEL = "channel"

MAX_LISTED = 10                    # offending names listed per error before "... and N more"


def normalize(column: str) -> str:
    return column.strip().lower()


def parse_run(filename: str) -> str | None:
    """Return "R250929" from "R250929CT7A_DIV250_stim1", or None.

    The organoid marker is fused onto the run number, so the run is the leading
    R<digits> token rather than everything before the first underscore.
    """
    m = re.match(r"R\d+", filename)
    return m.group() if m else None


@dataclass
class Table:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]

    def column(self, normalized: str) -> str:
        """This file's spelling of a normalized column name."""
        return next(c for c in self.fieldnames if normalize(c) == normalized)

    def aligned(self, header: list[str]) -> list[dict[str, str]]:
        """These rows, rekeyed to the merged header's spelling and column order."""
        source = [self.column(normalize(c)) for c in header]
        if source == self.fieldnames:
            return self.rows
        return [dict(zip(header, (row[c] for c in source))) for row in self.rows]


def read_table(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            sys.exit(f"Error: '{path}' is empty.")
        rows = list(reader)
    if not rows:
        sys.exit(f"Error: '{path}' has a header but no data rows.")
    return Table(path, reader.fieldnames, rows)


def check_headers(tables: list[Table]) -> list[str]:
    """Return the merged header (the first file's), or exit if the files disagree."""
    first = tables[0]
    expected = {normalize(c): c for c in first.fieldnames}
    if COL_FILENAME not in expected:
        sys.exit(f"Error: '{first.path}' has no 'FileName' column.\n"
                 f"       Found columns: {', '.join(first.fieldnames)}")

    for t in tables[1:]:
        found = {normalize(c): c for c in t.fieldnames}
        if set(found) == set(expected):
            continue
        lines = [f"Error: columns in '{t.path}' do not match those in '{first.path}'."]
        if missing := [c for k, c in expected.items() if k not in found]:
            lines.append(f"       Missing: {', '.join(missing)}")
        if extra := [c for k, c in found.items() if k not in expected]:
            lines.append(f"       Unexpected: {', '.join(extra)}")
        sys.exit("\n".join(lines))

    return first.fieldnames


def check_runs(tables: list[Table]) -> str:
    """Return the run ID shared by every row of every file, or exit if they differ."""
    run_of: dict[Path, str] = {}
    unparsed_of: dict[Path, set[str]] = {}

    for t in tables:
        col = t.column(COL_FILENAME)
        runs: dict[str, str] = {}          # run ID -> first file name carrying it
        unparsed: set[str] = set()
        for row in t.rows:
            name = (row[col] or "").strip()
            if run := parse_run(name):
                runs.setdefault(run, name)
            else:
                unparsed.add(name)

        if not runs:
            sys.exit(f"Error: no FileName in '{t.path}' carries a run ID "
                     f"(expected a leading R<digits>, e.g. 'R250929CT7A_DIV250_base').")
        if len(runs) > 1:
            lines = [f"Error: '{t.path}' holds recordings from {len(runs)} runs:"]
            lines += [f"       {run}, e.g. {name}" for run, name in sorted(runs.items())]
            sys.exit("\n".join(lines))
        run_of[t.path] = next(iter(runs))
        if unparsed:
            unparsed_of[t.path] = unparsed

    if len(set(run_of.values())) > 1:
        lines = ["Error: the input files are not from the same run."]
        lines += [f"       {path}: {run}" for path, run in run_of.items()]
        sys.exit("\n".join(lines))

    # Warn only once the merge is certain to go ahead, so the names listed here
    # really are the ones passed through without a run check.
    for path, names in unparsed_of.items():
        print(f"Warning: could not parse a run ID from {len(names)} file name(s) "
              f"in '{path}'; they are merged unchecked:", file=sys.stderr)
        for name in sorted(names):
            print(f"  {name}", file=sys.stderr)

    return next(iter(run_of.values()))


def check_duplicates(tables: list[Table]) -> None:
    """Exit if a recording appears in more than one file.

    A FileName repeats within a file once per channel, which is expected. The
    same FileName in two files means the inputs overlap, and merging them would
    double that recording's rows.
    """
    seen: dict[str, Path] = {}
    shared: dict[tuple[Path, Path], list[str]] = {}

    for t in tables:
        col = t.column(COL_FILENAME)
        names = {name for row in t.rows if (name := (row[col] or "").strip())}
        for name in sorted(names):
            if first := seen.get(name):
                shared.setdefault((first, t.path), []).append(name)
            else:
                seen[name] = t.path

    if shared:
        lines = ["Error: the same recording appears in more than one input file."]
        for (first, second), names in shared.items():
            lines.append(f"       '{first}' and '{second}' share {len(names)} recording(s):")
            lines += [f"         {name}" for name in names]
        sys.exit("\n".join(lines))


def recording_key(filename: str) -> str:
    """The file name with its DIV<n> token blanked out.

    Two file names that agree on this name the same slice under the same
    condition, recorded on different days.
    """
    return re.sub(r"DIV\d*", "DIV", filename)


def check_recordings(tables: list[Table]) -> None:
    """Exit if a recording is present twice, within one file or across several.

    Two forms are caught: a FileName + Channel pair on more than one row, and
    two FileNames that differ only in their DIV<n> token. Either way a reader
    downstream would have two readings for one channel of one recording.
    """
    repeated: dict[tuple[str, str], int] = {}       # (FileName, Channel) -> rows
    names_of: dict[str, set[str]] = {}              # recording_key -> FileNames

    for t in tables:
        name_col = t.column(COL_FILENAME)
        has_channel = any(normalize(c) == COL_CHANNEL for c in t.fieldnames)
        channel_col = t.column(COL_CHANNEL) if has_channel else None
        for row in t.rows:
            name = (row[name_col] or "").strip()
            if not name:
                continue
            names_of.setdefault(recording_key(name), set()).add(name)
            if channel_col is not None:
                key = (name, (row[channel_col] or "").strip())
                repeated[key] = repeated.get(key, 0) + 1

    lines: list[str] = []
    if twice := sorted(k for k, n in repeated.items() if n > 1):
        lines.append(f"Error: {len(twice)} channel(s) appear on more than one row "
                     f"of the same recording:")
        lines += [f"         {name}, channel {ch}" for name, ch in twice[:MAX_LISTED]]
        if len(twice) > MAX_LISTED:
            lines.append(f"         ... and {len(twice) - MAX_LISTED} more")
    if clashes := sorted(sorted(v) for v in names_of.values() if len(v) > 1):
        lines.append(f"Error: {len(clashes)} recording(s) are present under more than "
                     f"one DIV (same slice and condition):")
        lines += [f"         {' / '.join(names)}" for names in clashes[:MAX_LISTED]]
        if len(clashes) > MAX_LISTED:
            lines.append(f"         ... and {len(clashes) - MAX_LISTED} more")
    if lines:
        sys.exit("\n".join(lines))


def resolve_output_path(inputs: list[Path], output: Path | None) -> Path:
    if output is None:
        path = inputs[0].parent / DEFAULT_NAME
    elif output.suffix.lower() != ".csv":
        path = output / DEFAULT_NAME
    elif output.parent == Path("."):
        path = inputs[0].parent / output.name
    else:
        path = output

    for source in inputs:
        if path.resolve() == source.resolve():
            sys.exit(f"Error: output '{path}' would overwrite the input file '{source}'.")
    return path


def write_merged(output: Path, header: list[str], tables: list[Table]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t in tables:
            writer.writerows(t.aligned(header))
    return sum(len(t.rows) for t in tables)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csvs", type=Path, nargs="+",
                        help="Two or more CSVs with identical columns, from one run.")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output file, directory, or bare filename (see above).")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.input_csvs) < 2:
        sys.exit("Error: give at least two CSV files to merge.")
    given: set[Path] = set()
    for path in args.input_csvs:
        if not path.is_file():
            sys.exit(f"Error: input file '{path}' does not exist.")
        if (resolved := path.resolve()) in given:
            sys.exit(f"Error: input file '{path}' was given more than once.")
        given.add(resolved)
    output_path = resolve_output_path(args.input_csvs, args.output)

    tables = [read_table(p) for p in args.input_csvs]
    header = check_headers(tables)
    run = check_runs(tables)
    check_duplicates(tables)
    check_recordings(tables)

    print(f"Merging {len(tables)} files from run {run}:")
    for t in tables:
        print(f"  {t.path.name}: {len(t.rows)} row(s)")
    written = write_merged(output_path, header, tables)
    print(f"Wrote {written} rows to '{output_path}'.")


if __name__ == "__main__":
    main()
