#!/usr/bin/env python3
"""Keep or drop rows of a CSV by their "Grp" value.

One kind of selection is made per run:

    --grp           keep ONLY rows whose Grp is one of these values
    --drop          drop rows whose Grp is one of these values
    --drop-prefix   drop rows whose Grp starts with one of these prefixes

--grp is a keep list and --drop/--drop-prefix are drop lists, so --grp cannot be
combined with either. --drop and --drop-prefix may be combined, and the union of
the two is dropped. All three take several values, either comma-separated
(--drop a,b) or by repeating the flag (--drop a --drop b).

Matching ignores case and surrounding whitespace throughout. --grp and --drop
match the whole value, so "prestim" matches "PreStim" but not "prestim2"; only
--drop-prefix matches partially. The "Grp" column itself is found
case-insensitively, and a leading byte-order mark is ignored.

The input file is never modified. A per-Grp summary of what would be kept and
dropped is printed, along with the recordings being lost, and confirmation is
requested before the remaining rows are written to a new CSV.

Usage:
    python3 grp_filter.py input.csv --list
    python3 grp_filter.py input.csv --grp PreStim,PostStim
    python3 grp_filter.py input.csv --drop CCTL --drop-prefix B
    python3 grp_filter.py input.csv --drop-prefix C -o filtered.csv

--drop-prefix C reproduces this script's earlier behaviour of dropping CCTL,
CMOS and CMUT.

--list prints each distinct Grp value with its row count and exits, ignoring
every other flag; run it first to see what a file holds.

OUTPUT may be:
    omitted                   -> <input_dir>/<input_stem>_filtered.csv
    a bare filename (x.csv)   -> <input_dir>/x.csv
    a directory               -> <dir>/<input_stem>_filtered.csv
    a path ending in .csv     -> used as given

Exits 1 if the input is missing or holds no data rows, has no "Grp" column, no
selection flag is given, --grp is combined with a drop flag, the selection would
keep no rows at all, or the output path would overwrite the input.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# Column names are matched case-insensitively (the user's file may say "grp").
COL_GRP = "grp"
COL_FILENAME = "filename"

BLANK_LABEL = "(blank)"   # how an empty Grp is shown in the tables
MAX_DETAIL_ROWS = 20      # recordings named before the dropped listing is capped


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csv", type=Path, help="Source CSV file.")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file, directory, or bare filename (see above).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print each distinct Grp value with its row count, then exit. "
             "Every other flag is ignored.",
    )
    # append rather than nargs="+", which would swallow the positional:
    # "grp_filter.py --drop CTL data.csv" would read data.csv as a second value.
    parser.add_argument(
        "--grp", action="append", metavar="VALUE", default=None,
        help="Keep ONLY rows whose Grp is one of these values (whole value, "
             "case-insensitive). Comma-separated or repeated. Cannot be combined "
             "with --drop or --drop-prefix.",
    )
    parser.add_argument(
        "--drop", action="append", metavar="VALUE", default=None,
        help="Drop rows whose Grp is one of these values (whole value, "
             "case-insensitive). Comma-separated or repeated.",
    )
    parser.add_argument(
        "--drop-prefix", action="append", metavar="PREFIX", default=None,
        help="Drop rows whose Grp starts with one of these prefixes "
             "(case-insensitive). '--drop-prefix C' drops CCTL, CMOS and CMUT.",
    )
    return parser.parse_args()


def resolve_output_path(input_csv: Path, output: Path | None) -> Path:
    default_name = f"{input_csv.stem}_filtered.csv"
    if output is None:
        path = input_csv.parent / default_name
    elif output.suffix.lower() != ".csv":
        path = output / default_name
    elif output.parent == Path("."):
        path = input_csv.parent / output.name
    else:
        path = output

    if path.resolve() == input_csv.resolve():
        sys.exit(f"Error: output '{path}' would overwrite the input file.")
    return path


def normalize(value: str) -> str:
    return value.strip().lower()


def label(value: str) -> str:
    """How a Grp value is shown in a table; an empty one needs a stand-in."""
    return value or BLANK_LABEL


def find_column(fieldnames: list[str], normalized: str) -> str | None:
    """This file's spelling of a normalized column name, or None."""
    return next((c for c in fieldnames if normalize(c) == normalized), None)


def split_values(values: list[str] | None) -> list[str]:
    """Flatten repeated flags and comma lists into the spellings the user typed.

    Empty tokens are discarded, so "--grp a,,b" is just a and b. Duplicates
    collapse case-insensitively, keeping the first spelling seen so that any
    warning echoes what was typed.
    """
    seen: dict[str, str] = {}
    for value in values or []:
        for token in value.split(","):
            token = token.strip()
            if token:
                seen.setdefault(normalize(token), token)
    return list(seen.values())


def make_dropper(keep_only: list[str], drop_exact: list[str], drop_prefixes: list[str]):
    """Return drop(grp) -> True when a row carrying this Grp should be dropped.

    The decision depends only on the Grp string, so every Grp value is entirely
    kept or entirely dropped. That is what makes the per-Grp summary printed
    before the prompt a complete account of the change rather than a digest.
    """
    if keep_only:
        wanted = {normalize(v) for v in keep_only}
        return lambda grp: normalize(grp) not in wanted

    exact = {normalize(v) for v in drop_exact}
    prefixes = tuple(normalize(v) for v in drop_prefixes)  # "x".startswith(()) is False

    def drop(grp: str) -> bool:
        value = normalize(grp)
        return value in exact or value.startswith(prefixes)

    return drop


def grp_counts(rows: list[dict], grp_col: str) -> dict[str, int]:
    """Distinct Grp value -> row count, ordered case-insensitively."""
    counts = Counter((row[grp_col] or "").strip() for row in rows)
    return {value: counts[value] for value in sorted(counts, key=str.lower)}


def warn_unmatched(keep_only, drop_exact, drop_prefixes, counts) -> None:
    """Warn about selectors matching no Grp in this file -- usually a typo.

    Not an error: sweeping one --drop list across a batch of files where some
    lack a group is a fair use, and the summary below makes the no-op plain.
    """
    present = [normalize(v) for v in counts]
    misses = []
    for flag, values, is_prefix in (("--grp", keep_only, False),
                                    ("--drop", drop_exact, False),
                                    ("--drop-prefix", drop_prefixes, True)):
        for value in values:
            target = normalize(value)
            hit = (any(p.startswith(target) for p in present) if is_prefix
                   else target in present)
            if not hit:
                misses.append(f"{flag} {value}")
    if misses:
        print(f"Warning: matched no Grp value in this file: {', '.join(misses)}\n"
              f"         Grp values present: {', '.join(label(v) for v in counts)}",
              file=sys.stderr)


def print_table(rows: list[dict], columns: list[str], limit: int | None = None) -> None:
    if not rows or not columns:
        return
    shown = rows if limit is None else rows[:limit]
    widths = {c: max([len(c), *(len(r[c] or "") for r in shown)]) for c in columns}
    lines = [columns, ["-" * widths[c] for c in columns]]
    lines += [[r[c] or "" for c in columns] for r in shown]
    for cells in lines:
        print("  ".join(f"{v:<{widths[c]}}" for c, v in zip(columns, cells)).rstrip())
    if len(shown) < len(rows):
        print(f"... and {len(rows) - len(shown)} more.")


def print_counts(counts: dict[str, int]) -> None:
    print_table([{"Grp": label(v), "Rows": str(n)} for v, n in counts.items()],
                ["Grp", "Rows"])
    print(f"\n{len(counts)} distinct Grp value(s), {sum(counts.values())} row(s).")


def print_summary(counts: dict[str, int], drop) -> None:
    print_table([{"Grp": label(v), "Rows": str(n),
                  "Action": "drop" if drop(v) else "keep"} for v, n in counts.items()],
                ["Grp", "Rows", "Action"])
    total = sum(counts.values())
    dropped_rows = sum(n for v, n in counts.items() if drop(v))
    dropped_grps = sum(1 for v in counts if drop(v))
    print(f"\nDropping {dropped_rows} of {total} rows "
          f"({dropped_grps} of {len(counts)} Grp value(s)); keeping {total - dropped_rows}.")


def dropped_recordings(rows, grp_col, filename_col, drop) -> list[dict]:
    """The distinct (FileName, Grp) pairs being dropped, in first-seen order.

    A recording contributes one row per channel, so listing rows would repeat
    each recording dozens of times over.
    """
    seen: dict[tuple[str, str], dict] = {}
    for row in rows:
        grp = row[grp_col] or ""
        if drop(grp):
            key = (row[filename_col] or "", grp.strip())
            seen.setdefault(key, {filename_col: key[0], grp_col: label(key[1])})
    return list(seen.values())


def confirm() -> bool:
    """Ask before writing. A closed stdin is a no, not a traceback."""
    try:
        answer = input("\nWrite filtered CSV? (yes/no): ")
    except EOFError:
        print()
        return False
    return answer.strip().lower() in ("yes", "y")


def check_selection(args, keep_only, drop_exact, drop_prefixes) -> None:
    """Exit unless exactly one kind of selection was asked for."""
    for flag, given, values in (("--grp", args.grp, keep_only),
                                ("--drop", args.drop, drop_exact),
                                ("--drop-prefix", args.drop_prefix, drop_prefixes)):
        if given is not None and not values:
            sys.exit(f"Error: {flag} was given no values.")

    if keep_only and (drop_exact or drop_prefixes):
        sys.exit("Error: --grp cannot be combined with --drop or --drop-prefix;\n"
                 "       --grp keeps only the values it names, while the drop flags\n"
                 "       remove the values they name.")

    if not (keep_only or drop_exact or drop_prefixes):
        sys.exit("Error: no selection given. Use --grp to keep only certain Grp values,\n"
                 "       --drop to remove them by name, or --drop-prefix to remove them\n"
                 "       by leading characters ('--drop-prefix C' drops CCTL, CMOS and\n"
                 "       CMUT). Use --list to see what this file holds.")


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        sys.exit(f"Error: input file '{args.input_csv}' does not exist.")

    keep_only = split_values(args.grp)
    drop_exact = split_values(args.drop)
    drop_prefixes = split_values(args.drop_prefix)
    if not args.list:
        check_selection(args, keep_only, drop_exact, drop_prefixes)

    with args.input_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        grp_col = find_column(fieldnames, COL_GRP)
        if grp_col is None:
            sys.exit("Error: input CSV has no 'Grp' column.\n"
                     f"       Found columns: {', '.join(fieldnames)}")
        rows = list(reader)
    if not rows:
        sys.exit(f"Error: '{args.input_csv}' has a header but no data rows.")

    counts = grp_counts(rows, grp_col)
    if args.list:
        print_counts(counts)
        return

    output_path = resolve_output_path(args.input_csv, args.output)
    warn_unmatched(keep_only, drop_exact, drop_prefixes, counts)

    drop = make_dropper(keep_only, drop_exact, drop_prefixes)
    keep = [r for r in rows if not drop(r[grp_col] or "")]
    if not keep:
        sys.exit("Error: the selection would keep no rows; nothing to write.\n"
                 f"       Grp values present: {', '.join(label(v) for v in counts)}")

    print_summary(counts, drop)
    filename_col = find_column(fieldnames, COL_FILENAME)
    if filename_col:
        recordings = dropped_recordings(rows, grp_col, filename_col, drop)
        if recordings:
            print("\nRecordings dropped:")
            print_table(recordings, [filename_col, grp_col], limit=MAX_DETAIL_ROWS)

    if not confirm():
        print("Aborted. No file was written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keep)
    print(f"Wrote {len(keep)} rows to '{output_path}'.")


if __name__ == "__main__":
    main()
