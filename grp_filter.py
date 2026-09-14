#!/usr/bin/env python3
"""Drop rows whose "Grp" value starts with "C" (e.g. CCTL, CMOS, CMUT) from a CSV.

The input file is never modified. The rows to be dropped are listed and
confirmation is requested before the remaining rows are written to a new CSV.

Usage:
    python3 grp_filter.py input.csv [-o OUTPUT]

OUTPUT may be:
    omitted                   -> <input_dir>/<input_stem>_filtered.csv
    a bare filename (x.csv)   -> <input_dir>/x.csv
    a directory               -> <dir>/<input_stem>_filtered.csv
    a path ending in .csv     -> used as given

Exits 1 if the input is missing, has no "Grp" column, or the output path
would overwrite the input.
"""

import argparse
import csv
import sys
from pathlib import Path

PURGE_PREFIX = "C"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_csv", type=Path, help="Source CSV file.")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file, directory, or bare filename (see above).",
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


def purged(row: dict) -> bool:
    return (row["Grp"] or "").startswith(PURGE_PREFIX)


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(r[c] or "") for r in rows)) for c in columns}
    lines = [columns, ["-" * widths[c] for c in columns]]
    lines += [[r[c] or "" for c in columns] for r in rows]
    for cells in lines:
        print("  ".join(f"{v:<{widths[c]}}" for c, v in zip(columns, cells)).rstrip())


def main():
    args = parse_args()
    if not args.input_csv.is_file():
        sys.exit(f"Error: input file '{args.input_csv}' does not exist.")
    output_path = resolve_output_path(args.input_csv, args.output)

    with args.input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "Grp" not in fieldnames:
            sys.exit("Error: input CSV has no 'Grp' column.")
        rows = list(reader)

    purge = [r for r in rows if purged(r)]
    keep = [r for r in rows if not purged(r)]

    print(f"Rows to be purged: {len(purge)}")
    if purge:
        print_table(purge, [c for c in ("FileName", "Grp") if c in fieldnames])

    if input("Write filtered CSV? (yes/no): ").strip().lower() not in ("yes", "y"):
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
