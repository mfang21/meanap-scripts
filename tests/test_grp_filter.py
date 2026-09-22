"""Run with: python3 -m unittest discover -t . -s tests"""

import csv
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grp_filter as gf


HEADER = "FileName,Grp,Channel,FR\n"

MIXED_CSV = HEADER + (
    "R250929CT1A_DIV250_base,CTL,1,0.5\n"
    "R250929CT1A_DIV250_base,CTL,2,1.5\n"
    "R250929CT2A_DIV250_base,CCTL,1,0.7\n"
    "R250929PS1A_DIV250_base,PreStim,1,2.0\n"
    "R250929PS1A_DIV250_base,PreStim,2,2.4\n"
    "R250929PS2A_DIV250_base,prestim2,1,3.0\n"
    "R250929MO1A_DIV250_base,MOS,1,4.0\n"
    "R250929XX1A_DIV250_base,,1,9.9\n"
)

SAMPLE_CSV = (Path(__file__).resolve().parent
              / "sample-files" / "NeuronalActivity_NodeLevel_sample.csv")


def write_csv(tmp, name="data.csv", text=MIXED_CSV):
    path = Path(tmp) / name
    path.write_text(text)
    return path


def read_rows(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def rows_of(text):
    """The CSV text as DictReader would yield it."""
    return list(csv.DictReader(text.splitlines()))


def captured(fn, *args, **kwargs):
    """Whatever fn prints to stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


class SplitValues(unittest.TestCase):
    def test_a_single_value(self):
        self.assertEqual(gf.split_values(["CTL"]), ["CTL"])

    def test_a_comma_separated_list(self):
        self.assertEqual(gf.split_values(["CTL,MOS"]), ["CTL", "MOS"])

    def test_a_repeated_flag(self):
        self.assertEqual(gf.split_values(["CTL", "MOS"]), ["CTL", "MOS"])

    def test_commas_and_repeats_combined(self):
        self.assertEqual(gf.split_values(["CTL,MOS", "MUT"]), ["CTL", "MOS", "MUT"])

    def test_surrounding_whitespace_is_stripped(self):
        self.assertEqual(gf.split_values([" CTL , MOS "]), ["CTL", "MOS"])

    def test_internal_whitespace_is_preserved(self):
        self.assertEqual(gf.split_values(["Group A"]), ["Group A"])

    def test_empty_tokens_are_discarded(self):
        self.assertEqual(gf.split_values(["CTL,,MOS"]), ["CTL", "MOS"])

    def test_a_flag_of_only_empty_tokens_yields_nothing(self):
        self.assertEqual(gf.split_values([" "]), [])

    def test_an_absent_flag_yields_nothing(self):
        self.assertEqual(gf.split_values(None), [])

    def test_duplicates_collapse_keeping_the_first_spelling(self):
        self.assertEqual(gf.split_values(["CTL", "ctl"]), ["CTL"])


class MakeDropper(unittest.TestCase):
    def test_grp_keeps_only_the_named_values(self):
        drop = gf.make_dropper(["PreStim"], [], [])
        self.assertFalse(drop("PreStim"))
        self.assertTrue(drop("CTL"))

    def test_grp_ignores_case(self):
        drop = gf.make_dropper(["prestim"], [], [])
        self.assertFalse(drop("PreStim"))

    def test_grp_does_not_match_a_longer_value(self):
        drop = gf.make_dropper(["PreStim"], [], [])
        self.assertTrue(drop("prestim2"))

    def test_grp_keeps_several_named_values(self):
        drop = gf.make_dropper(["CTL", "MOS"], [], [])
        self.assertFalse(drop("CTL"))
        self.assertFalse(drop("MOS"))
        self.assertTrue(drop("MUT"))

    def test_drop_removes_the_named_values(self):
        drop = gf.make_dropper([], ["CTL"], [])
        self.assertTrue(drop("CTL"))
        self.assertFalse(drop("MOS"))

    def test_drop_is_exact_rather_than_a_prefix(self):
        drop = gf.make_dropper([], ["C"], [])
        self.assertFalse(drop("CTL"))

    def test_drop_ignores_case(self):
        drop = gf.make_dropper([], ["ctl"], [])
        self.assertTrue(drop("CTL"))

    def test_drop_prefix_removes_by_leading_characters(self):
        drop = gf.make_dropper([], [], ["C"])
        self.assertTrue(drop("CCTL"))
        self.assertTrue(drop("CTL"))
        self.assertFalse(drop("MOS"))

    def test_drop_prefix_ignores_case(self):
        drop = gf.make_dropper([], [], ["c"])
        self.assertTrue(drop("CCTL"))

    def test_the_two_drop_flags_remove_the_union(self):
        drop = gf.make_dropper([], ["MOS"], ["C"])
        self.assertTrue(drop("CCTL"))
        self.assertTrue(drop("MOS"))
        self.assertFalse(drop("MUT"))

    def test_no_prefixes_at_all_drops_nothing(self):
        drop = gf.make_dropper([], [], [])
        self.assertFalse(drop("CTL"))

    def test_surrounding_whitespace_in_the_value_is_ignored(self):
        drop = gf.make_dropper([], ["CTL"], [])
        self.assertTrue(drop("  CTL  "))

    def test_a_blank_grp_is_dropped_by_grp_and_kept_by_drop(self):
        self.assertTrue(gf.make_dropper(["CTL"], [], [])(""))
        self.assertFalse(gf.make_dropper([], ["CTL"], [])(""))


class FindColumn(unittest.TestCase):
    def test_the_exact_spelling(self):
        self.assertEqual(gf.find_column(["FileName", "Grp"], gf.COL_GRP), "Grp")

    def test_another_spelling_of_the_same_name(self):
        self.assertEqual(gf.find_column(["filename", "GRP"], gf.COL_GRP), "GRP")

    def test_surrounding_whitespace_in_the_header(self):
        self.assertEqual(gf.find_column([" Grp "], gf.COL_GRP), " Grp ")

    def test_an_absent_column_is_none(self):
        self.assertIsNone(gf.find_column(["FileName", "FR"], gf.COL_GRP))


class GrpCounts(unittest.TestCase):
    def setUp(self):
        self.counts = gf.grp_counts(rows_of(MIXED_CSV), "Grp")

    def test_every_row_is_counted_against_its_value(self):
        self.assertEqual(self.counts["CTL"], 2)
        self.assertEqual(self.counts["CCTL"], 1)

    def test_values_differing_only_in_case_stay_distinct(self):
        self.assertIn("PreStim", self.counts)
        self.assertIn("prestim2", self.counts)

    def test_a_blank_grp_is_its_own_value(self):
        self.assertEqual(self.counts[""], 1)

    def test_values_are_ordered_case_insensitively(self):
        self.assertEqual(list(self.counts),
                         ["", "CCTL", "CTL", "MOS", "PreStim", "prestim2"])


class PrintTable(unittest.TestCase):
    ROWS = [{"A": "one", "B": "1"}, {"A": "two", "B": "2"}, {"A": "three", "B": "3"}]

    def test_empty_rows_print_nothing(self):
        self.assertEqual(captured(gf.print_table, [], ["A"]), "")

    def test_empty_columns_print_nothing(self):
        self.assertEqual(captured(gf.print_table, self.ROWS, []), "")

    def test_columns_are_padded_to_the_widest_cell(self):
        lines = captured(gf.print_table, self.ROWS, ["A", "B"]).splitlines()
        self.assertEqual(lines[0], "A      B")
        self.assertEqual(lines[1], "-----  -")

    def test_a_none_cell_renders_as_blank(self):
        out = captured(gf.print_table, [{"A": None}], ["A"])
        self.assertEqual(out.splitlines(), ["A", "-", ""])

    def test_a_limit_truncates_and_reports_the_remainder(self):
        out = captured(gf.print_table, self.ROWS, ["A"], limit=2)
        self.assertIn("two", out)
        self.assertNotIn("three", out)
        self.assertIn("... and 1 more.", out)

    def test_a_limit_wider_than_the_table_adds_no_tail(self):
        out = captured(gf.print_table, self.ROWS, ["A"], limit=10)
        self.assertNotIn("more", out)


class ResolveOutputPath(unittest.TestCase):
    def test_the_default_name_lands_beside_the_input(self):
        got = gf.resolve_output_path(Path("/data/in.csv"), None)
        self.assertEqual(got, Path("/data/in_filtered.csv"))

    def test_a_bare_filename_lands_beside_the_input(self):
        got = gf.resolve_output_path(Path("/data/in.csv"), Path("out.csv"))
        self.assertEqual(got, Path("/data/out.csv"))

    def test_a_directory_gets_the_default_name(self):
        got = gf.resolve_output_path(Path("/data/in.csv"), Path("/else"))
        self.assertEqual(got, Path("/else/in_filtered.csv"))

    def test_a_full_path_is_used_as_given(self):
        got = gf.resolve_output_path(Path("/data/in.csv"), Path("/else/out.csv"))
        self.assertEqual(got, Path("/else/out.csv"))

    def test_overwriting_the_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            with self.assertRaises(SystemExit) as cm, redirect_stderr(io.StringIO()):
                gf.resolve_output_path(path, path)
            self.assertIn("would overwrite", str(cm.exception))


class Cli(unittest.TestCase):
    def run_cli(self, tmp, *args, answer="yes\n", expect_ok=True):
        proc = subprocess.run(
            [sys.executable, str(Path(gf.__file__)), *args],
            capture_output=True, text=True, cwd=tmp, input=answer,
        )
        if expect_ok:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        else:
            self.assertNotEqual(proc.returncode, 0, proc.stdout)
        return proc

    def filter_to(self, tmp, *args, **kwargs):
        """Run against a fresh MIXED_CSV; return (path, output rows)."""
        path = write_csv(tmp)
        out = Path(tmp) / "out.csv"
        self.run_cli(tmp, str(path), *args, "-o", str(out), **kwargs)
        return path, read_rows(out)[1]

    def grps(self, rows):
        return [r["Grp"] for r in rows]

    # --- selection -----------------------------------------------------
    def test_drop_prefix_c_reproduces_the_legacy_behaviour(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--drop-prefix", "C")
            self.assertEqual(self.grps(rows), ["PreStim", "PreStim", "prestim2", "MOS", ""])

    def test_grp_keeps_only_the_named_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--grp", "PreStim")
            self.assertEqual(self.grps(rows), ["PreStim", "PreStim"])

    def test_grp_matches_case_insensitively(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--grp", "prestim")
            self.assertEqual(self.grps(rows), ["PreStim", "PreStim"])

    def test_grp_does_not_match_a_longer_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--grp", "PreStim")
            self.assertNotIn("prestim2", self.grps(rows))

    def test_grp_keeps_several_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--grp", "PreStim,MOS")
            self.assertEqual(self.grps(rows), ["PreStim", "PreStim", "MOS"])

    def test_the_two_drop_flags_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--drop", "CCTL", "--drop-prefix", "M")
            self.assertEqual(self.grps(rows), ["CTL", "CTL", "PreStim", "PreStim",
                                               "prestim2", ""])

    def test_repeated_and_comma_separated_values_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, repeated = self.filter_to(tmp, "--drop", "CTL", "--drop", "CCTL")
            _, comma = self.filter_to(tmp, "--drop", "CTL,CCTL")
            self.assertEqual(repeated, comma)

    def test_the_positional_survives_a_preceding_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, "--drop", "CTL", str(path), "--list")
            self.assertIn("8 row(s)", proc.stdout)

    # --- rejections ----------------------------------------------------
    def test_no_selection_flag_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), expect_ok=False)
            self.assertIn("no selection given", proc.stderr)
            self.assertFalse((Path(tmp) / "data_filtered.csv").exists())

    def test_grp_with_drop_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--grp", "CTL", "--drop", "MOS",
                                expect_ok=False)
            self.assertIn("cannot be combined", proc.stderr)

    def test_grp_with_drop_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--grp", "CTL", "--drop-prefix", "M",
                                expect_ok=False)
            self.assertIn("cannot be combined", proc.stderr)

    def test_a_selector_of_only_empty_values_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", " , ", expect_ok=False)
            self.assertIn("--drop was given no values", proc.stderr)

    def test_a_selection_keeping_no_rows_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--grp", "nosuchgroup", expect_ok=False)
            self.assertIn("would keep no rows", proc.stderr)
            self.assertFalse((Path(tmp) / "data_filtered.csv").exists())

    def test_a_missing_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_cli(tmp, "nope.csv", "--drop", "CTL", expect_ok=False)
            self.assertIn("does not exist", proc.stderr)

    def test_an_input_without_a_grp_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, text="FileName,FR\na,1\n")
            proc = self.run_cli(tmp, str(path), "--drop", "CTL", expect_ok=False)
            self.assertIn("no 'Grp' column", proc.stderr)
            self.assertIn("Found columns: FileName, FR", proc.stderr)

    def test_a_header_only_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp, text=HEADER)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL", expect_ok=False)
            self.assertIn("no data rows", proc.stderr)

    def test_overwriting_the_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL", "-o", str(path),
                                expect_ok=False)
            self.assertIn("would overwrite", proc.stderr)
            self.assertEqual(path.read_text(), MIXED_CSV)

    # --- --list --------------------------------------------------------
    def test_list_prints_each_grp_with_its_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--list")
            self.assertIn("CTL       2", proc.stdout)
            self.assertIn(gf.BLANK_LABEL, proc.stdout)
            self.assertIn("6 distinct Grp value(s), 8 row(s).", proc.stdout)

    def test_list_ignores_selection_flags_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            self.run_cli(tmp, str(path), "--list", "--grp", "CTL", "--drop", "MOS")
            self.assertFalse((Path(tmp) / "data_filtered.csv").exists())

    def test_list_needs_no_output_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            self.run_cli(tmp, str(path), "--list", "-o", str(path))

    # --- reporting and the prompt --------------------------------------
    def test_the_summary_names_each_grp_once_rather_than_each_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL")
            summary = proc.stdout.split("Recordings dropped:")[0]
            listed = [l for l in summary.splitlines()
                      if l and not l.startswith(("Grp", "---", "Dropping"))]
            self.assertEqual(len(listed), 6)  # one line per Grp value, not per row
            self.assertIn("Dropping 2 of 8 rows (1 of 6 Grp value(s)); keeping 6.",
                          proc.stdout)

    def test_dropped_recordings_are_listed_once_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL")
            self.assertIn("Recordings dropped:", proc.stdout)
            self.assertEqual(proc.stdout.count("R250929CT1A_DIV250_base"), 1)

    def test_an_unmatched_selector_warns_but_still_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            out = Path(tmp) / "out.csv"
            proc = self.run_cli(tmp, str(path), "--drop", "CTL,NOPE", "-o", str(out))
            self.assertIn("matched no Grp value in this file: --drop NOPE", proc.stderr)
            self.assertEqual(len(read_rows(out)[1]), 6)

    def test_answering_no_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL", answer="no\n")
            self.assertIn("Aborted", proc.stdout)
            self.assertFalse((Path(tmp) / "data_filtered.csv").exists())

    def test_a_closed_stdin_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            proc = self.run_cli(tmp, str(path), "--drop", "CTL", answer="")
            self.assertIn("Aborted", proc.stdout)
            self.assertFalse((Path(tmp) / "data_filtered.csv").exists())

    # --- output --------------------------------------------------------
    def test_the_default_output_lands_beside_the_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_csv(tmp)
            self.run_cli(tmp, str(path), "--drop", "CTL")
            self.assertTrue((Path(tmp) / "data_filtered.csv").is_file())

    def test_the_header_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows = self.filter_to(tmp, "--drop", "CTL")
            self.assertEqual(list(rows[0]), ["FileName", "Grp", "Channel", "FR"])

    def test_the_input_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.filter_to(tmp, "--drop", "CTL")
            self.assertEqual(path.read_text(), MIXED_CSV)


class SampleFile(unittest.TestCase):
    """Checks against a real-world-shaped export: 12 columns and a byte-order mark."""

    def run_cli(self, tmp, *args, answer="yes\n"):
        proc = subprocess.run(
            [sys.executable, str(Path(gf.__file__)), *args],
            capture_output=True, text=True, cwd=tmp, input=answer,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def copy(self, tmp):
        path = Path(tmp) / SAMPLE_CSV.name
        shutil.copy(SAMPLE_CSV, path)
        return path

    def test_list_reports_ctl_mos_and_mut(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self.run_cli(tmp, str(self.copy(tmp)), "--list")
            self.assertIn("3 distinct Grp value(s), 5 row(s).", proc.stdout)
            for grp in ("CTL", "MOS", "MUT"):
                self.assertIn(grp, proc.stdout)

    def test_the_grp_column_is_found_despite_the_byte_order_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.copy(tmp)
            out = Path(tmp) / "out.csv"
            self.run_cli(tmp, str(path), "--drop-prefix", "C", "-o", str(out))
            fieldnames, rows = read_rows(out)
            self.assertEqual(fieldnames[0], "FileName")
            self.assertEqual([r["Grp"] for r in rows], ["MOS", "MUT", "MOS"])

    def test_the_output_carries_no_byte_order_mark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.copy(tmp)
            out = Path(tmp) / "out.csv"
            self.run_cli(tmp, str(path), "--drop-prefix", "C", "-o", str(out))
            self.assertTrue(out.read_bytes().startswith(b"FileName"))


if __name__ == "__main__":
    unittest.main()
