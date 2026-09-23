"""Run with: python3 -m unittest discover -t . -s tests"""

import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import merge_csv as mc


HEADER = "FileName,Grp,Channel,FR\n"

BASE_CSV = HEADER + (
    "R250929CT7A_DIV250_base,BCTL,1,0.5\n"
    "R250929CT7A_DIV250_base,BCTL,2,1.5\n"
    "R250929MO1A_DIV250_base,BMOS,1,2.0\n"
)
STIM1_CSV = HEADER + (
    "R250929CT7A_DIV250_stim1,BCTL,1,3.5\n"
    "R250929MO1A_DIV250_stim1,BMOS,1,4.5\n"
)
STIM3_CSV = HEADER + (
    "R250929CT7A_DIV250_stim3,BCTL,1,6.5\n"
)


def table(name, text):
    """A Table built from CSV text, as read_table() would return it."""
    rows = list(csv.DictReader(text.splitlines()))
    return mc.Table(Path(name), list(rows[0]), rows)


def write_files(tmp, **files):
    """Write {stem: csv text} into tmp; return the paths in the order given."""
    paths = []
    for stem, text in files.items():
        path = Path(tmp) / f"{stem}.csv"
        path.write_text(text)
        paths.append(path)
    return paths


def read_rows(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


class ParseRun(unittest.TestCase):
    def test_strips_the_fused_organoid_marker(self):
        self.assertEqual(mc.parse_run("R250929CT7A_DIV250_stim1"), "R250929")

    def test_ignores_everything_after_the_run_number(self):
        self.assertEqual(mc.parse_run("R250929MO12b_DIV_stim3"), "R250929")

    def test_run_alone(self):
        self.assertEqual(mc.parse_run("R250929"), "R250929")

    def test_no_match(self):
        self.assertIsNone(mc.parse_run("nothing_here"))
        self.assertIsNone(mc.parse_run(""))
        self.assertIsNone(mc.parse_run("XR250929CT7A_DIV250_base"))


class CheckHeaders(unittest.TestCase):
    def test_returns_the_first_file_s_header(self):
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", STIM1_CSV)]
        self.assertEqual(mc.check_headers(tables), ["FileName", "Grp", "Channel", "FR"])

    def test_case_and_order_may_differ(self):
        shuffled = "fr,Channel,grp,FILENAME\n0.5,1,BCTL,R250929CT7A_DIV250_stim1\n"
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", shuffled)]
        self.assertEqual(mc.check_headers(tables), ["FileName", "Grp", "Channel", "FR"])

    def test_missing_column_is_rejected(self):
        short = "FileName,Grp,Channel\nR250929CT7A_DIV250_stim1,BCTL,1\n"
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", short)]
        with self.assertRaises(SystemExit) as cm:
            mc.check_headers(tables)
        self.assertIn("Missing: FR", str(cm.exception))

    def test_extra_column_is_rejected(self):
        wide = HEADER.rstrip("\n") + ",Extra\nR250929CT7A_DIV250_stim1,BCTL,1,0.5,x\n"
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", wide)]
        with self.assertRaises(SystemExit) as cm:
            mc.check_headers(tables)
        self.assertIn("Unexpected: Extra", str(cm.exception))

    def test_filename_column_is_required(self):
        tables = [table("base.csv", "Grp,FR\nBCTL,0.5\n")]
        with self.assertRaises(SystemExit) as cm:
            mc.check_headers(tables)
        self.assertIn("no 'FileName' column", str(cm.exception))


class Aligned(unittest.TestCase):
    def test_identical_header_passes_the_rows_straight_through(self):
        t = table("base.csv", BASE_CSV)
        self.assertIs(t.aligned(["FileName", "Grp", "Channel", "FR"]), t.rows)

    def test_differing_case_and_order_are_rekeyed(self):
        shuffled = "fr,Channel,grp,FILENAME\n0.5,1,BCTL,R250929CT7A_DIV250_stim1\n"
        rekeyed = table("stim1.csv", shuffled).aligned(["FileName", "Grp", "Channel", "FR"])
        self.assertEqual(rekeyed, [{"FileName": "R250929CT7A_DIV250_stim1", "Grp": "BCTL",
                                    "Channel": "1", "FR": "0.5"}])


class CheckRuns(unittest.TestCase):
    def test_one_run_across_every_file(self):
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", STIM1_CSV),
                  table("stim3.csv", STIM3_CSV)]
        self.assertEqual(mc.check_runs(tables), "R250929")

    def test_two_runs_in_one_file_is_rejected(self):
        mixed = BASE_CSV + "R250930CT7A_DIV250_base,BCTL,1,0.9\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_runs([table("base.csv", mixed)])
        self.assertIn("2 runs", str(cm.exception))
        self.assertIn("R250930", str(cm.exception))

    def test_differing_runs_across_files_is_rejected(self):
        other = STIM1_CSV.replace("R250929", "R250930")
        with self.assertRaises(SystemExit) as cm:
            mc.check_runs([table("base.csv", BASE_CSV), table("stim1.csv", other)])
        self.assertIn("not from the same run", str(cm.exception))
        self.assertIn("R250930", str(cm.exception))

    def test_file_without_any_run_id_is_rejected(self):
        nameless = HEADER + "not_a_recording,BCTL,1,0.5\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_runs([table("base.csv", nameless)])
        self.assertIn("no FileName", str(cm.exception))

    def test_unparseable_names_warn_but_do_not_stop_the_merge(self):
        odd = BASE_CSV + "not_a_recording,BCTL,1,0.9\n"
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(mc.check_runs([table("base.csv", odd)]), "R250929")
        self.assertIn("could not parse a run ID from 1 file name(s)", err.getvalue())
        self.assertIn("not_a_recording", err.getvalue())

    def test_no_merge_warning_when_a_later_file_aborts_the_merge(self):
        odd = BASE_CSV + "not_a_recording,BCTL,1,0.9\n"
        other = STIM1_CSV.replace("R250929", "R250930")
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            mc.check_runs([table("base.csv", odd), table("stim1.csv", other)])
        self.assertNotIn("merged unchecked", err.getvalue())


class CheckDuplicates(unittest.TestCase):
    def test_disjoint_files_pass(self):
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", STIM1_CSV),
                  table("stim3.csv", STIM3_CSV)]
        self.assertIsNone(mc.check_duplicates(tables))

    def test_a_name_repeating_within_one_file_is_expected(self):
        # BASE_CSV lists R250929CT7A_DIV250_base twice, once per channel.
        self.assertIsNone(mc.check_duplicates([table("base.csv", BASE_CSV)]))

    def test_the_same_file_twice_is_rejected(self):
        tables = [table("base.csv", BASE_CSV), table("copy.csv", BASE_CSV)]
        with self.assertRaises(SystemExit) as cm:
            mc.check_duplicates(tables)
        message = str(cm.exception)
        self.assertIn("appears in more than one input file", message)
        self.assertIn("share 2 recording(s)", message)
        self.assertIn("R250929CT7A_DIV250_base", message)
        self.assertIn("R250929MO1A_DIV250_base", message)

    def test_a_single_overlapping_recording_is_rejected(self):
        overlap = STIM1_CSV + "R250929CT7A_DIV250_base,BCTL,9,9.0\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_duplicates([table("base.csv", BASE_CSV), table("stim1.csv", overlap)])
        message = str(cm.exception)
        self.assertIn("'base.csv' and 'stim1.csv' share 1 recording(s)", message)
        self.assertIn("R250929CT7A_DIV250_base", message)
        self.assertNotIn("R250929MO1A", message)

    def test_overlap_between_non_adjacent_files_is_found(self):
        overlap = STIM3_CSV + "R250929MO1A_DIV250_base,BMOS,1,9.0\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_duplicates([table("base.csv", BASE_CSV), table("stim1.csv", STIM1_CSV),
                                 table("stim3.csv", overlap)])
        self.assertIn("'base.csv' and 'stim3.csv'", str(cm.exception))

    def test_blank_file_names_are_not_treated_as_a_repeat(self):
        blank = STIM1_CSV + ",BCTL,9,9.0\n"
        self.assertIsNone(mc.check_duplicates(
            [table("base.csv", BASE_CSV + ",BCTL,9,9.0\n"), table("stim1.csv", blank)]))


class CheckRecordings(unittest.TestCase):
    def test_distinct_recordings_pass(self):
        tables = [table("base.csv", BASE_CSV), table("stim1.csv", STIM1_CSV),
                  table("stim3.csv", STIM3_CSV)]
        self.assertIsNone(mc.check_recordings(tables))

    def test_a_channel_repeated_within_a_recording_is_rejected(self):
        doubled = BASE_CSV + "R250929CT7A_DIV250_base,BCTL,2,9.9\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_recordings([table("base.csv", doubled)])
        message = str(cm.exception)
        self.assertIn("more than one row", message)
        self.assertIn("R250929CT7A_DIV250_base, channel 2", message)

    def test_the_same_slice_and_condition_at_two_divs_is_rejected(self):
        other_div = HEADER + "R250929CT7A_DIV251_base,BCTL,6,9.0\n"
        with self.assertRaises(SystemExit) as cm:
            mc.check_recordings([table("base.csv", BASE_CSV), table("late.csv", other_div)])
        message = str(cm.exception)
        self.assertIn("more than one DIV", message)
        self.assertIn("R250929CT7A_DIV250_base / R250929CT7A_DIV251_base", message)

    def test_two_divs_within_one_file_are_rejected_too(self):
        mixed = BASE_CSV + "R250929MO1A_DIV251_base,BMOS,1,2.0\n"
        with self.assertRaises(SystemExit):
            mc.check_recordings([table("base.csv", mixed)])

    def test_other_conditions_of_the_slice_are_not_a_clash(self):
        tables = [table("base.csv", BASE_CSV),
                  table("stim1.csv", HEADER + "R250929CT7A_DIV251_stim1,BCTL,1,3.5\n")]
        self.assertIsNone(mc.check_recordings(tables))

    def test_without_a_channel_column_only_the_div_check_runs(self):
        text = "FileName,FR\nR250929CT7A_DIV250_base,1\nR250929CT7A_DIV250_base,2\n"
        self.assertIsNone(mc.check_recordings([table("a.csv", text)]))


class ResolveOutputPath(unittest.TestCase):
    def test_default_name_beside_the_first_input(self):
        inputs = [Path("/data/run/base.csv"), Path("/data/run/stim1.csv")]
        self.assertEqual(mc.resolve_output_path(inputs, None),
                         Path("/data/run") / mc.DEFAULT_NAME)

    def test_bare_filename_lands_beside_the_first_input(self):
        inputs = [Path("/data/run/base.csv"), Path("/data/run/stim1.csv")]
        self.assertEqual(mc.resolve_output_path(inputs, Path("merged.csv")),
                         Path("/data/run/merged.csv"))

    def test_directory_gets_the_default_name(self):
        inputs = [Path("/data/run/base.csv")]
        self.assertEqual(mc.resolve_output_path(inputs, Path("/out")),
                         Path("/out") / mc.DEFAULT_NAME)

    def test_full_path_used_as_given(self):
        inputs = [Path("/data/run/base.csv")]
        self.assertEqual(mc.resolve_output_path(inputs, Path("/out/x.csv")),
                         Path("/out/x.csv"))

    def test_overwriting_an_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, stim1 = write_files(tmp, base=BASE_CSV, stim1=STIM1_CSV)
            with self.assertRaises(SystemExit) as cm:
                mc.resolve_output_path([base, stim1], stim1)
            self.assertIn("would overwrite", str(cm.exception))


class Cli(unittest.TestCase):
    def run_cli(self, tmp, *args, expect_ok=True):
        proc = subprocess.run(
            [sys.executable, str(Path(mc.__file__)), *args],
            capture_output=True, text=True, cwd=tmp,
        )
        if expect_ok:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        else:
            self.assertNotEqual(proc.returncode, 0, proc.stdout)
        return proc

    def test_three_conditions_merge_into_the_default_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, stim1=STIM1_CSV, stim3=STIM3_CSV)
            self.run_cli(tmp, *map(str, paths))

            out = Path(tmp) / mc.DEFAULT_NAME
            self.assertTrue(out.is_file())
            fieldnames, rows = read_rows(out)
            self.assertEqual(fieldnames, ["FileName", "Grp", "Channel", "FR"])
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["FileName"] for r in rows[:2]],
                             ["R250929CT7A_DIV250_base"] * 2)
            self.assertEqual(rows[-1]["FR"], "6.5")

    def test_inputs_are_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, stim1=STIM1_CSV)
            self.run_cli(tmp, *map(str, paths))
            self.assertEqual(paths[0].read_text(), BASE_CSV)
            self.assertEqual(paths[1].read_text(), STIM1_CSV)

    def test_output_flag_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, stim1=STIM1_CSV)
            out = Path(tmp) / "combined.csv"
            self.run_cli(tmp, *map(str, paths), "-o", str(out))
            self.assertTrue(out.is_file())
            self.assertFalse((Path(tmp) / mc.DEFAULT_NAME).exists())

    def test_a_single_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (base,) = write_files(tmp, base=BASE_CSV)
            proc = self.run_cli(tmp, str(base), expect_ok=False)
            self.assertIn("at least two", proc.stderr)

    def test_missing_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (base,) = write_files(tmp, base=BASE_CSV)
            proc = self.run_cli(tmp, str(base), str(Path(tmp) / "nope.csv"), expect_ok=False)
            self.assertIn("does not exist", proc.stderr)

    def test_nothing_is_written_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV,
                                stim1=STIM1_CSV.replace("R250929", "R250930"))
            proc = self.run_cli(tmp, *map(str, paths), expect_ok=False)
            self.assertIn("not from the same run", proc.stderr)
            self.assertFalse((Path(tmp) / mc.DEFAULT_NAME).exists())

    def test_a_recording_at_two_divs_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV,
                                late=HEADER + "R250929CT7A_DIV251_base,BCTL,6,9.0\n")
            proc = self.run_cli(tmp, *map(str, paths), expect_ok=False)
            self.assertIn("more than one DIV", proc.stderr)
            self.assertFalse((Path(tmp) / mc.DEFAULT_NAME).exists())

    def test_the_same_path_twice_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, stim1=STIM1_CSV)
            proc = self.run_cli(tmp, *map(str, paths), str(paths[0]), expect_ok=False)
            self.assertIn("given more than once", proc.stderr)
            self.assertFalse((Path(tmp) / mc.DEFAULT_NAME).exists())

    def test_two_copies_under_different_names_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, copy=BASE_CSV)
            proc = self.run_cli(tmp, *map(str, paths), expect_ok=False)
            self.assertIn("appears in more than one input file", proc.stderr)
            self.assertFalse((Path(tmp) / mc.DEFAULT_NAME).exists())

    def test_empty_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_files(tmp, base=BASE_CSV, empty=HEADER)
            proc = self.run_cli(tmp, *map(str, paths), expect_ok=False)
            self.assertIn("no data rows", proc.stderr)


SAMPLE_CSV = Path(__file__).resolve().parent / "sample-files" / "NeuronalActivity_NodeLevel_sample.csv"


class SampleFile(unittest.TestCase):
    """Checks against a real-world-shaped export: 12 columns, NaN padding, and a
    file name whose DIV segment has no digits."""

    def sibling(self, tmp):
        """A second file of the same run, sharing the sample's 12-column header."""
        header, first = SAMPLE_CSV.read_text().splitlines()[:2]
        path = Path(tmp) / "sibling.csv"
        path.write_text(f"{header}\n{first.replace('_stim1', '_stimLR')}\n")
        return path

    def test_merges_with_a_sibling_of_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "merged.csv"
            proc = subprocess.run(
                [sys.executable, str(Path(mc.__file__)), str(SAMPLE_CSV),
                 str(self.sibling(tmp)), "-o", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("run R250929", proc.stdout)

            fieldnames, rows = read_rows(out)
            self.assertEqual(len(fieldnames), 12)
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[-1]["FileName"], "R250929CT1A_DIV250_stimLR")
            self.assertEqual(rows[1]["channelBurstRate"], "NaN")

    def test_malformed_div_segment_still_yields_a_run(self):
        self.assertEqual(mc.parse_run("R250929MO7C_DIV_stim3"), "R250929")


if __name__ == "__main__":
    unittest.main()
