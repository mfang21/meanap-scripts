"""Run with: python3 -m unittest discover -t . -s tests"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matplotlib import cbook

import fr_boxplots as fb


def rec(filename, grp, channel, fr):
    organoid, slc = fb.parse_organoid(filename)
    return fb.Record(filename, grp, channel, fr, organoid, slc, fb.parse_stim(filename))


RECORDS = [
    rec("R250929CT7A_DIV250", "BCTL", 1, 0.5),
    rec("R250929CT7A_DIV250", "BCTL", 2, 1.5),
    rec("R250929CT7B_DIV250", "BCTL", 1, 0.7),
    rec("R250929CT7B_DIV250", "BCTL", 2, 9.0),
    rec("R250929CT2A_DIV250", "BCTL", 1, 0.6),
    rec("R250929CT2A_DIV250", "BCTL", 2, 1.2),
    rec("R250929MO1A_DIV250", "BMOS", 1, 2.0),
    rec("R250929MO1A_DIV250", "BMOS", 2, 2.5),
]


class ParseOrganoid(unittest.TestCase):
    def test_docstring_example(self):
        self.assertEqual(fb.parse_organoid("R250929MO7B_DIV250_stim1"), ("MO7", "MO7B"))

    def test_lowercase_slice_letter_is_uppercased(self):
        self.assertEqual(fb.parse_organoid("R250929MO12b_DIV250"), ("MO12", "MO12B"))

    def test_any_marker_is_read_from_the_file_name(self):
        self.assertEqual(fb.parse_organoid("R250929XY3A_DIV250_base"), ("XY3", "XY3A"))

    def test_the_slice_must_follow_the_run_id(self):
        self.assertEqual(fb.parse_organoid("CT7A_DIV250_base"), (fb.UNKNOWN, fb.UNKNOWN))
        self.assertEqual(fb.parse_organoid("R250929_CT7A_DIV250"), (fb.UNKNOWN, fb.UNKNOWN))

    def test_no_match(self):
        self.assertEqual(fb.parse_organoid("nothing_here"), (fb.UNKNOWN, fb.UNKNOWN))


class ParseRunName(unittest.TestCase):
    def test_run_div_and_suffix(self):
        self.assertEqual(fb.parse_run_name("R250929CT7A_DIV250_base"), "R250929_DIV250_base")

    def test_no_trailing_suffix_after_div(self):
        self.assertIsNone(fb.parse_run_name("R250929CT7A_DIV250"))

    def test_no_match(self):
        self.assertIsNone(fb.parse_run_name("nothing_here"))

    def test_find_run_name_uses_first_parseable_record(self):
        records = RECORDS + [rec("R250929CT2A_DIV250_treatment", "BCTL", 1, 0.4)]
        self.assertEqual(fb.find_run_name(records), "R250929_DIV250_treatment")

    def test_find_run_name_none_when_nothing_parses(self):
        self.assertIsNone(fb.find_run_name(RECORDS))


MIXED_STIM = [
    rec("R250929CT7A_DIV250_stim3", "CTL", 1, 3.0),
    rec("R250929CT7A_DIV250_stim1", "CTL", 1, 1.0),
    rec("R250929MO5A_DIV250_stim3", "MOS", 1, 3.5),
    rec("R250929MO5A_DIV250_stim1", "MOS", 1, 1.5),
]


class ParseStim(unittest.TestCase):
    def test_numbered_stim(self):
        self.assertEqual(fb.parse_stim("R250929CT1A_DIV250_stim1"), "stim1")

    def test_direction_stim(self):
        self.assertEqual(fb.parse_stim("R250929CT1A_DIV250_stimLR"), "stimLR")

    def test_base(self):
        self.assertEqual(fb.parse_stim("R250929CT7A_DIV250_base"), "base")

    def test_last_token_wins(self):
        self.assertEqual(fb.parse_stim("R250929CT7A_DIV250_CAM_strpd"), "strpd")

    def test_no_div_segment(self):
        self.assertIsNone(fb.parse_stim("R250929CT7A_DIV250"))
        self.assertIsNone(fb.parse_stim("nothing_here"))


class SplitByStim(unittest.TestCase):
    def test_buckets_in_natural_order(self):
        buckets = fb.split_by_stim(MIXED_STIM)
        self.assertEqual([s for s, _ in buckets], ["stim1", "stim3"])
        for stim, recs in buckets:
            self.assertTrue(all(r.stim == stim for r in recs))
            self.assertEqual(len(recs), 2)

    def test_single_condition_is_one_bucket(self):
        buckets = fb.split_by_stim(RECORDS)
        self.assertEqual(len(buckets), 1)
        self.assertEqual(buckets[0], (None, RECORDS))

    def test_unparseable_stim_sorts_last(self):
        buckets = fb.split_by_stim(MIXED_STIM + RECORDS)
        self.assertEqual([s for s, _ in buckets], ["stim1", "stim3", None])

    def test_boxes_are_not_pooled_across_stims(self):
        # Pooled, channel 1 of CTL would span 1.0 and 3.0; split, each box holds one.
        by_stim = dict(fb.split_by_stim(MIXED_STIM))
        self.assertEqual(fb.box_stats(by_stim["stim1"], "CTL")[1]["med"], 1.0)
        self.assertEqual(fb.box_stats(by_stim["stim3"], "CTL")[1]["med"], 3.0)


class StimPath(unittest.TestCase):
    def test_inserts_stim_when_several(self):
        self.assertEqual(fb.stim_path(Path("out/viewer.html"), "stim1", True),
                         Path("out/viewer_stim1.html"))

    def test_untouched_for_a_single_condition(self):
        self.assertEqual(fb.stim_path(Path("out/viewer.html"), "base", False),
                         Path("out/viewer.html"))
        self.assertEqual(fb.stim_path(Path("out/viewer.html"), None, True),
                         Path("out/viewer.html"))


class BoxStats(unittest.TestCase):
    def test_matches_matplotlib_rule(self):
        stats = fb.box_stats(RECORDS, "BCTL")
        expected = cbook.boxplot_stats([1.5, 9.0, 1.2], whis=1.5)[0]
        self.assertEqual(stats[2]["q1"], expected["q1"])
        self.assertEqual(stats[2]["med"], expected["med"])
        self.assertEqual(stats[2]["q3"], expected["q3"])
        self.assertEqual(stats[2]["whislo"], expected["whislo"])
        self.assertEqual(stats[2]["whishi"], expected["whishi"])
        self.assertEqual(stats[2]["n"], 3)

    def test_all_groups_pools_every_record(self):
        self.assertEqual(fb.box_stats(RECORDS, fb.ALL_GROUPS)[1]["n"], 4)


class Payload(unittest.TestCase):
    def test_round_trips_and_covers_every_group(self):
        payload = json.loads(json.dumps(fb.build_payload(RECORDS, "x.csv")))
        self.assertEqual(payload["groups"], [fb.ALL_GROUPS, "BCTL", "BMOS"])
        self.assertEqual(payload["organoids"]["BCTL"], ["CT2", "CT7"])
        self.assertEqual(set(payload["stats"]), {fb.ALL_GROUPS, "BCTL", "BMOS"})
        self.assertEqual(len(payload["records"]), len(RECORDS))
        self.assertEqual(payload["colors"]["palette"], fb.PALETTE)
        self.assertIsNone(payload["runName"])

    def test_run_name_present_when_parseable(self):
        records = RECORDS + [rec("R250929CT7A_DIV250_base", "BCTL", 1, 0.5)]
        payload = json.loads(json.dumps(fb.build_payload(records, "x.csv")))
        self.assertEqual(payload["runName"], "R250929_DIV250_base")

    def test_stim_round_trips(self):
        payload = json.loads(json.dumps(fb.build_payload(RECORDS, "x.csv")))
        self.assertIsNone(payload["stim"])
        payload = json.loads(json.dumps(fb.build_payload(MIXED_STIM, "x.csv", "stim1")))
        self.assertEqual(payload["stim"], "stim1")

    def test_run_name_names_the_bucket_s_own_stim(self):
        # Pooled, find_run_name() would report whichever stim came first.
        for stim, recs in fb.split_by_stim(MIXED_STIM):
            payload = fb.build_payload(recs, "x.csv", stim)
            self.assertEqual(payload["runName"], f"R250929_DIV250_{stim}")


class RenderHtml(unittest.TestCase):
    def test_script_terminator_in_filename_is_escaped(self):
        hostile = RECORDS + [rec("</script><b>x</b>CT1A", "BCTL", 1, 0.1)]
        html = fb.render_html(fb.build_payload(hostile, "x.csv"), fb.initial_state())
        self.assertNotIn("</script><b>", html)
        self.assertIn("<\\/script><b>", html)

    def test_page_structure(self):
        html = fb.render_html(fb.build_payload(RECORDS, "x.csv"), fb.initial_state())
        self.assertIn(f'src="{fb.PLOTLY_JS_URL}"', html)
        self.assertIn('id="payload"', html)
        self.assertIn('id="initial"', html)
        self.assertNotIn("__PAYLOAD__", html)
        self.assertNotIn("__INITIAL__", html)


STIM_CSV = (
    "FileName,Grp,Channel,FR\n"
    "R250929CT7A_DIV250_stim1,CTL,1,0.5\n"
    "R250929CT7A_DIV250_stim1,CTL,2,1.5\n"
    "R250929CT7A_DIV250_stim3,CTL,1,3.5\n"
    "R250929CT7A_DIV250_stim3,CTL,2,4.5\n"
    "R250929MO1A_DIV250_stim1,MOS,1,2.0\n"
    "R250929MO1A_DIV250_stim3,MOS,1,6.0\n"
)


class Cli(unittest.TestCase):
    def run_cli(self, tmp, *args):
        proc = subprocess.run(
            [sys.executable, str(Path(fb.__file__)), *args],
            capture_output=True, text=True, cwd=tmp,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def test_html_output_needs_no_grp(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "data.csv"
            csv_path.write_text(
                "FileName,Grp,Channel,FR\n"
                "R250929CT7A_DIV250,BCTL,1,0.5\n"
                "R250929CT7A_DIV250,BCTL,2,1.5\n"
                "R250929MO1A_DIV250,BMOS,1,2.0\n"
            )
            out = Path(tmp) / "viewer.html"
            self.run_cli(tmp, str(csv_path), "-o", str(out))
            self.assertTrue(out.is_file())
            self.assertIn('id="payload"', out.read_text())

    def test_single_stim_keeps_the_requested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "base.csv"
            csv_path.write_text(
                "FileName,Grp,Channel,FR\n"
                "R250929CT7A_DIV250_base,BCTL,1,0.5\n"
                "R250929CT7A_DIV250_base,BCTL,2,1.5\n"
            )
            out = Path(tmp) / "viewer.html"
            self.run_cli(tmp, str(csv_path), "-o", str(out))
            self.assertTrue(out.is_file())
            self.assertFalse((Path(tmp) / "viewer_base.html").exists())

    def test_two_stims_write_two_viewers(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stim1_3.csv"
            csv_path.write_text(STIM_CSV)
            self.run_cli(tmp, str(csv_path), "-o", str(Path(tmp) / "viewer.html"))

            self.assertFalse((Path(tmp) / "viewer.html").exists())
            for stim, other in (("stim1", "stim3"), ("stim3", "stim1")):
                out = Path(tmp) / f"viewer_{stim}.html"
                self.assertTrue(out.is_file())
                html = out.read_text()
                self.assertIn(f"_{stim}", html)
                self.assertNotIn(f"_{other}", html)

    def test_stim_flag_narrows_to_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stim1_3.csv"
            csv_path.write_text(STIM_CSV)
            out = Path(tmp) / "viewer.html"
            self.run_cli(tmp, str(csv_path), "--stim", "stim1", "-o", str(out))

            self.assertTrue(out.is_file())
            self.assertFalse((Path(tmp) / "viewer_stim1.html").exists())
            self.assertNotIn("_stim3", out.read_text())

    def test_unknown_stim_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stim1_3.csv"
            csv_path.write_text(STIM_CSV)
            proc = subprocess.run(
                [sys.executable, str(Path(fb.__file__)), str(csv_path),
                 "--stim", "stim9", "-o", str(Path(tmp) / "viewer.html")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("stim1, stim3", proc.stderr)

    def test_two_stims_write_two_static_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stim1_3.csv"
            csv_path.write_text(STIM_CSV)
            self.run_cli(tmp, str(csv_path), "--grp", "all",
                         "-o", str(Path(tmp) / "overview.png"))
            self.assertTrue((Path(tmp) / "overview_stim1.png").is_file())
            self.assertTrue((Path(tmp) / "overview_stim3.png").is_file())


SAMPLE_CSV = Path(__file__).resolve().parent / "sample-files" / "NeuronalActivity_NodeLevel_sample.csv"


class SampleFile(unittest.TestCase):
    """Checks against a real-world-shaped export: extra unused columns padded
    with NaN, an FR of exactly 0, and one file name whose DIV segment has no
    digits (so it doesn't match the usual DIV<n>_ pattern)."""

    def test_load_records_reads_every_row(self):
        records = fb.load_records(SAMPLE_CSV)
        self.assertEqual(len(records), 5)

    def test_extra_nan_columns_are_ignored(self):
        records = fb.load_records(SAMPLE_CSV)
        by_filename = {r.filename: r for r in records}
        self.assertEqual(by_filename["R250929MO7C_DIV_stim3"].fr, 0.95)

    def test_fr_of_zero_is_kept_not_treated_as_missing(self):
        records = fb.load_records(SAMPLE_CSV)
        by_filename = {r.filename: r for r in records}
        self.assertEqual(by_filename["R250929MO2A_DIV250_stimRL"].fr, 0)

    def test_organoid_parsed_for_every_group_marker(self):
        records = fb.load_records(SAMPLE_CSV)
        by_filename = {r.filename: r for r in records}
        self.assertEqual(by_filename["R250929CT1A_DIV250_stim1"].organoid, "CT1")
        self.assertEqual(by_filename["R250929MT3D_DIV250_base"].organoid, "MT3")
        self.assertEqual(by_filename["R250929MO2A_DIV250_stimRL"].organoid, "MO2")

    def test_malformed_div_segment_yields_no_stim(self):
        records = fb.load_records(SAMPLE_CSV)
        by_filename = {r.filename: r for r in records}
        self.assertIsNone(by_filename["R250929MO7C_DIV_stim3"].stim)

    def test_cli_runs_end_to_end_on_the_sample_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "viewer.html"
            proc = subprocess.run(
                [sys.executable, str(Path(fb.__file__)), str(SAMPLE_CSV), "-o", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            html_files = list(Path(tmp).glob("*.html"))
            self.assertEqual(len(html_files), 5)
            for f in html_files:
                self.assertIn('id="payload"', f.read_text())


if __name__ == "__main__":
    unittest.main()
