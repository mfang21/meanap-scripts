"""Run with: python3 -m unittest test_fr_boxplots.py"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from matplotlib import cbook

import fr_boxplots as fb


def rec(filename, grp, channel, fr):
    organoid, slc = fb.parse_organoid(filename, grp)
    return fb.Record(filename, grp, channel, fr, organoid, slc)


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
        self.assertEqual(fb.parse_organoid("R250929CT7A_DIV250", "BCTL"), ("CT7", "CT7A"))

    def test_lowercase_slice_letter_is_uppercased(self):
        self.assertEqual(fb.parse_organoid("R250929MO12b_DIV250", "BMOS"), ("MO12", "MO12B"))

    def test_unknown_group_tries_every_marker(self):
        self.assertEqual(fb.parse_organoid("R250929MT3A_DIV250", "WHAT"), ("MT3", "MT3A"))

    def test_no_match(self):
        self.assertEqual(fb.parse_organoid("nothing_here", "BCTL"), (fb.UNKNOWN, fb.UNKNOWN))


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


class Cli(unittest.TestCase):
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
            proc = subprocess.run(
                [sys.executable, str(Path(fb.__file__)), str(csv_path), "-o", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(out.is_file())
            self.assertIn('id="payload"', out.read_text())


if __name__ == "__main__":
    unittest.main()
