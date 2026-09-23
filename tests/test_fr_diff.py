"""Run with: python3 -m unittest discover -t . -s tests"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fr_boxplots as fb
import fr_diff as fd


SAMPLE_CSV = Path(__file__).resolve().parent / "sample-files" / "NeuronalActivity_NodeLevel_sample.csv"

HEADER = "FileName,Grp,Channel,FR\n"


def rec(filename, grp, channel, fr):
    """A Record built the way load_records() builds one, through the real parsers."""
    organoid, slc = fb.parse_organoid(filename)
    return fb.Record(filename, grp, channel, fr, organoid, slc, fb.parse_stim(filename))


def panels_of(records):
    """build_panels() with its stderr notes swallowed."""
    with redirect_stderr(io.StringIO()):
        return fd.build_panels(records)


def write_csv(tmp, name, rows):
    """Write a four-column CSV of (filename, grp, channel, fr) tuples."""
    path = Path(tmp) / name
    path.write_text(HEADER + "".join(f"{f},{g},{c},{v}\n" for f, g, c, v in rows))
    return path


PAIRED_ROWS = [
    ("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
    ("R250929CT1A_DIV250_base", "CTL", 2, 4.0),
    ("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
    ("R250929CT1A_DIV250_stim1", "CTL", 2, 2.0),
    ("R250929CT2C_DIV250_base", "CTL", 1, 1.0),
    ("R250929CT2C_DIV250_stimLR", "CTL", 1, 1.5),
]


class ParseRun(unittest.TestCase):
    def test_strips_the_fused_organoid_marker(self):
        self.assertEqual(fd.parse_run("R250929CT1A_DIV250_base"), "R250929")

    def test_malformed_div_segment_still_yields_a_run(self):
        self.assertEqual(fd.parse_run("R250929MO7C_DIV_stim3"), "R250929")

    def test_a_name_without_a_run_number_is_unparseable(self):
        self.assertIsNone(fd.parse_run("CT1A_DIV250_base"))


class PctDiff(unittest.TestCase):
    def test_doubling_is_a_hundred_percent(self):
        self.assertEqual(fd.pct_diff(2.0, 4.0), 100.0)

    def test_halving_is_minus_fifty_percent(self):
        self.assertEqual(fd.pct_diff(4.0, 2.0), -50.0)

    def test_no_change_is_zero_percent(self):
        self.assertEqual(fd.pct_diff(3.0, 3.0), 0.0)

    def test_falling_silent_is_minus_a_hundred_percent(self):
        self.assertEqual(fd.pct_diff(3.0, 0.0), -100.0)

    def test_a_baseline_of_zero_has_no_percentage(self):
        self.assertIsNone(fd.pct_diff(0.0, 5.0))


class IsGrounded(unittest.TestCase):
    def test_a_zero_baseline_grounds_the_channel(self):
        self.assertTrue(fd.is_grounded(0.0, (("stim1", 3.0),)))

    def test_every_stim_reading_at_zero_grounds_the_channel(self):
        self.assertTrue(fd.is_grounded(2.0, (("stim1", 0.0), ("stim3", 0.0))))

    def test_one_stim_reading_above_zero_is_enough_to_keep_it(self):
        self.assertFalse(fd.is_grounded(2.0, (("stim1", 0.0), ("stim3", 0.1))))

    def test_a_live_channel_is_not_grounded(self):
        self.assertFalse(fd.is_grounded(2.0, (("stim1", 3.0),)))

    def test_a_baseline_with_no_stim_reading_at_all_is_not_grounded(self):
        self.assertFalse(fd.is_grounded(2.0, ()))


class BuildPanels(unittest.TestCase):
    def test_a_base_and_a_stim_of_one_slice_make_one_panel(self):
        panels, unpaired = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertEqual(unpaired, [])
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0].slice, "CT1A")
        self.assertEqual(panels[0].run, "R250929")
        self.assertEqual([d.pct for d in panels[0].diffs], [50.0])

    def test_two_slices_of_one_organoid_do_not_pair(self):
        panels, unpaired = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1B_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertEqual(panels, [])
        self.assertEqual({u.reason for u in unpaired},
                         {fd.REASON_NO_STIM, fd.REASON_NO_BASE})

    def test_the_same_slice_in_two_runs_does_not_pair(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250930CT1A_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertEqual(panels, [])

    def test_a_base_only_slice_is_reported_and_not_plotted(self):
        panels, unpaired = panels_of([rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0)])
        self.assertEqual(panels, [])
        self.assertEqual(len(unpaired), 1)
        self.assertEqual(unpaired[0].reason, fd.REASON_NO_STIM)
        self.assertEqual(unpaired[0].slice, "CT1A")

    def test_a_stim_only_slice_is_reported_and_not_plotted(self):
        panels, unpaired = panels_of([rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0)])
        self.assertEqual(panels, [])
        self.assertEqual(len(unpaired), 1)
        self.assertEqual(unpaired[0].reason, fd.REASON_NO_BASE)

    def test_two_stims_of_one_slice_share_a_panel(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250929CT1A_DIV250_stim3", "CTL", 1, 1.0),
        ])
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0].stims, ["stim1", "stim3"])
        self.assertEqual({(d.stim, d.pct) for d in panels[0].diffs},
                         {("stim1", 50.0), ("stim3", -50.0)})

    def test_a_channel_missing_from_the_base_is_set_aside(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 9, 3.0),
        ])
        self.assertEqual([d.channel for d in panels[0].diffs], [1])
        self.assertEqual(panels[0].missing_base, [9])

    def test_a_channel_missing_from_the_stim_is_simply_absent(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_base", "CTL", 9, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertEqual([d.channel for d in panels[0].diffs], [1])
        self.assertEqual(panels[0].missing_base, [])

    def test_a_zero_baseline_grounds_the_channel_instead_of_giving_a_percentage(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 0.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250929CT1A_DIV250_stimRL", "CTL", 1, 0.0),
        ])
        self.assertEqual(panels[0].diffs, [])
        self.assertEqual(len(panels[0].grounded), 1)
        grounded = panels[0].grounded[0]
        self.assertEqual(grounded.channel, 1)
        self.assertEqual(grounded.base_fr, 0.0)
        self.assertEqual(grounded.readings, (("stim1", 3.0), ("stimRL", 0.0)))

    def test_a_channel_silent_under_every_stim_is_grounded_not_minus_one_hundred(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 0.0),
            rec("R250929CT1A_DIV250_stim3", "CTL", 1, 0.0),
        ])
        self.assertEqual(panels[0].diffs, [])
        self.assertEqual([g.channel for g in panels[0].grounded], [1])
        self.assertEqual(panels[0].grounded[0].base_fr, 2.0)
        self.assertEqual(panels[0].grounded[0].readings,
                         (("stim1", 0.0), ("stim3", 0.0)))

    def test_a_channel_silent_under_only_one_stim_still_reads_minus_one_hundred(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 0.0),
            rec("R250929CT1A_DIV250_stim3", "CTL", 1, 3.0),
        ])
        self.assertEqual(panels[0].grounded, [])
        self.assertEqual({(d.stim, d.pct) for d in panels[0].diffs},
                         {("stim1", -100.0), ("stim3", 50.0)})

    def test_a_zero_baseline_no_stim_recorded_produces_nothing(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 0.0),
            rec("R250929CT1A_DIV250_base", "CTL", 2, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 2, 3.0),
        ])
        self.assertEqual(panels[0].grounded, [])
        self.assertEqual([d.channel for d in panels[0].diffs], [2])

    def test_a_duplicate_channel_keeps_the_first_value_and_says_so(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, _ = fd.build_panels([
                rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
                rec("R250929CT1A_DIV250_base", "CTL", 1, 99.0),
                rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            ])
        self.assertEqual([d.base_fr for d in panels[0].diffs], [2.0])
        self.assertIn("duplicate", buf.getvalue())

    def test_two_unparseable_slices_of_one_run_do_not_pair_with_each_other(self):
        panels, _ = panels_of([
            rec("R250929_DIV250_base", "CTL", 1, 2.0),
            rec("R250929_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertEqual(panels, [])

    def test_an_unparseable_condition_is_excluded_and_reported(self):
        panels, unpaired = panels_of([rec("R250929MO7C_DIV_stim3", "MOS", 13, 0.95)])
        self.assertEqual(panels, [])
        self.assertEqual([u.reason for u in unpaired], [fd.REASON_UNPARSED])
        self.assertEqual(unpaired[0].conditions, ("R250929MO7C_DIV_stim3",))

    def test_an_unknown_condition_is_still_plotted_with_a_note(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, _ = fd.build_panels([
                rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
                rec("R250929CT1A_DIV250_drug", "CTL", 1, 3.0),
            ])
        self.assertEqual(panels[0].stims, ["drug"])
        self.assertEqual([d.pct for d in panels[0].diffs], [50.0])
        self.assertIn("drug", buf.getvalue())

    def test_channels_are_the_sorted_union_of_diffs_and_grounded_channels(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 7, 2.0),
            rec("R250929CT1A_DIV250_base", "CTL", 3, 0.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 7, 3.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 3, 1.0),
        ])
        self.assertEqual(panels[0].channels, [3, 7])


class PanelTitle(unittest.TestCase):
    def test_one_run_is_named_by_slice_alone(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertFalse(fd.multi_run(panels))
        self.assertEqual(fd.panel_title(panels[0], False), "CT1A")

    def test_two_runs_qualify_the_slice_with_the_run(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250930CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250930CT1A_DIV250_stim1", "CTL", 1, 3.0),
        ])
        self.assertTrue(fd.multi_run(panels))
        self.assertEqual([fd.panel_title(p, True) for p in panels],
                         ["R250929 CT1A", "R250930 CT1A"])


class StimLabel(unittest.TestCase):
    def test_the_four_patterns_have_their_own_names(self):
        self.assertEqual([fd.stim_label(t) for t in ("stim1", "stim3", "stimLR", "stimRL")],
                         ["Stim 1", "Stim 3", "Stim LR", "Stim RL"])

    def test_an_unknown_token_names_itself(self):
        self.assertEqual(fd.stim_label("drug"), "drug")

    def test_every_token_gets_a_colour(self):
        colors = [fd.stim_color(t, i) for i, t in
                  enumerate(["stim1", "stim3", "stimLR", "stimRL", "drug", "washout"])]
        self.assertEqual(len(colors), 6)
        self.assertTrue(all(c.startswith("#") for c in colors))
        self.assertEqual(len(set(colors[:4])), 4)


class Payload(unittest.TestCase):
    def build(self):
        panels, _ = panels_of([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_base", "CTL", 2, 0.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 2, 1.0),
            rec("R250929CT1A_DIV250_stim3", "CTL", 9, 1.0),
            rec("R250929CT2C_DIV250_base", "CTL", 1, 1.0),
            rec("R250929CT2C_DIV250_stimLR", "CTL", 1, 0.5),
        ])
        return panels, fd.build_payload(panels, "data.csv")

    def test_it_survives_a_json_round_trip(self):
        _, payload = self.build()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_conditions_are_listed_in_natural_order(self):
        _, payload = self.build()
        self.assertEqual(payload["stims"], ["stim1", "stim3", "stimLR"])
        self.assertEqual(set(payload["labels"]), set(payload["colors"]["stim"]))

    def test_the_range_spans_every_panel(self):
        panels, payload = self.build()
        every = [d.pct for p in panels for d in p.diffs]
        self.assertEqual(payload["range"]["min"], min(0.0, min(every)))
        self.assertEqual(payload["range"]["max"], max(0.0, max(every)))

    def test_panel_ids_are_unique_and_qualified_by_run(self):
        _, payload = self.build()
        ids = [p["id"] for p in payload["panels"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("R250929/CT1A", ids)

    def test_the_counts_agree_with_the_arrays(self):
        _, payload = self.build()
        for p in payload["panels"]:
            self.assertEqual(p["n"]["paired"], len({q["c"] for q in p["points"]}))
            self.assertEqual(p["n"]["grounded"], len(p["grounded"]))
            self.assertEqual(p["n"]["missingBase"], len(p["missingBase"]))

    def test_a_grounded_channel_is_carried_as_a_bare_channel_number(self):
        _, payload = self.build()
        ct1a = next(p for p in payload["panels"] if p["slice"] == "CT1A")
        self.assertEqual(ct1a["grounded"], [2])
        self.assertEqual(ct1a["missingBase"], [9])

    def test_the_grounded_colour_travels_with_the_payload(self):
        _, payload = self.build()
        self.assertEqual(payload["colors"]["grounded"], fd.GROUNDED_COLOR)

    def test_a_grounded_channel_still_gets_an_x_position(self):
        _, payload = self.build()
        ct1a = next(p for p in payload["panels"] if p["slice"] == "CT1A")
        self.assertIn(2, ct1a["channels"])


class RenderHtml(unittest.TestCase):
    def page(self, filename="R250929CT1A_DIV250_base"):
        panels, _ = panels_of([
            rec(filename, "CTL", 1, 2.0),
            rec(filename.replace("_base", "_stim1"), "CTL", 1, 3.0),
        ])
        return fd.render_html(fd.build_payload(panels, "data.csv"), fd.initial_state())

    def test_every_placeholder_is_substituted(self):
        html = self.page()
        for token in ("__PAYLOAD__", "__INITIAL__", "__PLOTLY_JS_URL__"):
            self.assertNotIn(token, html)
        self.assertIn('id="payload"', html)
        self.assertIn('id="initial"', html)
        self.assertIn(f'src="{fb.PLOTLY_JS_URL}"', html)

    def test_a_script_tag_in_a_file_name_cannot_close_the_json_block(self):
        # The name still has to pair, since only paired recordings reach the
        # payload; the condition is the last underscore-separated token, so a
        # hostile middle segment survives parsing and lands in "baseFile".
        html = self.page("R250929CT1A_DIV250_</script><b>x</b>_base")
        self.assertNotIn("</script><b>", html)
        self.assertIn("<\\/script><b>", html)


class Describe(unittest.TestCase):
    def listing(self, records):
        panels, unpaired = panels_of(records)
        buf = io.StringIO()
        with redirect_stdout(buf):
            fd.describe(panels, unpaired)
        return buf.getvalue()

    def test_a_paired_slice_names_its_conditions_and_counts(self):
        out = self.listing([
            rec("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
            rec("R250929CT1A_DIV250_base", "CTL", 2, 0.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 1, 3.0),
            rec("R250929CT1A_DIV250_stim1", "CTL", 2, 1.0),
            rec("R250929CT1A_DIV250_stim3", "CTL", 9, 1.0),
        ])
        self.assertIn("R250929 CT1A: base + stim1, stim3", out)
        self.assertIn("1 channels paired", out)
        self.assertIn("1 grounded or stimulated (2)", out)
        self.assertIn("1 stim-only", out)
        self.assertIn("1 of 1 slice(s) paired.", out)

    def test_each_unpaired_slice_gives_its_reason(self):
        out = self.listing([
            rec("R250929MT3D_DIV250_base", "MUT", 1, 2.0),
            rec("R250929MO2A_DIV250_stimRL", "MOS", 1, 2.0),
            rec("R250929MO7C_DIV_stim3", "MOS", 13, 0.95),
        ])
        self.assertIn("Not paired:", out)
        self.assertIn("R250929 MT3D: base only (no stim recording)", out)
        self.assertIn("R250929 MO2A: stimRL only (no base recording)", out)
        self.assertIn("condition not parsed from the file name (R250929MO7C_DIV_stim3)", out)
        self.assertIn("0 of 3 slice(s) paired.", out)


class Cli(unittest.TestCase):
    def run_cli(self, tmp, *args, expect_ok=True):
        proc = subprocess.run(
            [sys.executable, str(Path(fd.__file__)), *args],
            capture_output=True, text=True, cwd=tmp,
        )
        if expect_ok:
            self.assertEqual(proc.returncode, 0, proc.stderr)
        else:
            self.assertNotEqual(proc.returncode, 0, proc.stdout)
        return proc

    def initial_of(self, html):
        start = html.index('id="initial">') + len('id="initial">')
        return json.loads(html[start:html.index("</script>", start)])

    def test_it_writes_one_viewer_holding_every_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", PAIRED_ROWS)
            out = Path(tmp) / "diff.html"
            self.run_cli(tmp, str(csv_path), "-o", str(out))
            html = out.read_text()
            self.assertIn('id="payload"', html)
            self.assertIn("Stim LR", html)
            self.assertEqual(len(list(Path(tmp).glob("*.html"))), 1)

    def test_an_unpairable_file_fails_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", [
                ("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
                ("R250929CT2C_DIV250_stim1", "CTL", 1, 3.0),
            ])
            out = Path(tmp) / "diff.html"
            proc = self.run_cli(tmp, str(csv_path), "-o", str(out), expect_ok=False)
            self.assertIn("--list", proc.stderr)
            self.assertFalse(out.exists())

    def test_list_still_works_on_an_unpairable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", [
                ("R250929CT1A_DIV250_base", "CTL", 1, 2.0),
                ("R250929CT2C_DIV250_stim1", "CTL", 1, 3.0),
            ])
            proc = self.run_cli(tmp, str(csv_path), "--list")
            self.assertIn("no stim recording", proc.stdout)
            self.assertIn("no base recording", proc.stdout)

    def test_slice_sets_the_initial_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", PAIRED_ROWS)
            out = Path(tmp) / "diff.html"
            self.run_cli(tmp, str(csv_path), "--slice", "CT1A", "-o", str(out))
            self.assertEqual(self.initial_of(out.read_text())["slice"], "R250929/CT1A")

    def test_an_unknown_slice_is_rejected_with_the_available_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", PAIRED_ROWS)
            proc = self.run_cli(tmp, str(csv_path), "--slice", "CT9Z",
                                "-o", "diff.html", expect_ok=False)
            self.assertIn("R250929/CT1A", proc.stderr)

    def test_dpi_becomes_the_export_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", PAIRED_ROWS)
            out = Path(tmp) / "diff.html"
            self.run_cli(tmp, str(csv_path), "--dpi", "96", "-o", str(out))
            self.assertEqual(self.initial_of(out.read_text())["scale"], 1.0)

    def test_per_panel_y_unticks_the_shared_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", PAIRED_ROWS)
            out = Path(tmp) / "diff.html"
            self.run_cli(tmp, str(csv_path), "--per-panel-y", "-o", str(out))
            self.assertIs(self.initial_of(out.read_text())["sameY"], False)


class SampleFile(unittest.TestCase):
    """Checks against a real-world-shaped export: 12 columns, NaN padding, an FR of
    exactly 0, and a file name whose DIV segment has no digits. Its five rows are
    five different slices, so nothing in it pairs -- which is the point."""

    def rows(self):
        return SAMPLE_CSV.read_text().splitlines()

    def derived(self, tmp, name, stim_row, base_fr):
        """A pairable file: one of the sample's stim rows plus a _base sibling of it,
        over the sample's own 12-column header."""
        header = self.rows()[0]
        stim = next(r for r in self.rows()[1:] if stim_row in r)
        cells = stim.split(",")
        base = list(cells)
        base[0] = cells[0].rsplit("_", 1)[0] + "_base"
        base[4] = str(base_fr)
        path = Path(tmp) / name
        path.write_text("\n".join([header, stim, ",".join(base)]) + "\n")
        return path

    def test_nothing_in_the_sample_pairs(self):
        panels, unpaired = panels_of(fb.load_records(SAMPLE_CSV))
        self.assertEqual(panels, [])
        self.assertIn(("R250929MO7C_DIV_stim3",),
                      [u.conditions for u in unpaired if u.reason == fd.REASON_UNPARSED])

    def test_the_cli_refuses_the_sample_and_writes_no_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "diff.html"
            proc = subprocess.run(
                [sys.executable, str(Path(fd.__file__)), str(SAMPLE_CSV), "-o", str(out)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("nothing to", proc.stderr)
            self.assertFalse(out.exists())

    def test_list_names_every_recording_in_the_sample(self):
        proc = subprocess.run(
            [sys.executable, str(Path(fd.__file__)), str(SAMPLE_CSV), "--list"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for slc in ("CT1A", "MT3D", "CT2C", "MO2A"):
            self.assertIn(slc, proc.stdout)
        self.assertIn("R250929MO7C_DIV_stim3", proc.stdout)
        self.assertIn("0 of 5 slice(s) paired.", proc.stdout)

    def test_a_derived_pair_computes_a_percentage_over_the_real_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "paired.csv", "_stim1", 2.0)
            panels, _ = panels_of(fb.load_records(path))
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0].slice, "CT1A")
            self.assertEqual(len(panels[0].diffs), 1)
            self.assertAlmostEqual(panels[0].diffs[0].pct, 41.722222, places=4)

    def test_a_derived_zero_baseline_grounds_the_channel_and_plots_no_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "zero.csv", "_stimRL", 0)
            panels, _ = panels_of(fb.load_records(path))
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0].diffs, [])
            self.assertEqual(panels[0].grounded[0].readings, (("stimRL", 0.0),))

    def test_the_cli_writes_a_viewer_for_a_derived_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "paired.csv", "_stim1", 2.0)
            out = Path(tmp) / "diff.html"
            proc = subprocess.run(
                [sys.executable, str(Path(fd.__file__)), str(path), "-o", str(out)],
                capture_output=True, text=True, cwd=tmp,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('id="payload"', out.read_text())


if __name__ == "__main__":
    unittest.main()
