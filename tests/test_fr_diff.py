"""Run with: python3 -m unittest discover -t . -s tests"""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fr_boxplots as fb
import fr_diff as fd


SAMPLE_CSV = Path(__file__).resolve().parent / "sample-files" / "NeuronalActivity_NodeLevel_sample.csv"

HEADER = "FileName,Grp,Channel,FR\n"
STIMS = fd.REQUIRED_STIMS


def rec(filename, grp, channel, fr):
    """A Record built the way load_records() builds one, through the real parsers."""
    organoid, slc = fb.parse_organoid(filename)
    return fb.Record(filename, grp, channel, fr, organoid, slc, fb.parse_stim(filename))


def full_set(stem="R250929CT1A_DIV250", grp="CTL", base=None, stims=None, skip=()):
    """Rows of one complete slice: a prestim plus every pattern.

    `base` maps channel -> baseline FR (default {1: 2.0}); `stims` maps a token
    to {channel: FR}, and any pattern it leaves out reads 3.0 on every baseline
    channel. Tokens in `skip` are left out entirely, making the set incomplete.
    """
    base = {1: 2.0} if base is None else base
    stims = stims or {}
    rows = [(f"{stem}_prestim", grp, c, fr) for c, fr in base.items()]
    for t in STIMS:
        if t in skip:
            continue
        for c, fr in stims.get(t, {c: 3.0 for c in base}).items():
            rows.append((f"{stem}_{t}", grp, c, fr))
    return rows


def recs(rows):
    return [rec(*r) for r in rows]


def panels_of(records):
    """build_panels() with its stderr notes swallowed."""
    with redirect_stderr(io.StringIO()):
        return fd.build_panels(records)


def write_csv(tmp, name, rows):
    """Write a four-column CSV of (filename, grp, channel, fr) tuples."""
    path = Path(tmp) / name
    path.write_text(HEADER + "".join(f"{f},{g},{c},{v}\n" for f, g, c, v in rows))
    return path


PAIRED_ROWS = full_set() + full_set("R250929CT2C_DIV250", base={1: 1.0})


class ParseRun(unittest.TestCase):
    def test_strips_the_fused_organoid_marker(self):
        self.assertEqual(fd.parse_run("R250929CT1A_DIV250_prestim"), "R250929")

    def test_malformed_div_segment_still_yields_a_run(self):
        self.assertEqual(fd.parse_run("R250929MO7C_DIV_stim3"), "R250929")

    def test_a_name_without_a_run_number_is_unparseable(self):
        self.assertIsNone(fd.parse_run("CT1A_DIV250_prestim"))


class GroupOf(unittest.TestCase):
    def test_a_prefixed_group_is_the_same_group(self):
        self.assertEqual(fd.group_of("BCTL"), "CTL")
        self.assertEqual(fd.group_of("CTL"), "CTL")
        self.assertEqual(fd.group_of(" bmos "), "MOS")


class PctDiff(unittest.TestCase):
    def test_doubling_is_a_hundred_percent(self):
        self.assertEqual(fd.pct_diff(2.0, 4.0), 100.0)

    def test_halving_is_minus_fifty_percent(self):
        self.assertEqual(fd.pct_diff(4.0, 2.0), -50.0)

    def test_no_change_is_zero_percent(self):
        self.assertEqual(fd.pct_diff(3.0, 3.0), 0.0)

    def test_falling_silent_is_minus_a_hundred_percent(self):
        self.assertEqual(fd.pct_diff(3.0, 0.0), -100.0)


class Completeness(unittest.TestCase):
    def test_a_complete_slice_makes_one_panel(self):
        panels, unpaired = panels_of(recs(full_set()))
        self.assertEqual(unpaired, [])
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0].slice, "CT1A")
        self.assertEqual(panels[0].run, "R250929")
        self.assertEqual(panels[0].stims, ["stim1", "stim3", "stimLR", "stimRL"])
        self.assertEqual({d.pct for d in panels[0].diffs}, {50.0})

    def test_a_slice_missing_a_pattern_is_not_plotted(self):
        panels, unpaired = panels_of(recs(full_set(skip=("stim3", "stimRL"))))
        self.assertEqual(panels, [])
        self.assertEqual(unpaired[0].reason, fd.REASON_INCOMPLETE)
        self.assertEqual(unpaired[0].missing, ("stim3", "stimRL"))

    def test_a_prestim_only_slice_is_not_plotted(self):
        panels, unpaired = panels_of(recs(full_set(skip=STIMS)))
        self.assertEqual(panels, [])
        self.assertEqual(unpaired[0].reason, fd.REASON_NO_STIM)

    def test_a_stim_only_slice_is_not_plotted(self):
        rows = [r for r in full_set() if not r[0].endswith("_prestim")]
        panels, unpaired = panels_of(recs(rows))
        self.assertEqual(panels, [])
        self.assertEqual(unpaired[0].reason, fd.REASON_NO_BASE)

    def test_base_is_not_the_baseline(self):
        rows = [(f.replace("_prestim", "_base"), g, c, v) for f, g, c, v in full_set()]
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, unpaired = fd.build_panels(recs(rows))
        self.assertEqual(panels, [])
        self.assertEqual(unpaired[0].reason, fd.REASON_NO_BASE)
        self.assertIn("base were ignored", buf.getvalue())

    def test_base_beside_prestim_is_ignored(self):
        rows = full_set() + [("R250929CT1A_DIV250_base", "CTL", 1, 99.0)]
        panels, _ = panels_of(recs(rows))
        self.assertEqual({d.base_fr for d in panels[0].diffs}, {2.0})

    def test_an_unknown_condition_is_ignored_with_a_note(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, _ = fd.build_panels(recs(full_set() + [("R250929CT1A_DIV250_drug",
                                                            "CTL", 1, 3.0)]))
        self.assertEqual(panels[0].stims, list(STIMS))
        self.assertIn("drug", buf.getvalue())


class Pairing(unittest.TestCase):
    def test_two_slices_of_one_organoid_do_not_pair(self):
        rows = [r for r in full_set() if r[0].endswith("_prestim")] + \
               [r for r in full_set("R250929CT1B_DIV250") if not r[0].endswith("_prestim")]
        panels, unpaired = panels_of(recs(rows))
        self.assertEqual(panels, [])
        self.assertEqual({u.reason for u in unpaired},
                         {fd.REASON_NO_STIM, fd.REASON_NO_BASE})

    def test_the_same_slice_in_two_runs_does_not_pair(self):
        rows = [r for r in full_set() if r[0].endswith("_prestim")] + \
               [r for r in full_set("R250930CT1A_DIV250") if not r[0].endswith("_prestim")]
        panels, _ = panels_of(recs(rows))
        self.assertEqual(panels, [])

    def test_pairing_ignores_the_group_prefix(self):
        rows = [(f, "BCTL" if f.endswith("_prestim") else "CTL", c, v)
                for f, _, c, v in full_set()]
        panels, _ = panels_of(recs(rows))
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0].grp, "CTL")

    def test_groups_that_really_differ_are_noted(self):
        rows = [(f, "MOS" if f.endswith("_stim1") else "CTL", c, v)
                for f, _, c, v in full_set()]
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, _ = fd.build_panels(recs(rows))
        self.assertEqual(panels[0].grp, "CTL/MOS")
        self.assertIn("disagree on its group", buf.getvalue())

    def test_a_condition_recorded_at_two_divs_is_not_plotted(self):
        rows = full_set() + [("R250929CT1A_DIV251_prestim", "CTL", 6, 9.0)]
        panels, unpaired = panels_of(recs(rows))
        self.assertEqual(panels, [])
        self.assertEqual(unpaired[0].reason, fd.REASON_TWICE)
        self.assertEqual(unpaired[0].conditions, ("prestim",))

    def test_two_unparseable_slices_of_one_run_do_not_pair_with_each_other(self):
        panels, _ = panels_of(recs(full_set("R250929_DIV250")))
        self.assertEqual(panels, [])

    def test_an_unparseable_condition_is_excluded_and_reported(self):
        panels, unpaired = panels_of([rec("R250929MO7C_DIV_stim3", "MOS", 13, 0.95)])
        self.assertEqual(panels, [])
        self.assertEqual([u.reason for u in unpaired], [fd.REASON_UNPARSED])
        self.assertEqual(unpaired[0].conditions, ("R250929MO7C_DIV_stim3",))

    def test_a_duplicate_channel_keeps_the_first_value_and_says_so(self):
        rows = full_set() + [("R250929CT1A_DIV250_prestim", "CTL", 1, 99.0)]
        buf = io.StringIO()
        with redirect_stderr(buf):
            panels, _ = fd.build_panels(recs(rows))
        self.assertEqual({d.base_fr for d in panels[0].diffs}, {2.0})
        self.assertIn("more than one row", buf.getvalue())


class Channels(unittest.TestCase):
    def test_a_channel_missing_from_the_baseline_is_set_aside(self):
        stims = {t: {1: 3.0, 9: 3.0} for t in STIMS}
        panels, _ = panels_of(recs(full_set(stims=stims)))
        self.assertEqual({d.channel for d in panels[0].diffs}, {1})
        self.assertEqual(panels[0].missing_base, [9])

    def test_a_channel_missing_from_a_stim_is_counted(self):
        base = {1: 2.0, 9: 2.0}
        stims = {"stim1": {1: 3.0}}
        panels, _ = panels_of(recs(full_set(base=base, stims=stims)))
        self.assertEqual(panels[0].missing_stim, [9])
        self.assertEqual({d.stim for d in panels[0].diffs if d.channel == 9},
                         set(STIMS) - {"stim1"})

    def test_a_channel_in_no_stim_is_counted_and_plots_nothing(self):
        base = {1: 2.0, 9: 0.0}
        stims = {t: {1: 3.0} for t in STIMS}
        panels, _ = panels_of(recs(full_set(base=base, stims=stims)))
        self.assertEqual(panels[0].missing_stim, [9])
        self.assertEqual(panels[0].excluded, [])

    def test_channels_are_the_sorted_union_of_diffs_and_excluded_channels(self):
        panels, _ = panels_of(recs(full_set(base={7: 2.0, 3: 0.0})))
        self.assertEqual(panels[0].channels, [3, 7])


class Exclusions(unittest.TestCase):
    def test_channel_15_is_grounded_in_every_experiment(self):
        panels, _ = panels_of(recs(full_set(base={1: 2.0, 15: 4.0})))
        self.assertEqual({d.channel for d in panels[0].diffs}, {1})
        [ex] = panels[0].excluded
        self.assertEqual((ex.channel, ex.reason), (15, fd.EXCLUDED_GROUNDED))
        self.assertEqual(len(ex.readings), len(STIMS))
        self.assertEqual(panels[0].unplotted, [15])

    def test_a_zero_baseline_is_excluded_instead_of_giving_a_percentage(self):
        panels, _ = panels_of(recs(full_set(base={1: 0.0})))
        self.assertEqual(panels[0].diffs, [])
        [ex] = panels[0].excluded
        self.assertEqual((ex.channel, ex.reason, ex.base_fr), (1, fd.EXCLUDED_ZERO_BASE, 0.0))

    def test_silence_under_every_stim_is_plotted_at_minus_one_hundred(self):
        stims = {t: {1: 0.0} for t in STIMS}
        panels, _ = panels_of(recs(full_set(stims=stims)))
        self.assertEqual(panels[0].excluded, [])
        self.assertEqual({d.pct for d in panels[0].diffs}, {-100.0})

    def test_a_stimulating_channel_is_excluded_from_its_own_pattern_only(self):
        with mock.patch.dict(fd.STIMULATED_CHANNELS, {"stim1": frozenset({1})}):
            panels, _ = panels_of(recs(full_set()))
        [ex] = panels[0].excluded
        self.assertEqual((ex.reason, ex.readings), (fd.EXCLUDED_STIMULATING, (("stim1", 3.0),)))
        self.assertEqual({d.stim for d in panels[0].diffs}, set(STIMS) - {"stim1"})
        self.assertEqual(panels[0].unplotted, [])

    def test_the_known_stimulating_electrodes_are_excluded_from_every_pattern(self):
        panels, _ = panels_of(recs(full_set(base={1: 2.0, 21: 0.0, 71: 1.5})))
        self.assertEqual({d.channel for d in panels[0].diffs}, {1})
        self.assertEqual({(e.channel, e.reason) for e in panels[0].excluded},
                         {(21, fd.EXCLUDED_STIMULATING), (71, fd.EXCLUDED_STIMULATING)})
        self.assertEqual(panels[0].unplotted, [21, 71])

    def test_a_zero_baseline_on_a_part_stimulating_channel_is_still_named(self):
        with mock.patch.dict(fd.STIMULATED_CHANNELS, {"stim1": frozenset({1}),
                                                      "stim3": frozenset(),
                                                      "stimLR": frozenset(),
                                                      "stimRL": frozenset()}):
            panels, _ = panels_of(recs(full_set(base={1: 0.0})))
        self.assertEqual([e.reason for e in panels[0].excluded],
                         [fd.EXCLUDED_STIMULATING, fd.EXCLUDED_ZERO_BASE])
        self.assertEqual(panels[0].diffs, [])

    def test_a_negative_baseline_is_skipped(self):
        panels, _ = panels_of(recs(full_set(base={1: 2.0, 2: -1.0})))
        self.assertEqual({d.channel for d in panels[0].diffs}, {1})
        self.assertEqual(panels[0].excluded, [])


class PanelTitle(unittest.TestCase):
    def test_one_run_is_named_by_slice_alone(self):
        panels, _ = panels_of(recs(full_set()))
        self.assertFalse(fd.multi_run(panels))
        self.assertEqual(fd.panel_title(panels[0], False), "CT1A")

    def test_two_runs_qualify_the_slice_with_the_run(self):
        panels, _ = panels_of(recs(full_set() + full_set("R250930CT1A_DIV250")))
        self.assertTrue(fd.multi_run(panels))
        self.assertEqual([fd.panel_title(p, True) for p in panels],
                         ["R250929 CT1A", "R250930 CT1A"])


class StimLabel(unittest.TestCase):
    def test_the_four_patterns_have_their_own_names(self):
        self.assertEqual([fd.stim_label(t) for t in ("stim1", "stim3", "stimLR", "stimRL")],
                         ["Stim 1", "Stim 3", "Stim LR", "Stim RL"])

    def test_every_pattern_has_its_own_colour(self):
        self.assertEqual(len(set(fd.STIM_COLORS.values())), len(STIMS))
        self.assertNotIn(fd.EXCLUDED_COLOR, fd.STIM_COLORS.values())


class Payload(unittest.TestCase):
    def build(self):
        panels, _ = panels_of(recs(
            full_set(base={1: 2.0, 2: 0.0},
                     stims={"stim1": {1: 3.0, 2: 1.0, 9: 1.0}})
            + full_set("R250929CT2C_DIV250", base={1: 1.0})))
        return panels, fd.build_payload(panels, "data.csv")

    def test_it_survives_a_json_round_trip(self):
        _, payload = self.build()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_conditions_are_the_four_patterns(self):
        _, payload = self.build()
        self.assertEqual(payload["stims"], list(STIMS))
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
            self.assertEqual(p["n"]["excluded"], len(p["excluded"]))
            self.assertEqual(p["n"]["missingBase"], len(p["missingBase"]))

    def test_an_excluded_channel_is_carried_as_a_bare_channel_number(self):
        _, payload = self.build()
        ct1a = next(p for p in payload["panels"] if p["slice"] == "CT1A")
        self.assertEqual(ct1a["excluded"], [2])
        self.assertEqual(ct1a["missingBase"], [9])
        self.assertIn(2, ct1a["channels"])

    def test_the_excluded_colour_travels_with_the_payload(self):
        _, payload = self.build()
        self.assertEqual(payload["colors"]["excluded"], fd.EXCLUDED_COLOR)


class RenderHtml(unittest.TestCase):
    def page(self, stem="R250929CT1A_DIV250"):
        panels, _ = panels_of(recs(full_set(stem)))
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
        html = self.page("R250929CT1A_DIV250_</script><b>x</b>")
        self.assertNotIn("</script><b>", html)
        self.assertIn("<\\/script><b>", html)


class Describe(unittest.TestCase):
    def listing(self, records):
        panels, unpaired = panels_of(records)
        buf = io.StringIO()
        with redirect_stdout(buf):
            fd.describe(panels, unpaired)
        return buf.getvalue()

    def test_a_plotted_slice_names_its_conditions_and_counts(self):
        out = self.listing(recs(full_set(base={1: 2.0, 2: 0.0, 15: 1.0},
                                         stims={"stim1": {1: 3.0, 2: 1.0, 15: 1.0, 9: 1.0}})))
        self.assertIn("R250929 CT1A: prestim + stim1, stim3, stimLR, stimRL", out)
        self.assertIn("1 channels paired", out)
        self.assertIn("2 excluded: 2 (0 Hz baseline), 15 (grounded)", out)
        self.assertIn("1 stim-only", out)
        self.assertIn("1 of 1 slice(s) plotted.", out)

    def test_a_stimulating_channel_names_its_pattern(self):
        with mock.patch.dict(fd.STIMULATED_CHANNELS, {"stimLR": frozenset({1})}):
            out = self.listing(recs(full_set()))
        self.assertIn("1 (stimulating: stimLR)", out)

    def test_each_unplotted_slice_gives_its_reason(self):
        out = self.listing(recs(
            [r for r in full_set("R250929MT3D_DIV250") if r[0].endswith("_prestim")]
            + [("R250929MO2A_DIV250_stimRL", "MOS", 1, 2.0)]
            + full_set("R250929CT4B_DIV250", skip=("stimLR",))
        ) + [rec("R250929MO7C_DIV_stim3", "MOS", 13, 0.95)])
        self.assertIn("Not plotted:", out)
        self.assertIn("R250929 MT3D: prestim only (no stim recording)", out)
        self.assertIn("R250929 MO2A: stimRL only (no prestim recording)", out)
        self.assertIn("R250929 CT4B: prestim, stim1, stim3, stimRL (incomplete: no stimLR)", out)
        self.assertIn("condition not parsed from the file name (R250929MO7C_DIV_stim3)", out)
        self.assertIn("0 of 4 slice(s) plotted.", out)


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

    def test_an_incomplete_file_fails_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", full_set(skip=("stim1",)))
            out = Path(tmp) / "diff.html"
            proc = self.run_cli(tmp, str(csv_path), "-o", str(out), expect_ok=False)
            self.assertIn("--list", proc.stderr)
            self.assertFalse(out.exists())

    def test_list_still_works_on_an_incomplete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(tmp, "data.csv", full_set(skip=("stim1",)))
            proc = self.run_cli(tmp, str(csv_path), "--list")
            self.assertIn("incomplete: no stim1", proc.stdout)

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

    def derived(self, tmp, name, base_fr):
        """A complete set built from the sample's CT1A stim1 row: that row as-is,
        a prestim sibling reading `base_fr`, and the three other patterns, all
        over the sample's own 12-column header."""
        header = self.rows()[0]
        stim = next(r for r in self.rows()[1:] if r.startswith("R250929CT1A_DIV250_stim1"))
        cells = stim.split(",")
        lines = [header, stim]
        for token, fr in [("prestim", base_fr)] + [(t, cells[4]) for t in STIMS[1:]]:
            row = list(cells)
            row[0] = "R250929CT1A_DIV250_" + token
            row[4] = str(fr)
            lines.append(",".join(row))
        path = Path(tmp) / name
        path.write_text("\n".join(lines) + "\n")
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

    def test_list_names_every_stim_recording_in_the_sample(self):
        proc = subprocess.run(
            [sys.executable, str(Path(fd.__file__)), str(SAMPLE_CSV), "--list"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for slc in ("CT1A", "CT2C", "MO2A"):
            self.assertIn(slc, proc.stdout)
        self.assertIn("R250929MO7C_DIV_stim3", proc.stdout)
        # MT3D's only recording is a _base, which this analysis ignores.
        self.assertNotIn("MT3D", proc.stdout)
        self.assertIn("MT3D", proc.stderr)
        self.assertIn("0 of 4 slice(s) plotted.", proc.stdout)

    def test_a_derived_set_computes_a_percentage_over_the_real_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "paired.csv", 2.0)
            panels, _ = panels_of(fb.load_records(path))
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0].slice, "CT1A")
            self.assertEqual(len(panels[0].diffs), len(STIMS))
            self.assertAlmostEqual(panels[0].diffs[0].pct, 41.722222, places=4)

    def test_a_derived_zero_baseline_is_excluded_and_plots_no_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "zero.csv", 0)
            panels, _ = panels_of(fb.load_records(path))
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0].diffs, [])
            self.assertEqual(panels[0].excluded[0].reason, fd.EXCLUDED_ZERO_BASE)

    def test_the_cli_writes_a_viewer_for_a_derived_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.derived(tmp, "paired.csv", 2.0)
            out = Path(tmp) / "diff.html"
            proc = subprocess.run(
                [sys.executable, str(Path(fd.__file__)), str(path), "-o", str(out)],
                capture_output=True, text=True, cwd=tmp,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('id="payload"', out.read_text())


if __name__ == "__main__":
    unittest.main()
