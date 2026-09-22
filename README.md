# MEA-NAP Scripts

Utility scripts for post-processing output from [MEA-NAP](https://github.com/SAND-Lab/MEA-NAP), the MATLAB pipeline for analyzing microelectrode array (MEA) recordings. These scripts operate on MEA-NAP's CSV exports to handle specific downstream tasks: merging a run's per-condition exports, filtering recording groups, visualizing per-channel firing rate distributions, and checking normality of selected metrics ahead of statistical testing.

## Scripts

### `fr_boxplots.py`
Plots per-channel firing rate distributions from a MEA-NAP node-level CSV (`FileName`, `Grp`, `Channel`, `FR`). By default it opens an interactive viewer in your web browser, with dropdowns for group and organoid, hover tooltips identifying the source recording of each point, and a toolbar button that saves the current view as a high-resolution PNG. The viewer is a single self-contained HTML file; it loads the Plotly.js charting library from the internet the first time it runs on a machine. Static figures can also be rendered directly to a file.

Each stim type / condition in the file (e.g. `stim1`, `stim3`, `base`) is analysed as its own experiment, with its own boxes, organoids and slices computed from that condition's recordings alone. A CSV holding several conditions therefore produces several viewers/figures — one browser tab per condition, or one output file per condition when saving (`-o viewer.html` becomes `viewer_stim1.html`, `viewer_stim3.html`, ...). Use `--stim` to restrict the run to a single condition.

```
python3 fr_boxplots.py NeuronalActivity_NodeLevel.csv          # interactive viewer (opens in browser)
python3 fr_boxplots.py data.csv --list                         # list stims/groups/organoids/slices
python3 fr_boxplots.py data.csv -o viewer.html                 # write the interactive viewer(s) to file(s) to share
python3 fr_boxplots.py data.csv --stim stim1 -o viewer.html    # a single condition only
python3 fr_boxplots.py data.csv --grp BCTL --organoid CT7 -o bctl_ct7.png   # save a static figure
```

### `merge_csv.py`
Merges two or more MEA-NAP CSV exports that share a column layout and a recording run into one file — typically the per-condition node-level exports of a single run (`_base`, `_stim1`, `_stim3`, `_stimLR`, `_stimRL`).

Nothing is written until three checks pass. Every file must carry the same set of columns (compared ignoring case and order; the merged file uses the first file's spelling and order); every row must come from the same run, identified by the leading `R<digits>` token of its `FileName` — `R250929` in `R250929CT7A_DIV250_stim1`; and no recording may appear in more than one file. A file holding two runs, or one from a different run than the rest, stops the merge, as does an overlap between inputs — a `FileName` repeats within a file once per channel, which is expected, but the same `FileName` in two files would double that recording's rows. Rows are otherwise passed through untouched and in the order given: no de-duplication and no reordering. The inputs are never modified; the default output is `NeuronalActivity_NodeLevel_base_stim_merged.csv`, written beside the first input.

```
python3 merge_csv.py base.csv stim1.csv stim3.csv
python3 merge_csv.py stimLR.csv stimRL.csv -o combined.csv
```

### `grp_filter.py`
Removes rows whose `Grp` value starts with `C` (e.g. `CCTL`, `CMOS`, `CMUT`) from a CSV. Lists the rows to be dropped and asks for confirmation before writing a new file; the input is never modified.

```
python3 grp_filter.py input.csv
python3 grp_filter.py input.csv -o filtered.csv
```

### `assess_normality.R`
Runs Q-Q plots, Shapiro-Wilk, and/or Kolmogorov-Smirnov tests on selected numeric columns, broken out by `Grp`. Q-Q plots are saved as PNGs; test statistics print to the console.

```
./assess_normality.R input.csv --columns FR,Burst_Rate
./assess_normality.R input.csv --columns FR --tests shapiro,ks --groups BCTL,BMOS
```

## Installation

The Python scripts depend only on `matplotlib`. `assess_normality.R` uses base R and requires no additional packages.

### Instructions (Terminal)

With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/mfang21/meanap-scripts.git
cd meanap-scripts
uv sync
```

With pip:

```
git clone https://github.com/mfang21/meanap-scripts.git
cd meanap-scripts
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

R scripts run with any current R installation: `Rscript assess_normality.R ...`.

### Instructions (General)

1. Install [Python](https://www.python.org/downloads/) (3.10 or later).
2. Install [R](https://cran.r-project.org/) if you plan to run `assess_normality.R`.
3. Download this repository: on this page, click **Code → Download ZIP**, then unzip it.
4. Open Terminal (macOS) or Command Prompt (Windows) and navigate to the unzipped folder:
   ```
   cd path/to/meanap-scripts
   ```
5. Install the required Python library:
   ```
   pip install -r requirements.txt
   ```
6. Run a script:
   ```
   python3 fr_boxplots.py your_data.csv
   ```
   On Windows, run the R script as `Rscript assess_normality.R your_data.csv --columns FR`.
