# MEA-NAP Scripts

Utility scripts for post-processing output from [MEA-NAP](https://github.com/SAND-Lab/MEA-NAP), the MATLAB pipeline for analyzing microelectrode array (MEA) recordings. These scripts operate on MEA-NAP's CSV exports to handle specific downstream tasks: filtering recording groups, visualizing per-channel firing rate distributions, and checking normality of selected metrics ahead of statistical testing.

## Scripts

### `fr_boxplots.py`
Plots per-channel firing rate distributions from a MEA-NAP node-level CSV (`FileName`, `Grp`, `Channel`, `FR`). Opens an interactive viewer by default, with dropdowns for group and organoid and hover tooltips identifying the source recording of each point. Can also render directly to a file.

```
python3 fr_boxplots.py NeuronalActivity_NodeLevel.csv          # interactive viewer
python3 fr_boxplots.py data.csv --list                         # list groups/organoids/slices
python3 fr_boxplots.py data.csv --grp BCTL --organoid CT7 -o bctl_ct7.png   # save a figure
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

1. Install [Python](https://www.python.org/downloads/) (3.10 or later). Python.org builds include `tkinter`, which the `fr_boxplots.py` viewer requires.
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
