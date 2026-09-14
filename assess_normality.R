#!/usr/bin/env Rscript
#
# assess_normality.R -- normality diagnostics for CSV columns, per group.
#
# Runs any combination of a normal Q-Q plot, the Shapiro-Wilk test, and a
# one-sample Kolmogorov-Smirnov test on the requested columns of a CSV,
# separately for each value of its "Grp" column.
#
# Usage:
#   ./assess_normality.R <input.csv> --columns <c1,c2,...>
#                        [--tests <qq,shapiro,ks>] [--groups <g1,g2,...>]
#                        [--outdir <dir>] [--no-view]
#
# Options:
#   --columns   Required. Comma-separated column names to assess.
#   --tests     Subset of qq, shapiro, ks. Default: all three.
#   --groups    Subset of the values in "Grp". Default: every distinct value.
#   --outdir    Base directory for plots. Default: the input file's directory.
#   --no-view   Do not open saved plots in the system image viewer.
#
# Q-Q plots are written to <outdir>/qq_plots/YYMMDD_<group>_<column>_qqplot.png.
#
# Non-numeric and missing values are dropped before testing. A group/column
# with fewer than 3 usable observations is skipped, and Shapiro-Wilk is
# skipped above 5000 (R's limit). The K-S test uses the sample mean and SD
# as its reference normal, which biases its p-value upward; treat it as a
# rough check alongside the other two.
#
# Exits 1 with a message on invalid arguments or input.

VALID_TESTS <- c("qq", "shapiro", "ks")
USAGE <- "Usage: ./assess_normality.R <input.csv> --columns <c1,c2,...> [--tests <qq,shapiro,ks>] [--groups <g1,g2,...>] [--outdir <dir>] [--no-view]"

die <- function(fmt, ...) {
  message(sprintf(fmt, ...))
  quit(status = 1)
}

split_csv <- function(x) trimws(strsplit(x, ",")[[1]])

sanitize <- function(x) gsub("[^A-Za-z0-9_-]+", "_", x)

# Returns the requested subset of `valid`, or all of `valid` when nothing was requested.
choose <- function(requested, valid, what) {
  if (length(requested) == 0) return(valid)
  chosen <- split_csv(requested)
  unknown <- setdiff(chosen, valid)
  if (length(unknown) > 0) {
    die("Unknown %s: %s\nAvailable %s: %s", what, toString(unknown), what, toString(valid))
  }
  chosen
}

parse_args <- function(args) {
  opts <- list(no_view = FALSE)
  i <- 1
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--no-view") {
      opts$no_view <- TRUE
      i <- i + 1
    } else if (a %in% c("--tests", "--groups", "--columns", "--outdir")) {
      if (i == length(args)) die("Flag %s requires a value.\n%s", a, USAGE)
      opts[[substring(a, 3)]] <- args[i + 1]
      i <- i + 2
    } else if (is.null(opts$input)) {
      opts$input <- a
      i <- i + 1
    } else {
      die("Unrecognized argument '%s'.\n%s", a, USAGE)
    }
  }
  if (is.null(opts$input)) die(USAGE)
  opts
}

save_qq_plot <- function(values, path, title) {
  png(path, width = 600, height = 600)
  on.exit(dev.off())
  qqnorm(values, main = title)
  qqline(values, col = "red")
}

open_in_viewer <- function(path) {
  opener <- switch(Sys.info()[["sysname"]], Darwin = "open", Linux = "xdg-open", NULL)
  if (!is.null(opener)) {
    system2(opener, shQuote(path), stdout = FALSE, stderr = FALSE, wait = FALSE)
  }
}

report <- function(label, stat_name, result) {
  verdict <- if (result$p.value < 0.05) "reject normality" else "fail to reject normality"
  cat(sprintf("%s: %s = %.5f, p-value = %.5g (%s)\n",
              label, stat_name, result$statistic, result$p.value, verdict))
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))

if (!file.exists(opts$input)) die("Input file '%s' does not exist.", opts$input)
data <- read.csv(opts$input, stringsAsFactors = FALSE, check.names = FALSE)
if (!"Grp" %in% names(data)) die("Input CSV has no 'Grp' column.")
if (is.null(opts$columns)) die("--columns is required.\nAvailable columns: %s", toString(names(data)))

grp <- as.character(data[["Grp"]])
tests <- choose(tolower(opts$tests), VALID_TESTS, "tests")
groups <- choose(opts$groups, unique(grp), "groups")
columns <- choose(opts$columns, names(data), "columns")

if ("qq" %in% tests) {
  plot_dir <- file.path(if (is.null(opts$outdir)) dirname(opts$input) else opts$outdir, "qq_plots")
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("\nQ-Q plots will be saved to: %s\n", plot_dir))
}
if ("ks" %in% tests) {
  cat("Note: K-S uses the sample mean/SD as its reference normal, which inflates the p-value.\n")
}

for (group in groups) {
  rows <- data[grp %in% group, , drop = FALSE]

  for (column in columns) {
    cat(sprintf("\n=== Group: %s | Column: %s ===\n", group, column))

    values <- suppressWarnings(as.numeric(rows[[column]]))
    dropped <- sum(is.na(values))
    values <- values[!is.na(values)]
    if (dropped > 0) cat(sprintf("Dropped %d non-numeric/NA value(s).\n", dropped))
    if (length(values) < 3) {
      cat(sprintf("Skipped: %d usable observation(s), need at least 3.\n", length(values)))
      next
    }
    cat(sprintf("N: %d\n", length(values)))

    if ("qq" %in% tests) {
      path <- file.path(plot_dir, sprintf("%s_%s_%s_qqplot.png",
                                          format(Sys.Date(), "%y%m%d"), sanitize(group), sanitize(column)))
      save_qq_plot(values, path, sprintf("Normal Q-Q Plot\nGroup: %s, Column: %s", group, column))
      if (!opts$no_view) open_in_viewer(path)
    }

    if ("shapiro" %in% tests) {
      if (length(values) > 5000) cat("Shapiro-Wilk: skipped (n > 5000).\n")
      else report("Shapiro-Wilk", "W", shapiro.test(values))
    }

    if ("ks" %in% tests) {
      report("Kolmogorov-Smirnov", "D",
             suppressWarnings(ks.test(values, "pnorm", mean(values), sd(values))))
    }
  }
}
