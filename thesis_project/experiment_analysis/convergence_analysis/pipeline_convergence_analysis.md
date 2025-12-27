# pipeline_convergence_analysis

## 1.Convergence and Post-Peak Behaviour Analysis Pipeline

### 1.1 Input Data Sources

1.1.1 Training Run Directory

All convergence-related metrics are computed from completed training runs
stored under the directory:

    thesis_project/runs_seed0/

The directory follows the hierarchical structure:

    <ENV>/<ALGO>/<RUN_ID>/

where:
- <ENV> denotes the environment name (e.g. CartPole-v1, Hopper-v4);
- <ALGO> denotes the algorithm (a2c or ppo);
- <RUN_ID> uniquely identifies a single hyperparameter configuration and seed.

1.1.2 Evaluation Logs

For each run, convergence metrics are derived exclusively from the evaluation
logs produced during training, located at:

    <RUN_ID>/eval/evaluations.npz

This file contains:
- timesteps: evaluation timesteps;
- results or mean_returns: evaluation episode returns.

No training rewards, checkpoints, or intermediate policy snapshots are used
in this analysis.

Only runs with a sufficient number of evaluation points are considered.
Runs with fewer than three evaluation points are excluded from the strict
analysis.


1.2 Per-Run Metric Computation

1.2.1 Script: compute_convergence_metrics.py

The script:

    experiment_analysis/convergence_analysis/compute_convergence_metrics.py

iterates over all runs in runs_seed0 and computes convergence metrics
independently for each run.

1.2.2 Preprocessing

For each run:
- the evaluation return curve is extracted from evaluations.npz;
- a moving-average smoothing with a fixed window size is applied to reduce
  high-frequency stochastic fluctuations.

1.2.3 Strict Convergence Metrics

Using the definitions in Section 4.3.3, the following quantities are computed:

- R_max:
    the maximum of the smoothed evaluation curve;

- R_end:
    the mean of the final 5--10 smoothed evaluation points;

- Delta_post:
    Delta_post = R_end - R_max;

- s_final:
    the average slope over the final 10% of evaluation points.

Based on these quantities, each run is classified under a strict criterion
into one of the following categories:
- post_peak_degradation;
- uncertain.

Due to the narrow tolerance of the strict criterion, no explicit
"converged_strict" category is observed in practice.

1.2.4 Relaxed Convergence Metrics

To assess robustness, a relaxed variant of the convergence criterion is also
computed for each run:

- the peak performance is estimated using a high quantile of the smoothed
  evaluation curve rather than a single maximum;
- a larger tolerance for post-peak performance degradation is allowed;
- the slope threshold for late-phase stability is relaxed.

Each run is classified into one of:
- converged_relaxed;
- post_peak_degradation;
- uncertain.

Runs with insufficient evaluation points for the relaxed computation are
marked as not_computed.

1.2.5 Per-Run Output

For each run, the script records:
- environment;
- algorithm;
- run identifier and seed;
- strict convergence metrics and label;
- relaxed convergence metrics and label.


1.3 Aggregated Output Files

1.3.1 Per-Run Metrics Table

The per-run results are written to:

    experiment_analysis/convergence_analysis/convergence_metrics.csv

Each row corresponds to a single training run and contains both strict and
relaxed convergence metrics.

This file serves as the single source of truth for all subsequent convergence
analyses.

1.3.2 Aggregation Script

The script:

    experiment_analysis/convergence_analysis/summarise_convergence.py

reads convergence_metrics.csv and aggregates results across runs.

1.3.3 Aggregated Counts

For both strict and relaxed criteria, the number of runs assigned to each
behavioural category is computed per (environment, algorithm) pair and saved
to:

    convergence_trend_counts_strict.csv
    convergence_trend_counts_relaxed.csv

1.3.4 Aggregated Fractions

The counts are normalised to fractions to allow comparison across environments
with different numbers of runs. The resulting tables are saved to:

    convergence_trend_fractions_strict.csv
    convergence_trend_fractions_relaxed.csv


1.4 Visualisation

1.4.1 Relaxed Convergence Bar Plot

The script:

    plot_relaxed_convergence_bars.py

reads convergence_trend_fractions_relaxed.csv and generates a stacked bar chart
showing the fraction of runs classified as:
- converged_relaxed;
- post_peak_degradation;
- uncertain

for each environment–algorithm pair.

The resulting figure is saved as:

    relaxed_convergence_stacked_bars_clean.png

and is included in Section 5.4 of the thesis.

1.4.2 Diagnostic Curves (Optional)

For qualitative inspection, individual evaluation curves can be plotted
directly from evaluations.npz using auxiliary scripts (e.g. raw_curve.py).
These plots are not used quantitatively in the analysis and serve only for
sanity checking and illustrative purposes.


1.5 Usage in the Thesis

- Section 4.3.3 defines the convergence metrics implemented in
  compute_convergence_metrics.py;
- Section 5.4 reports the aggregated strict and relaxed convergence results
  based on the CSV outputs produced by this pipeline;
- Tables in Section 5.4 are generated from
  convergence_trend_fractions_strict.csv and
  convergence_trend_fractions_relaxed.csv;
- The relaxed convergence bar plot provides a visual summary of the
  distribution of convergence behaviours across environments and algorithms.
