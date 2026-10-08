# NASA SMAP/MSL Project v0

## Current status

- Dataset downloaded from the Telemanom-compatible Hugging Face mirror.
- Official anomaly intervals retained in `data/telemanom/labeled_anomalies.csv`.
- SMAP and MSL z-score baselines completed.
- SMAP LSTM Autoencoder v0 completed on CUDA.
- Initial report and metrics are in `results_v0`.

## Main scripts

- `run_baseline_v0.py`: SMAP z-score baseline
- `run_msl_baseline_v0.py`: MSL z-score baseline
- `run_lstm_v0.py`: SMAP LSTM Autoencoder v0
- `build_initial_report.py`: generate the initial report

## Main results

See `results_v0/初版实验报告.md`, `smap_zscore_summary.json`, `msl_zscore_summary.json`, and `smap_lstm_ae_summary.json`.

The LSTM result is a feasibility run, not the final model. It currently underperforms the statistical baseline, so the next iteration should tune the threshold, scoring alignment, training length, and model objective before drawing conclusions.
