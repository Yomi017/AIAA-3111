from pathlib import Path
import pandas as pd, numpy as np, json
import matplotlib.pyplot as plt
root=Path(r'E:/Class/AIAA 3111/Project'); data=root/'data/telemanom/data'; out=root/'results_v0'
lab=pd.read_csv(data/'../labeled_anomalies.csv'); chan='A-1'; tr=pd.read_parquet(data/'train'/f'{chan}.parquet'); te=pd.read_parquet(data/'test'/f'{chan}.parquet'); feat='value'; xtr=tr[feat].to_numpy(float); xte=te[feat].to_numpy(float); mu=xtr.mean();sd=xtr.std() or 1; score=np.abs((xte-mu)/sd); y=np.zeros(len(xte)); row=lab[lab.chan_id==chan].iloc[0]; seqs=json.loads(row.anomaly_sequences.replace("'",'"'))
plt.figure(figsize=(12,4)); plt.plot(xte,color='#4c78a8',lw=.7,label='telemetry');
for a,b in seqs: plt.axvspan(a,b,color='#e45756',alpha=.22)
plt.title('SMAP A-1 test telemetry with official anomaly intervals'); plt.xlabel('timestep'); plt.ylabel('value'); plt.legend(); plt.tight_layout(); plt.savefig(out/'smap_A-1_telemetry_anomalies.png',dpi=160); plt.close()
plt.figure(figsize=(12,4)); plt.plot(score,color='#f58518',lw=.7,label='absolute z-score');
for a,b in seqs: plt.axvspan(a,b,color='#e45756',alpha=.22)
th=float(pd.read_json(out/'smap_zscore_summary.json').get('threshold',0)) if False else 0
plt.title('SMAP A-1 z-score anomaly score'); plt.xlabel('timestep'); plt.ylabel('score'); plt.legend(); plt.tight_layout(); plt.savefig(out/'smap_A-1_zscore_score.png',dpi=160); plt.close()
readme='''# NASA SMAP/MSL Project v0\n\n## Current status\n\n- Dataset downloaded from the Telemanom-compatible Hugging Face mirror.\n- Official anomaly intervals retained in `data/telemanom/labeled_anomalies.csv`.\n- SMAP and MSL z-score baselines completed.\n- SMAP LSTM Autoencoder v0 completed on CUDA.\n- Initial report and metrics are in `results_v0`.\n\n## Main scripts\n\n- `run_baseline_v0.py`: SMAP z-score baseline\n- `run_msl_baseline_v0.py`: MSL z-score baseline\n- `run_lstm_v0.py`: SMAP LSTM Autoencoder v0\n- `build_initial_report.py`: generate the initial report\n\n## Main results\n\nSee `results_v0/初版实验报告.md`, `smap_zscore_summary.json`, `msl_zscore_summary.json`, and `smap_lstm_ae_summary.json`.\n\nThe LSTM result is a feasibility run, not the final model. It currently underperforms the statistical baseline, so the next iteration should tune the threshold, scoring alignment, training length, and model objective before drawing conclusions.\n'''
(root/'README_v0.md').write_text(readme,encoding='utf8')
