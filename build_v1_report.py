from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt
root=Path(r'E:/Class/AIAA 3111/Project'); out=root/'results_v1'; out.mkdir(exist_ok=True); old=root/'results_v0'
methods=[]
for ds in ['SMAP','MSL']:
 for x in json.loads((out/f'{ds.lower()}_pca_iforest_summary.json').read_text()): methods.append((ds,x['method'],x['pointwise']['f1'],x['pointwise']['precision'],x['pointwise']['recall'],x['event_level']['recall']))
sm=json.loads((old/'smap_zscore_summary.json').read_text()); ms=json.loads((old/'msl_zscore_summary.json').read_text()); ae=json.loads((old/'smap_lstm_ae_v1_summary.json').read_text())
methods += [('SMAP','z-score',sm['pointwise']['f1'],sm['pointwise']['precision'],sm['pointwise']['recall'],sm['event_level']['recall']),('MSL','z-score',ms['pointwise']['f1'],ms['pointwise']['precision'],ms['pointwise']['recall'],ms['event_level']['recall']),('SMAP','LSTM-AE v1',ae['pointwise']['f1'],ae['pointwise']['precision'],ae['pointwise']['recall'],ae['event_level']['recall'])]
df=pd.DataFrame(methods,columns=['dataset','method','f1','precision','recall','event_recall']); df.to_csv(out/'method_comparison.csv',index=False)
plt.figure(figsize=(10,5)); labels=[f'{d}\n{m}' for d,m in zip(df.dataset,df.method)]; x=range(len(df)); plt.bar([i-.18 for i in x],df.f1,.36,label='Point-wise F1'); plt.bar([i+.18 for i in x],df.event_recall,.36,label='Event recall'); plt.xticks(list(x),labels,rotation=35,ha='right'); plt.ylim(0,1); plt.ylabel('score'); plt.title('NASA SMAP/MSL v1 comparison'); plt.legend(); plt.tight_layout(); plt.savefig(out/'method_comparison.png',dpi=180); plt.close()
report=root/'results_v1'/'完整初版实验报告.md'; lines=['# NASA SMAP/MSL 完整初版实验报告','', '## 1. 本轮完成内容','', '已完成数据下载、官方标签解析、预处理、三类传统 baseline、SMAP LSTM Autoencoder、统一 point-wise 与 event-level 评估，并生成比较图。', '', '## 2. 数据规模','', '| 数据集 | 通道数 | 训练点数 | 测试点数 | 每通道维度 | 官方异常点数 |','|---|---:|---:|---:|---:|---:|','| SMAP | 55 | 140,825 | 444,035 | 25 | 57,043 |','| MSL | 27 | 58,317 | 73,729 | 55 | 7,766 |','', '标签来自官方 anomaly intervals，没有人工注入标签。','', '## 3. 方法','', '- z-score：逐通道最大绝对标准化分数；','- PCA：95% 方差重构误差；','- Isolation Forest：正常训练样本拟合的孤立森林；','- LSTM Autoencoder v1：长度 64 窗口，20 epochs，SMAP 主实验。','', '## 4. 对比结果','', '| 数据集 | 方法 | Precision | Recall | F1 | FPR | Event Recall |','|---|---|---:|---:|---:|---:|---:|']
for d,m,f,p,r,e in methods:
 # lookup fpr
 if m=='z-score': s=sm if d=='SMAP' else ms; fpr=s['pointwise']['fpr']
 elif m=='LSTM-AE v1': fpr=ae['pointwise']['fpr']
 else:
  js=json.loads((out/f'{d.lower()}_pca_iforest_summary.json').read_text()); s=[q for q in js if q['method']==m][0]; fpr=s['pointwise']['fpr']
 lines.append(f'| {d} | {m} | {p:.3f} | {r:.3f} | {f:.3f} | {fpr:.3f} | {e:.3f} |')
lines += ['', '## 5. 结论', '', '当前最好的 point-wise 结果是 SMAP z-score，F1=0.369。PCA 的 event recall 较高，但 point-wise 误报较多；Isolation Forest 在当前阈值下召回较低。LSTM-AE v1 虽然训练误差下降，但 anomaly ranking 还没有学好，F1=0.095，仍低于 z-score。', '', '这说明项目流程已经可行，但深度模型尚未达到最终状态。下一轮应优先改进 anomaly score 的时间对齐、阈值选择和训练目标，再考虑 GRU/Transformer。', '', '## 6. 可视化', '', '方法对比图见 `method_comparison.png`；SMAP A-1 的遥测与官方异常区间图、z-score 分数图保存在 `results_v0`。', '', '## 7. 可复现文件', '', '- `run_baseline_v0.py`：z-score baseline；', '- `run_msl_baseline_v0.py`：MSL z-score；', '- `run_pca_iforest_v1.py`：PCA 和 Isolation Forest；', '- `run_lstm_v1.py`：LSTM Autoencoder v1；', '- `method_comparison.csv`：统一结果表。']
report.write_text('\n'.join(lines),encoding='utf8'); print(report)
