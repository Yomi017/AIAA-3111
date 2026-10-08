from pathlib import Path
import pandas as pd, json
root=Path(r'E:/Class/AIAA 3111/Project'); data=root/'data/telemanom/data'; out=root/'results_v0'
labels=pd.read_csv(data/'../labeled_anomalies.csv')
rows=[]
for sp in ['SMAP','MSL']:
 ids=labels[labels.spacecraft==sp].chan_id.tolist(); tr=[];te=[];dims=[]
 for c in ids:
  a=pd.read_parquet(data/'train'/f'{c}.parquet');b=pd.read_parquet(data/'test'/f'{c}.parquet');tr.append(len(a));te.append(len(b));dims.append(a.shape[1]-1)
 rows.append((sp,len(ids),sum(tr),sum(te),set(dims),int(labels[labels.spacecraft==sp].apply(lambda r: sum(int(x[1])-int(x[0])+1 for x in __import__('json').loads(r.anomaly_sequences.replace("'",'"'))),axis=1).sum())))
sm=json.loads((out/'smap_zscore_summary.json').read_text()); ms=json.loads((out/'msl_zscore_summary.json').read_text()); ae=json.loads((out/'smap_lstm_ae_summary.json').read_text())
report=f'''# NASA SMAP/MSL 初版实验报告\n\n## 1. 本轮完成内容\n\n已下载并整理 NASA SMAP/MSL 数据集及官方 anomaly intervals。当前完成了 SMAP、MSL 的逐通道 z-score baseline，以及 SMAP 的轻量 LSTM Autoencoder 初版。\n\n## 2. 数据检查\n\n| 数据集 | 通道数 | 训练点数 | 测试点数 | 每通道维度 | 标注异常点数 |\n|---|---:|---:|---:|---:|---:|\n'''
for r in rows: report+=f'| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} | {sorted(r[4])} | {r[5]:,} |\n'
report+='''\n标签由官方异常区间转换为逐时间点标签，没有人工注入标签。\n\n## 3. 方法\n\n### 3.1 z-score baseline\n\n对每个通道的每个输入维度，只使用训练集计算均值和标准差：\n\n\\[z_t=\\max_j\\left|\\frac{x_{t,j}-\\mu_j}{\\sigma_j}\\right|\\]\n\n阈值取训练序列最后 20% 的分数 99.5% 分位数。\n\n### 3.2 LSTM Autoencoder\n\n使用长度为 64 的多变量窗口，训练 5 个 epoch，模型学习正常窗口的重构误差。\n\n\\[e_t=\\operatorname{MSE}(X_t,\\hat X_t)\\]\n\n阈值取验证窗口误差的 99.5% 分位数。\n\n## 4. 初步结果\n\n| 数据集/模型 | Precision | Recall | F1 | FPR | Event Recall |\n|---|---:|---:|---:|---:|---:|\n'''
report+=f"| SMAP z-score | {sm['pointwise']['precision']:.3f} | {sm['pointwise']['recall']:.3f} | {sm['pointwise']['f1']:.3f} | {sm['pointwise']['fpr']:.3f} | {sm['event_level']['recall']:.3f} |\n"
report+=f"| MSL z-score | {ms['pointwise']['precision']:.3f} | {ms['pointwise']['recall']:.3f} | {ms['pointwise']['f1']:.3f} | {ms['pointwise']['fpr']:.3f} | {ms['event_level']['recall']:.3f} |\n"
report+=f"| SMAP LSTM-AE v0 | {ae['pointwise']['precision']:.3f} | {ae['pointwise']['recall']:.3f} | {ae['pointwise']['f1']:.3f} | {ae['pointwise']['fpr']:.3f} | {ae['event_level']['recall']:.3f} |\n"
report+='''\n## 5. 结果解释\n\nSMAP 的 z-score baseline 当前 F1 为 0.369，说明最简单的统计方法已经能捕捉部分异常，但误报仍然存在。MSL 的 point-wise F1 较低，说明不同通道的尺度和异常形态更复杂；不过 event-level recall 较高，表示不少异常事件至少被命中了一个时间点。\n\nSMAP LSTM-AE v0 的 F1 为 0.050，明显低于 z-score baseline。这不是最终结论，而是一个可复现的初版结果，主要说明当前 5 epoch、全局模型和 99.5% 阈值设置仍然欠拟合或不适合直接重构全部窗口。后续需要调整训练轮数、阈值选择、窗口评分方式，并加入 PCA/Isolation Forest 作为中间 baseline。\n\n## 6. 当前限制\n\n1. 目前是 point-wise max z-score，尚未完成 PCA 和 Isolation Forest。\n2. LSTM-AE 只进行了轻量初跑，不能作为最终模型结论。\n3. 当前 SMAP/MSL 结果使用逐通道评估，尚未做更严格的跨通道上下文消融。\n4. 事件级指标需要在报告中固定定义，避免只依赖 point-wise accuracy。\n\n## 7. 下一轮工作\n\n1. 加入 PCA reconstruction baseline。\n2. 调整 LSTM-AE 的训练轮数和阈值，并测试预测残差版本。\n3. 加入 Isolation Forest。\n4. 统一 SMAP/MSL 的事件级评估和可视化。\n5. 形成第一版 report、README 和 poster 图表。\n'''
(root/'results_v0'/'初版实验报告.md').write_text(report,encoding='utf8')
print(report)
