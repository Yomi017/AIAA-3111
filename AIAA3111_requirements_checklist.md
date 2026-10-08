# AIAA 3111 要求核对（v3）

| 原项目要求 | 当前状态 | 交付位置 |
|---|---|---|
| 至少属于四类数据挖掘任务之一 | 已满足：anomaly detection | `Report_v2.pdf` 第 2 节 |
| 数据获取 | 已满足：公开 Telemanom 镜像与下载脚本 | `download_hf_telemanom.py`、`research_v2/` |
| 数据预处理 | 已满足：时间索引检查、冲突标注处理、训练段标准化、窗口化 | `experiment_v2.py`、`README_v2.md` |
| 任务定义 | 已满足：输入、输出、异常分数、阈值和指标 | `Report_v2.pdf` 第 2 节 |
| 实验设计与实现 | 已满足：统一支持、开发/确认通道、无 point adjustment | `results_v2/confirmation_frozen.csv` |
| 比较不同方法 | 已满足：Z-score、PCA、IForest、近邻、GRU、DAE、USAD、轨迹密度、监督负对照 | `Report_v2.pdf` 第 4 节 |
| 结果分析、结论和可视化 | 已满足：误报、事件召回、bootstrap、消融、失败案例 | `results_v2/figures/`、`Report_v2.pdf` |
| Report.pdf 主文 6–10 页 | 已生成 7 页 | `results_v2/deliverables/Report_v2.pdf` |
| Report 结构 | 已覆盖 Introduction、Problem Formulation、Methodology、Experiments、Conclusion and Limitations、Disclosures、References、Appendix | `Report_v2.pdf` |
| AIGC 披露 | 已加入；外部协作者写明 None | `Report_v2.pdf` 第 7 节 |
| 代码、README、数据/下载说明 | 已满足 | `NASA_SMAP_MSL_v2_实验与模型.zip` |
| Poster.pdf | 尚未制作，需按老师发布格式单独生成 | 最终另交 `Poster.pdf` |
| Group XX、组员姓名/学号 | 提交前填写 | 报告封面和最终 ZIP 文件名 |

提交前：把报告重命名为 `Report.pdf`，填写组号和成员信息；把 ZIP 重命名为 `Group_XX_Project.zip`；Poster 单独提交且不放入 ZIP。
