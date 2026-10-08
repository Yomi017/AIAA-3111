# NASA SMAP/MSL v2：正常轨迹表示与稳定局部密度异常检测

本轮交付：完整实验、真实结果、研究依据、80 个已拟合模型、逐点预测、可视化和复现入口。任务属于课程要求的 **anomaly detection**。

首先打开 `results_v2/deliverables/实验结果与方法说明.html`。图表在 `results_v2/figures/`。

## 实测结果与证据边界

58 个确认通道（39 SMAP、19 MSL），没有 point adjustment：

| 方法 | SMAP F1 | MSL F1 | 两数据集等权平均 |
|---|---:|---:|---:|
| 正常子序列近邻基线 | 0.3782 | 0.2448 | 0.3115 |
| 开发集预选：稳定原始轨迹 LOF-50 | 0.3629 | 0.2383 | 0.3006 |
| 探索性候选：正常子空间稳定密度 | **0.4871** | **0.2659** | **0.3765** |

最后一行是确认比较后重点报告的候选，不能包装成预选方法在独立最终测试上获胜。其参数预先固定，数值稳定修复来自前缀一致性失败，而不是测试 F1 搜索。所有配置、失败方法和修订均保留。MSL 最强的时序 PCA 为 0.3035，候选并未全面领先。候选 MSL 误报率 32.9%，SMAP 事件召回 66.0%，仍有明显问题。

全 80 通道（含开发集，描述性）：候选 F1 为 SMAP 0.4564、MSL 0.2296，等权平均 0.3430；GRU-target 参考基线平均 0.3016。完整比较、bootstrap 区间、消融及三个神经随机种子见 HTML。

## 直接运行已拟合模型

当前机器使用 `C:\ProgramData\anaconda3\python.exe`。代码兼容 CPU；GPU 仅用于神经对照训练。当前实际训练设备是 RTX 5080，不是四张 4090。

在项目目录打开 PowerShell：

```powershell
$pythonExe = 'C:\ProgramData\anaconda3\python.exe'
& $pythonExe .\detector_v2.py predict `
  --model .\results_v2\models\subspace_lof50\P-1.joblib `
  --input .\data\telemanom\data\test\P-1.parquet `
  --output .\results_v2\P-1_inference.csv
```

输入支持：带 `value` 列的 Parquet/CSV，或目标值在第一列的 NPY。每个通道要配套使用自己的模型；不要把另一种遥测传感器输入 P-1 模型。输入值应遵循该基准的同一数据尺度，不应把未知物理单位的数据直接送入。模型内保存训练标准化、PCA、邻居参考库和阈值。

输出 `timestep,value,score,threshold,anomaly`。最前 64 点为 warmup，之后每点都有结果。预测不读取异常标签。最终方法只使用 `value`；指令输入已在 GRU 对照中评估。

## 从正常数据重新拟合最终候选

```powershell
& $pythonExe .\detector_v2.py fit --method subspace_lof50
& $pythonExe .\verify_exports_v2.py
& $pythonExe .\test_protocol_v2.py
```

候选没有随机初始化；无需重复种子。80 个模型已通过与实验评分一致、逐点告警一致、前缀因果一致检查。协议有 8 个单元测试。joblib 文件仅加载本包内可信生成的文件。

## 全流程复现

依赖版本见 `requirements_v2.txt` 和 `results_v2/provenance.json`。若已有该 Python 环境，无需再次安装。换机器时先准备适配硬件的 PyTorch，再安装其余依赖。

```powershell
.\reproduce_v2.ps1 -PythonExe 'C:\ProgramData\anaconda3\python.exe' -Fresh
```

`-Fresh` 会在项目下创建新的时间戳目录，复制代码、数据及冻结配置，在里面重新训练和评估；不会删除已有结果。不加 `-Fresh` 时复用已有 `done.json`，适合验证和重新生成报告。不要通过更换同名运行目录的 seed／epochs 来混合两套结果。

大致顺序：

1. 数据与异常区间检查；正常数据划分；基线及单步 GRU。
2. 多步 GRU、去噪 AE、USAD 改编、时序子空间和稳定密度模型。
3. 按已冻结配置评估开发／确认集，不再搜索确认标签阈值。
4. 神经基线另外运行 seed 7 和 2026。
5. 导出 80 个候选模型、重载检查、协议测试。
6. 汇总、配对通道 bootstrap、消融、来源哈希和图表报告。

完整复现会重新生成大量逐点结果。zip 中已附模型和结果，阅读与推理不需要重新训练。当前只使用一张本地显卡，最终密度方法在 CPU 就能运行。

## 评估约定

- 使用官方训练／测试划分。P-2 的两条冲突标注排除；T-10 不在有效评估标注清单。有效通道共 80 个。
- 原始公开数据已按测试 min/max 缩放：这是继承的预处理限制，不能声称完全无测试信息的原始工程评估。
- 正常训练序列通常 60% 拟合、20% 正常选模型、20% 校准。短通道后两段至少 96 点。所有新增标准化仅拟合前段。
- 通道 SHA-256 模 3 决定开发／确认划分，开发标签只用于配置选择。之前 v0/v1 看过全部官方测试，故本轮确认集不是完全未见的数据。
- 每个方法从同一测试索引 64 开始逐点评分。闭区间标注，真实区间不做预测补齐。
- 每个方法在开发集比较 span 1/16/64、正常校准或简化动态阈值后固定配置；最终表中所有方法均选正常校准 99.5% 分位。
- 动态阈值用完整无标签测试分数，是离线／传导式候选；本轮胜出方法不使用它。
- 主要 F1 为先在每个数据集汇总 TP/FP/FN，再计算 F1；最终“平均”为两个数据集 F1 的算术平均。
- AP 为逐通道 AP 的宏平均；事件召回是至少命中一次的真实事件比例；检测延迟只在已检出事件上统计，单位为匿名采样点。
- 5,000 次 bootstrap 按通道成对重采样，不按时间点独立采样。区间未校正多方法筛选，不能视为最终显著性证明。

## 方法与数值修正

最终候选：64 点正常窗口 → PCA 保留 95% 方差 → 白化 → k=50 的局部可达密度比 → 正常校准阈值。没有多模型分数拼接或投票。

标准 LOF 在重复轨迹上出现距离消减误差，D-16 完整／前缀预测曾不一致。`stable_lof_v2.py` 使用 Ball-tree 直接距离、12 位小数规范化及正常参考距离决定的下限。所有候选密度模型统一修复，参数不根据测试标签确定；旧结果保留作为审计，不用于最终表。

原始 Hankel PCA 常数窗口和 Gaussian 零协方差版本也存在数值退化；最终表使用 `hankel_pca_fixed`、`trajectory_gaussian_fixed`。旧同名无 fixed 输出不用于结论。

## 文件定位

- `experiment_v2.py`：统一数据加载、评价指标、传统基线、GRU。
- `iterate_v2.py`：多步预测和开发阶段阈值方案比较；不要对确认集调用 `--compare` 搜索参数。
- `reconstruction_v2.py`：DAE／USAD 改编实验。
- `density_v2.py`、`stable_lof_v2.py`：轨迹表示、密度与数值稳定。
- `detector_v2.py`：无标签拟合和单通道推理。
- `evaluate_frozen_v2.py`：冻结配置评价。
- `analyze_results_v2.py`、`make_report_v2.py`：统计及图表。
- `results_v2/frozen_selection.json`：首次查看确认指标前的选型记录。
- `results_v2/protocol_amendment.json`：短序列划分修订。
- `results_v2/numerical_fix.json`：LOF 数值修复说明。
- `results_v2/stable_dev`、`stable_confirm`：最终密度结果。
- `results_v2/confirmation_frozen.csv`、`confirmation_frozen_channels.csv`：可审查数据表。
- `results_v2/all80_descriptive.csv`：全 80 通道补充统计，包含开发数据。
- `research_v2/`：本次读取的原始仓库文档／源代码快照。权利归各原作者，引用与许可见 `THIRD_PARTY_v2.md`。

## 后续研究边界

可以把这版作为课程项目可运行的第二版。若要宣称方法有稳定、普适的研究优势，下一步应预先锁定方法，再补一个新的独立通道群／数据集；不能继续把当前确认集当作从未见过的最终测试集。候选的误报与跨划分不稳定仍需要解决。
