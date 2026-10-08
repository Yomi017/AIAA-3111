# AIAA 3111 NASA SMAP/MSL telemetry anomaly detection

## 给同学的完整下载入口

GitHub仓库：https://github.com/Yomi017/AIAA-3111 。训练/测试数据、拟合模型、全部历史与本轮实验、复现实验、研究来源、报告和独立海报均已随仓库提供，数据和模型是真实文件，无需另拉Git LFS。

- 完整源码与数据：下载仓库的 **Code → Download ZIP**，或 `git clone https://github.com/Yomi017/AIAA-3111.git`。
- 课程提交包：[Group_XX_Project.zip](https://github.com/Yomi017/AIAA-3111/releases/download/project-v4-complete/Group_XX_Project.zip)。
- 独立报告：[Report.pdf](submission/Report.pdf)；独立海报：[Poster.pdf](submission/Poster.pdf)。
- 旧版大压缩包也保存在[完整交付 Release](https://github.com/Yomi017/AIAA-3111/releases/tag/project-v4-complete)。GitHub单文件100MB限制只影响大ZIP，训练数据、模型与逐点结果全部直接保存在仓库中。

下载后在项目根目录运行下文的数据校验、环境准备和单通道推理命令。`validation_workspaces/`是提交包的重复解压检查副本，Python缓存是本机生成物；实际研究材料均保留。公开版本没有编造学生身份，报告与海报的组号仍待填写。

**当前交付是供组员核验、修改和署名的课程研究草稿，不是可以直接以学生原创名义提交的 AI 成品。** 原课程文件要求“不直接提交 AI-generated work”，且需披露 AIGC 与外部协作者。请完成阅读、理解、修改、身份填写与课程政策核对后提交。

主报告 `Report.pdf` 按课程模板组织：8 页主文，Disclosures、References、Appendix 各另起一页。海报 `Poster.pdf` 在 ZIP 外单独保留。课程未提供最终海报尺寸规范，当前为 A1 横向研究海报草稿；公布规范后应核对。不要把“PDF 已生成”当作课程政策或人工答辩已完成。

## 结论和证据范围

- 原探索性 PCA95 + 白化 + StableLOF50 候选在 58 确认通道的 SMAP/MSL F1=0.4871/0.2659；同协议子序列近邻=0.3782/0.2448。
- 本轮先声明四个机制分支，在 22 开发通道选择 **level_conditional**，冻结后一次性评估 58 确认通道。其 F1=0.4685/0.2679；未带来可靠提升，不改成确认集最优方法。
- MSL 原候选误报率32.9%；P-14、M-7、C-2 贡献62.3%的误报。幅值覆盖和轨迹覆盖两种问题同时存在。
- 旧 v0/v1 已看过官方测试；v2 比较过确认方法，`grid_subspace_v3.py` 还遍历了两种划分。全部结果只能作为探索性证据，不能称为 virgin final test 或 SOTA。该网格脚本已限制为开发集，并保留修复前代码。
- 数据继承原始发布中的测试 min/max 缩放。局部 fit-only 标准化无法撤销它。输入文件25/55维是一个目标值加匿名指令指示，并非25/55个可同步跨通道物理传感器。
- v3 HGB/logistic 为有标签源通道的独立监督任务，采用嵌套 group-held-out；不是正常训练任务的公平替换。

## 环境和依赖

实测 Python3.12、numpy1.26.4、pandas2.3.2、pyarrow16.1.0、scikit-learn1.5.1、torch2.7.0、matplotlib3.9.2、joblib1.4.2、requests2.32.5。权威版本见 `results_v4/provenance.json`；额外生成 PDF 需要 reportlab、渲染核验需要 PyMuPDF。`requirements_v4.txt` 固定实测版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_v4.txt
```

PyTorch仅被共用实验工具和旧神经对照引用；本轮四个机制模型均为 CPU 计算。不同硬件的神经训练不能保证 bitwise 一致。GPU 环境如需重训旧模型，应另安装合适的官方 PyTorch 构建。实际工作环境和来源哈希在 provenance，不放在主报告中。

## 数据获取和准备

本 ZIP 附完整当前公共数据镜像，不采用 sample-data 例外。路径为 `data/telemanom/data/train/*.parquet`、`data/telemanom/data/test/*.parquet` 与 `data/telemanom/labeled_anomalies.csv`。字段：连续 `timestep`、目标 `value`、匿名指令列。训练被基准视为正常；测试标签为闭区间。P-2 两条冲突标注均排除；T-10 没有有效标注，因此不纳入80通道评价。

来源：https://github.com/khundman/telemanom ，实际镜像 https://huggingface.co/datasets/appleparan/telemanom ，固定 revision `2d22e1061be83a88b7b9e48df35163d5147adc9d`。逐文件 SHA-256 在 `results_v2/provenance.json` 与新 provenance。

```powershell
python prepare_data_v4.py           # 校验已有完整数据
python prepare_data_v4.py --download # 缺文件才从固定 revision 下载，校验后写入
python audit_v4.py                  # 原始课程/旧结果审计证据重新计算
```

已有哈希不一致的文件不会被覆盖。网络不可用时可使用包内数据，不能改成 main/latest 后宣称精确复现。

## 完整运行

在解压目录运行。此命令创建新时间戳目录，不覆盖冻结记录或旧结果：

```powershell
.\reproduce_v4.ps1 -PythonExe python -ArtifactPythonExe python
```

顺序为数据校验/获取 → 审计 → 协议测试 → 开发22通道 → 冻结配置/源代码和开发结果哈希 → 一次确认58通道 → 新模型协议/重载检查、逐点CSV导出 → 通道统计、bootstrap、案例图 → 正式报告和海报。新模型每通道同时保存完整四分支配置与阈值；不得按通道挑选最好分支。确认开始后重复执行会被拒绝；若进程中断，请保留目录和日志，声明为失败尝试后开新 replication，而非静默改锁文件。

每步也可手动运行，`--out` 必须使用同一全新目录：

```powershell
python experiment_v4.py development --out results_v4/my_replication
python experiment_v4.py freeze --out results_v4/my_replication
python experiment_v4.py confirmation --out results_v4/my_replication
python analyze_v4.py --run results_v4/my_replication --out results_v4/my_replication/analysis
python build_report_v4.py --run results_v4/my_replication --analysis results_v4/my_replication/analysis --output results_v4/my_replication/deliverables
```

旧比较涉及 GRU、多步 GRU、DAE、USAD adaptation、经典方法，完整训练命令保留在 `reproduce_v2.ps1 -Fresh`（需要较长训练时间，创建独立副本）。v3监督重训：`python experiment_v3.py --run replication_group5`。这些不会参与本轮方法重新选择。报告的旧比较表来自已经保存并审计的 CSV；新复现命令重做本轮机制研究，**不会自动重训旧神经对照**。

## 单通道推理

模型须对应通道，输入须沿用该匿名基准尺度。无需输入测试标签。正常窗口检测在 t 看到 x_t，属于因果检测而非提前预报；不应表述为64步预测。

```powershell
python experiment_v4.py predict --model results_v4/mechanism/confirmation/P-1/model.joblib --input data/telemanom/data/test/P-1.parquet --method level_conditional --output P-1_predictions.csv
```

输出 `timestep,value,score,threshold,anomaly`，第64点起每点评分，前64点明确 warmup。其他冻结分支可用 `subspace_reference`、`normal_envelope`、`robust_trajectory` 做已声明消融，不得按确认标签择优。参考旧模型推理：`python detector_v2.py predict --model results_v2/models/subspace_lof50/P-1.joblib --input data/telemanom/data/test/P-1.parquet --output P-1_reference.csv`。joblib仅加载本包可信生成文件。

## 校验和打包

```powershell
python test_protocol_v2.py
python test_protocol_v4.py
python verify_exports_v2.py
python build_report_v4.py # 从解压包重新生成submission里的报告和独立海报
python validate_submission_v4.py
python package_submission_v4.py
```

协议测试包括闭区间标签、无 point adjustment、窗口对齐、常数/重复轨迹、校准不能改变拟合表示、未来扰动不改变过去分数、80模型保存重载和前缀一致、监督源/目标组无重叠。模型重载和同库版本下的确定性预测可精确匹配；跨线性代数实现只承诺数值容差和同协议，不承诺所有库/硬件 bitwise 一致。bootstrap按数据集分层、按通道成对抽样5000次，固定seed20261006，不代表80条独立航天器实验。

生成物位于 `submission/`（在 ZIP 内报告复制到根路径）；ZIP 内有 manifest 每个 payload 的大小和 SHA-256，ZIP 外另有总包 `.sha256`。ZIP 不含 Poster.pdf，后者单独提交。打包验证包括 `testzip`、manifest、hash、PDF总页数与主文界限。页面PNG在 `results_v4/render_check/`，不打入提交包。默认结果已执行一次全新重跑且开发/确认CSV与全部分支分数一致，验证记录见 `results_v4/reproduction_validation.json`。PDF生成使用Windows系统微软雅黑字体；其他系统需替换为支持中文的同类字体并重新检查排版。

## 目录与溯源

- `results_v4/MECHANISM_PLAN.md`、`audit.json`、`historical_score_audit.csv`、`data_inventory.csv`：假设、审计与完整数据清单。
- `results_v4/mechanism/`：预声明、冻结、确认开始记录、80模型、320逐点NPZ、80预选方法逐点CSV、逐通道CSV。
- `results_v4/regime_diagnostics.csv`、`paired_bootstrap.csv`、`channel_stability.csv`、`case_selection.csv`、`figures/`：可检验分析与全部失败结果。
- `research_v4/METHOD_REVIEW.md`、`sources.json` 及源快照；`research_v2/`及 `THIRD_PARTY_v2.md`：来源、协议差异和原作者许可。
- `results_v2/`、`results_v3/`：历史结果完整保留。v0/v1旧稀疏评估在原目录保留，不作为同协议提升证据。
- `COURSE_CHECKLIST.md`：要求—证据—缺口—修复动作，以及仍需学生完成的工作。

## 手动提交项

填写真实 Canvas Group、所有组员姓名/学号；本人阅读并修订 AI 草稿、核对课程政策与披露；核对后续海报格式；完成分工记录和互评、海报 Q&A；将 ZIP 重命名为真实 `Group_XX_Project.zip` 后与独立 Poster.pdf 按 Canvas 要求上传。提案/组队截止为课程文件所列10月15日23:59，最终项目/海报截止需以 Canvas 后续通知为准。没有进行任何自动提交。
