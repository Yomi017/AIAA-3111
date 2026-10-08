# AIAA 3111 Group Project 选题调研

课程项目要求一条完整的数据挖掘链路：数据获取、预处理、任务定义、实验设计与实现、不同方法比较、结果分析、结论和可视化。报告主文 6–10 页，评分重点是问题与方法是否讲清楚、实验是否公平可比较、代码能否复现结果。

推荐优先级：

1. **人类移动模式鲁棒分类**：最符合课程给出的参考方向，标签和评价清晰，风险最低。
2. **城市空气质量时空异常检测**：研究味道最好，能加入空间图结构和上下文异常，适合做出前沿对比。
3. **工业多变量时序异常检测**：模型和基准很多，适合用 2–4 张卡做系统比较，但评价细节容易出错。
4. **社交媒体主题/社区演化聚类**：可视化强，适合海报，但数据取得和中文文本清洗风险较高。
5. **城市出行模式挖掘**：最符合 pattern mining，但 GPU 用不上，前沿性和结果故事要靠问题设计。
6. **手势分类的正式项目扩展**：已有数据和代码基础，最容易交付，但不能原样重复 Mini Project，必须加入更严谨的跨人物/噪声鲁棒性实验。

## 课程约束转成选题检查表

一个合格题目至少需要回答：

- 数据从哪里来，是否允许公开下载，是否能随代码提供下载说明；
- 每一行/每个序列/每个时间点代表什么，标签或异常定义是什么；
- 数据如何切分，是否会因为同一个人、同一条轨迹或相邻时间窗口造成泄漏；
- 至少一个简单 baseline 和一个更强方法；
- 评价指标为什么适合任务，是否处理类别不平衡或异常比例极低；
- 至少一个条件变化、消融实验或鲁棒性实验；
- 能生成清楚的图：数据分布、训练曲线、混淆矩阵/PR 曲线、异常时间线、聚类投影或模式可视化；
- 结论不能只写“准确率最高”，还要解释什么因素导致差异、何时失败、有什么局限；
- `README` 能在新机器上说明环境、数据准备、运行命令和预期结果。

## 候选一：人类移动模式的鲁棒分类（首选）

### 题目

**Robust Human Mobility Classification under Unseen Users, Sparse Sampling and Noisy GPS**

根据 GPS 轨迹识别 walking、bus、car、bike 等交通方式，并研究模型遇到新用户、降低采样频率、GPS 噪声和缺失点时的泛化能力。

### 数据

使用 Microsoft GeoLife GPS Trajectories。官方资料说明该数据集包含约 182 名用户、17,000 余条轨迹和大量 GPS 点；只有带交通方式标签的子集用于监督分类。正式实现前应下载原始压缩包并统计真正可用的标签数量，不要在提案中先承诺一个未经核对的样本数。

数据处理可以是：按用户和轨迹读取经纬度与时间；计算速度、加速度、方向变化、停留时间、轨迹长度、采样间隔等；按固定窗口生成样本；以用户为单位切分 train/validation/test，确保测试用户从未出现在训练集。

### 方法梯度

1. Majority/most-frequent 和规则阈值，作为最低 baseline。
2. Random Forest、XGBoost/LightGBM 或 Logistic Regression，使用人工轨迹统计特征。
3. TCN、GRU 或 Transformer，对时间序列窗口直接建模。
4. 可选前沿分支：用 TS2Vec 的无监督时序表示学习，再接线性分类器；或者比较 PatchTST/TimesNet 风格的 patch 表示。

### 实验设计

- **Split 1**：随机窗口切分，作为传统但容易乐观的结果；
- **Split 2**：按用户留出，测量真实跨人物泛化；
- **Split 3**：降低采样频率；
- **Split 4**：加入可控 GPS 噪声、随机缺失和不规则采样；
- 比较人工特征模型和序列模型在不同条件下的 macro-F1、balanced accuracy、每类召回率和混淆矩阵。

### 计算量和风险

经典模型在 CPU 上即可完成。TCN/GRU/Transformer 通常 1 张 4090 就足够；4 张卡可以并行不同模型、不同噪声强度和 5 个随机种子。真正的风险不是显存，而是 GeoLife 标签预处理和用户泄漏。必须把用户 ID 放进 split 逻辑，并在 README 中保存生成窗口的规则。

### 适合项目的原因

这是课程 Project Guide 直接给出的分类示例，问题动机、数据来源和鲁棒性问题都容易解释。它比“比较几个分类器”更有研究性，因为可以回答：模型究竟学到了交通模式，还是记住了用户和采样设备的特征？

### 可靠来源

- Microsoft GeoLife 官方介绍：[GeoLife GPS Trajectory Dataset User Guide](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)
- 轨迹表示学习早期代表工作：[Identifying Human Mobility via Trajectory Embeddings, IJCAI 2017](https://doi.org/10.24963/ijcai.2017/234)
- 时序表示学习：[TS2Vec, AAAI 2022](https://doi.org/10.1609/aaai.v36i8.20881)
- Transformer 时序模型：[PatchTST](https://arxiv.org/abs/2211.14730)

## 候选二：城市空气质量的时空异常检测（研究性最强）

### 题目

**Context-Aware Spatio-Temporal Anomaly Detection in Multi-Site Air Quality Data**

检测某个监测站在某个时刻是否出现异常测量，并比较只看本站历史、使用天气/时间上下文、以及同时利用邻近站点信息的效果。

### 数据

使用 UCI Beijing Multi-Site Air Quality Data。它包含多个监测站的小时级空气污染物和气象变量。实现时应明确：站点、时间、PM2.5/PM10/NO2 等变量、缺失比例、训练和测试时间段。不要随机打乱时间点；建议用前段训练、中段验证、后段测试。

### 异常标签问题

这是该题的核心，也是它的研究价值来源。原始数据并不是每个点都有可靠的人工异常标签，可以采用两层评价：

1. **真实事件评价**：使用明显缺测、传感器失效或官方异常标记（若数据中存在），并在报告中说明定义；
2. **受控注入评价**：在测试段注入 point、contextual、collective 三类异常，例如尖峰、持续偏移、局部站点与邻站不一致，再用注入位置作为可复现 ground truth。

注入只应发生在测试集，不能把异常注入训练数据后再用同一异常模式评估。要同时报告 point-level precision/recall/F1 和 event-level detection delay，避免只报告 AUROC。

### 方法梯度

1. Seasonal naive、rolling z-score、Isolation Forest 和 Local Outlier Factor。
2. 自编码器或 LSTM/TCN reconstruction model。
3. MTAD-GAT、OmniAnomaly 或 TranAD 风格的多变量时序异常检测模型。
4. 前沿扩展：把监测站看作图节点，使用邻站信息做 graph-temporal encoder；与“只看单站”的模型进行消融。

### 计算量和风险

数据本身不算超大，1 张 4090 就能跑 baseline 和深度模型。4 张卡适合并行不同窗口长度、异常强度和模型；不建议从头训练大时序 foundation model。最大风险是异常定义不严谨、注入设置被模型“看穿”、时间泄漏和缺失值处理不透明。报告必须把这些写清楚。

### 可靠来源

- 数据集：[UCI Beijing Multi-Site Air Quality Data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- 图神经网络异常检测代表：[Graph Neural Network-Based Anomaly Detection in Multivariate Time Series, AAAI 2021](https://doi.org/10.1609/aaai.v35i5.16523)
- 多变量异常检测代表：[A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data, AAAI 2019](https://doi.org/10.1609/aaai.v33i01.33011409)
- 时序模型综述：[Transformers in Time Series: A Survey, IJCAI 2023](https://doi.org/10.24963/ijcai.2023/759)

## 候选三：工业多变量时序异常检测（模型比较最丰富）

### 题目

**Reliable Multivariate Industrial Anomaly Detection under Sensor Corruption and Operating-Mode Shift**

使用工业控制或服务器监控时序，比较经典统计方法、重构模型、图时序模型和 Transformer 异常检测方法，并研究传感器缺失、噪声和操作模式变化。

### 数据选择

- **SWaT**：水处理系统，带正常运行和攻击/异常阶段，适合做事件级评价；
- **WADI**：更复杂的工业过程，规模和预处理成本更高；
- **SMD、SMAP、MSL**：常用于多变量时序异常检测基准，适合快速复现实验，但要仔细核对标签和时间窗口。

提案最好只选一个主数据集，再用第二个数据集做外部验证；同时上 SWaT、WADI、SMD 容易造成工程量失控。

### 方法和评价

Isolation Forest/One-Class SVM → LSTM-AE/USAD → OmniAnomaly/MTAD-GAT/TranAD。评价至少包含事件级 precision、recall、F1、检测延迟和误报率；说明点级和事件级结果的差异。异常比例极低时，accuracy 没有意义。

### 计算量和风险

2–4 张 4090 可以在不同模型和随机种子之间并行；中等窗口的 Transformer 不会超过单卡显存。风险是不同论文采用了不同的后处理、阈值和标签修正规则，直接抄表格会不公平。必须统一窗口、阈值选择、后处理和评价脚本。

### 可靠来源

- OmniAnomaly 代码/论文入口：[NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
- MTAD-GAT 论文：[Multivariate Time-series Anomaly Detection via Graph Attention Network](https://arxiv.org/abs/2009.02040)
- NASA SMAP/MSL 数据入口可从相关基准仓库获取，但提交前应固定 commit 或下载快照，并记录 SHA-256。

## 候选四：社交媒体主题与社区演化聚类

### 题目

**Topic and Community Evolution in a Chinese Online Event: Representation and Time-Slice Clustering**

选择一个公开事件，按时间段对帖子做主题/用户/互动聚类，比较词袋、TF-IDF、中文 sentence embedding 和图结构表示，研究事件不同阶段的主题演化。

### 方法

- TF-IDF + K-means/Agglomerative clustering；
- 中文句向量 + HDBSCAN/UMAP；
- BERTopic 风格的 embedding + topic representation；
- 可选：用户互动图的 Louvain/Leiden 社区发现；
- 用 silhouette、DBI、topic coherence、人工主题纯度和时间稳定性比较，不要只放一张 t-SNE 图。

### 计算量和风险

中文 embedding 可以用 1–2 张 4090 批量计算；聚类和图算法主要受 CPU/RAM 影响。风险比模型更大：公开数据许可、平台反爬、中文文本清洗和主题数选择都会影响结果。最稳妥的做法是找有明确许可的公开数据集，而不是临时抓取平台数据。

### 适合展示的原因

海报容易做出时间线、主题关键词、社区图和阶段对比；但必须把“聚类结果是分析工具，不是客观真实类别”写进局限性。

## 候选五：城市出行的频繁模式与序列模式挖掘

### 题目

**Recurring Urban Mobility Patterns across Weekdays, Weekends and Rush Hours**

把 Citi Bike 或 NYC Taxi 记录转换成区域—时间交易、OD 交易或事件序列，比较不同时间条件下的频繁流、闭合模式和序列模式。

### 方法

- Apriori 与 FP-Growth：频繁项集；
- PrefixSpan 或 SPADE：序列模式；
- closed/maximal pattern 过滤，控制结果数量；
- 按工作日/周末、早晚高峰、区域做对比；
- 用 support、confidence、lift、覆盖率和模式稳定性评价。

### 计算量和风险

原始 NYC Taxi 数据量很大，但可以固定一个季度和城市区域；这类项目 CPU/RAM 比 GPU 更关键，4 张 4090 没有必要。风险是把连续坐标离散化得过于随意、模式数量爆炸、把共现误解为因果。它非常符合 pattern mining，但前沿感主要来自问题设计和时空条件比较，而不是使用大模型。

### 可靠来源

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [Citi Bike System Data](https://citibikenyc.com/system-data)
- 课程 Project Guide 已将城市移动模式列为 pattern-mining 参考方向。

## 候选六：手势识别正式扩展（交付风险最低，但要避免重复）

现有 Mini Project 已有 9000 张、18 类手势图片和一个可运行分类器。它可以作为正式项目的起点，但**不能直接把 Mini Project 打包重交**。正式题目应改为：

**Cross-Person and Corruption-Robust Hand Gesture Classification from MediaPipe Landmarks**

需要补充：

- 以图片来源或人物分组做更严格的 group split；
- 比较 landmarks + MLP、原图 CNN/轻量视觉模型和传统特征模型；
- 测试低分辨率、遮挡、旋转、噪声、左右手和光照变化；
- 报告宏平均 F1、每类召回率、混淆矩阵和推理延迟；
- 明确 `hand_landmarker.task` 只是预训练的关键点提取器，18 类分类器是另一个模型；
- 在报告中声明 AIGC、外部协作者和 Mini Project 代码的复用范围。

优势是数据和工程基础已经存在，1 张 4090 足以完成；缺点是与课堂 Mini Project 重合度高，可能被认为扩展不足。除非组内没有更好的方向，否则建议把它作为保底方案而不是首选。

## “前沿模型”应该怎样使用

不要把“使用一个 foundation model”本身当作研究问题。更稳妥的结构是：

- 经典 baseline：Random Forest、Isolation Forest、Apriori、TF-IDF + K-means 等；
- 可解释的深度模型：TCN、GRU、Autoencoder、MLP；
- 一个前沿模型或表示：TS2Vec、PatchTST、MTAD-GAT、TranAD、TabPFN 等；
- 同一数据切分、同一指标、同一阈值规则下比较；
- 用消融回答前沿组件到底带来了什么收益。

几个适合查阅的方向：

- TS2Vec：无监督时序表示学习，适合加到移动分类或异常检测前面；[AAAI 2022](https://doi.org/10.1609/aaai.v36i8.20881)
- PatchTST：将长时序切成 patch，适合时序分类/预测 baseline；[arXiv 2211.14730](https://arxiv.org/abs/2211.14730)
- TimesNet：时间变化模式建模，适合做时序模型对比；[arXiv 2210.02186](https://arxiv.org/abs/2210.02186)
- TabPFN：小规模表格分类的 foundation model，适合做小数据基线，但不适合包装成大规模 GPU 项目；[Nature 2025](https://doi.org/10.1038/s41586-024-08328-6)
- Chronos/TimesFM 等时序 foundation model：可以作为 zero-shot 或 frozen representation 对比，但要固定版本、下载模型并严格记录推理成本；不能只报告一个模型的输出。

## 4×4090 的现实使用计划

建议用多卡做并行实验，而不是盲目数据并行：

| GPU | 实验 |
| --- | --- |
| GPU 0 | baseline 与数据预处理验证 |
| GPU 1 | 深度模型主实验 |
| GPU 2 | 前沿模型/表示学习 |
| GPU 3 | 噪声、缺失、跨用户或第二随机种子 |

每次实验保存：配置文件、随机种子、训练日志、模型 checkpoint、指标 JSON、预测结果和 Git commit。这样报告中的表格可以由脚本重新生成。建议先用 1/10 数据跑通全链路，再扩到完整数据；不要一开始就启动 4 卡长训练。

## 当前建议的决策顺序

1. 如果组员希望**最稳妥拿到完整实验**：选 GeoLife 鲁棒移动分类。
2. 如果组员愿意处理异常标签和实验设计：选北京空气质量时空异常检测。
3. 如果组员有较强深度学习工程能力：选 SWaT/WADI 工业异常检测。
4. 如果组员擅长可视化和中文文本：选社交媒体主题/社区演化聚类。
5. 如果组员想严格贴合 pattern mining：选 Citi Bike/NYC Taxi 模式挖掘。
6. 只有在时间非常紧或组员想复用现有代码时，才选手势识别扩展。

## 提案阶段建议写法

提案只需要标题、简短介绍和全体成员信息，但建议介绍中明确四件事：数据集、任务类型、比较方法、核心实验条件。例如 GeoLife 题目可以写成：

> 本项目研究基于 GPS 轨迹的交通方式分类，并重点评估模型在未见用户、稀疏采样、GPS 噪声和缺失条件下的鲁棒性。我们将从 GeoLife 轨迹中构造速度、加速度、方向变化和时间间隔等特征，比较 Random Forest/XGBoost 等传统模型与 TCN/Transformer/TS2Vec 表示模型，使用按用户划分的测试集和 macro-F1、balanced accuracy、混淆矩阵评价泛化能力。项目将分析不同采样和噪声条件下的性能变化，并讨论数据标签、人物偏差和真实部署的局限性。

这段还不是最终提案；确定数据版本、组员和能实现的方法后再提交。不要在没有下载并检查 GeoLife 标签前承诺具体样本数或准确率。

## 关键本地文件

- 项目要求：[03_Project.pdf](03_Project.pdf)
- 评分标准：[02_Project_Assessment_Rubrics.pdf](02_Project_Assessment_Rubrics.pdf)
- 组队指南：[04_Canvas_Team_Formation_Guide.pdf](04_Canvas_Team_Formation_Guide.pdf)
- 报告模板：[05_Report_Template.docx](05_Report_Template.docx)
- API/HPC 指南：[06_AIGC_API_USER_GUIDE.pdf](06_AIGC_API_USER_GUIDE.pdf)
- [API_and_HPC_User_Guide.pdf](API_and_HPC_User_Guide.pdf)
