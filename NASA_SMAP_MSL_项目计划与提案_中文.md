# AIAA3111 项目计划与基础提案

## 一、与课程项目要求的匹配

### 1. 任务类型

本项目属于老师要求的四类数据挖掘任务中的：

\[
\boxed{\text{Anomaly Detection（异常检测）}}
\]

模型主要使用正常遥测数据学习系统的正常模式，再检测测试数据中的异常；数据集提供的 anomaly labels 用于客观评估检测结果。

### 2. 真实世界问题

NASA 航天器会持续产生多变量遥测数据。设备故障、传感器异常或系统状态变化可能先表现为时间序列中的异常模式。及时发现这些异常，有助于航天器运行监控和故障预警。

因此，本项目解决的是一个真实的航天器状态监测问题，而不是人为构造的抽象数据挖掘例子。

### 3. 数据挖掘方法

项目将比较统计方法、传统机器学习方法和时序深度学习方法：

- 滑动窗口 z-score；
- PCA 重构误差；
- Isolation Forest；
- LSTM 或 GRU Autoencoder；
- 可行时再加入 Transformer 作为扩展。

这些方法都用于从大量多变量数据中学习正常模式并发现偏离模式，符合 data mining 的方法要求。

### 4. 数据和标签

NASA SMAP/MSL 数据集包含：

- 多个遥测变量；
- 正常训练数据；
- 测试数据；
- 已标注的异常时间区间。

标签可以转换为逐时间点标签：

\[
y_t=\begin{cases}
1,&t\text{属于官方标注的异常区间}\\
0,&\text{否则}
\end{cases}
\]

因此不需要把人工注入异常作为主要 ground truth，可以直接使用数据集提供的标签评价模型。

### 5. 与项目提交要求的对应关系

原 PDF 要求项目针对真实问题，使用 classification、clustering、anomaly detection 或 pattern mining 中至少一种任务，并提交报告、代码、README、数据或下载说明、poster 和现场展示。本项目对应关系如下：

| 原项目要求 | 本项目对应内容 |
|---|---|
| 真实世界问题 | NASA 航天器遥测异常监测 |
| 数据挖掘任务 | Anomaly Detection |
| 数据集 | NASA SMAP/MSL |
| 方法 | z-score、PCA、Isolation Forest、LSTM/GRU Autoencoder |
| 量化实验 | Precision、Recall、F1、FPR、Detection Delay |
| 报告 | 问题、数据、方法、实验、结果、局限性 |
| 代码和 README | 数据下载、预处理、训练、测试和复现实验说明 |
| Poster | 问题、数据、方法、结果和结论 |

因此，换成 NASA SMAP/MSL 后仍然完全符合原 PDF 的 project 要求。

## 二、项目题目

**基于 NASA 航天器遥测数据的上下文感知异常检测**

英文题目可以写为：

**Context-Aware Anomaly Detection in NASA Spacecraft Telemetry Data**

## 三、项目要回答的问题

1. 只使用单个遥测变量时，能否检测官方标注的异常？
2. 同时使用多个变量，是否能提高异常检测效果？
3. 时间窗口信息是否比只看当前时刻更有效？
4. 哪种方法在检测质量和误报率之间表现最好？
5. 哪种方法能更早发现一个异常事件？

核心比较为：

\[
\text{单变量}
\quad vs \quad
\text{多变量}
\quad vs \quad
\text{多变量 + 时间窗口}
\]

## 四、六个实施步骤

### 第一步：确认数据集和标签

先下载 SMAP 和 MSL，检查：

- 文件是否能够正常获取；
- 训练集和测试集的格式；
- 通道数量和序列长度；
- anomaly labels 的格式；
- 标签能否与测试数据正确对齐。

这一步的通过标准是：可以画出至少一条正常序列和一条带异常区间的测试序列，并确认标签位置正确。

### 第二步：完成预处理

主要操作：

1. 检查缺失值和无效值；
2. 按通道归一化；
3. 只使用训练集计算归一化参数；
4. 将连续序列切分成固定长度窗口；
5. 将异常区间转换为 point-wise labels 和 window-level labels。

窗口标签可以定义为：

\[
y_i^{window}=1
\quad\text{if}\quad
\sum_{t\in W_i}y_t>0
\]

这一步的通过标准是：能够输出统一格式的 $X$、$y$，并且训练、验证、测试数据没有时间泄漏。

### 第三步：建立 Baseline

先不训练复杂神经网络，完成三个基础方法：

#### 滑动窗口 z-score

\[
z_t=\frac{x_t-\mu_t}{\sigma_t}
\]

当 $|z_t|>\tau$ 时判断为异常。

#### PCA 重构误差

\[
e_t=\|x_t-\hat{x}_t\|_2
\]

#### Isolation Forest

使用多变量窗口特征检测孤立样本。

这一步的目标不是追求最高分，而是确认：

- 标签和评价代码正确；
- 模型确实能产生异常分数；
- 指标计算没有问题。

### 第四步：训练主要模型

主要模型选择 LSTM 或 GRU Autoencoder。

模型使用正常训练窗口学习重构：

\[
X_{t-L:t}\rightarrow \hat{X}_{t-L:t}
\]

异常分数为：

\[
e_t=\|X_t-\hat{X}_t\|_2
\]

根据验证集确定阈值 $\tau$：

\[
\hat y_t=\mathbb{1}(e_t>\tau)
\]

如果 LSTM/GRU 运行稳定，再考虑 Transformer 作为扩展，不把 Transformer 作为项目能否完成的前置条件。

### 第五步：统一评估和对比

使用官方 anomaly labels 作为 ground truth，计算：

\[
\text{Precision}=\frac{TP}{TP+FP}
\]

\[
\text{Recall}=\frac{TP}{TP+FN}
\]

\[
F1=\frac{2PR}{P+R}
\]

\[
\text{FPR}=\frac{FP}{FP+TN}
\]

对于异常区间 $[t_s,t_e]$，检测延迟为：

\[
\text{Delay}=\hat t_{first}-t_s
\]

同时报告两类结果：

- point-wise 指标；
- event-level 指标。

这样可以避免模型只检测到异常区间中的一个点，却被误认为完整检测了整个事件。

### 第六步：形成项目成果

最后整理：

- 实验结果表；
- 方法对比图；
- 异常分数和真实标签的可视化；
- 错误案例分析；
- 项目报告；
- poster；
- 代码和 README；
- 数据下载和运行说明。

## 五、开始实现前必须确认的可行性

### 1. 数据可获取

必须先确认 SMAP/MSL 能够下载，并且文件没有权限或链接问题。

### 2. 标签可对齐

必须确认 anomaly intervals 能正确映射到测试序列的时间索引。否则后面的 Precision、Recall 和 F1 都不可信。

### 3. 数据规模可运行

先用一个较小的通道或较短窗口跑通代码，再扩大到完整数据。4 张 4090 对 LSTM/GRU Autoencoder 足够，但不需要一开始就使用四卡训练。

### 4. Baseline 能产生有效结果

如果 z-score、PCA 或 Isolation Forest 完全无法区分异常，应该先检查数据和标签，而不是直接增加模型复杂度。

### 5. 评价方式合理

异常数据通常类别极不平衡，所以不能只报告 accuracy，至少要报告：

\[
\text{Precision},\quad \text{Recall},\quad F1,\quad \text{FPR},\quad \text{Detection Delay}
\]

### 6. 防止数据泄漏

归一化参数、阈值和超参数只能使用训练集或验证集确定，不能使用测试集标签调参。

## 六、基础提案草稿

### 项目标题

基于 NASA 航天器遥测数据的上下文感知异常检测

### 项目简介

本项目研究 NASA SMAP/MSL 航天器多变量遥测数据中的异常检测问题。数据集包含多个遥测通道、正常训练数据、测试数据以及官方标注的异常时间区间。我们将把异常检测定义为：使用正常遥测数据学习系统的正常模式，并识别测试数据中偏离正常模式的时间点或时间区间。

项目首先实现滑动窗口 z-score、PCA 重构误差和 Isolation Forest 等传统方法，然后使用 LSTM 或 GRU Autoencoder 建立时序模型。我们将比较单变量输入、多变量输入和带时间窗口的多变量输入，研究上下文信息是否能够提高异常检测效果。模型结果将与官方 anomaly labels 对比，并使用 Precision、Recall、F1-score、False Positive Rate、事件检测率和 Detection Delay 进行评价。最终提交将包括实验报告、代码、README、数据下载说明和 poster。

### 项目类型

\[
\boxed{\text{Anomaly Detection}}
\]

### 当前阶段目标

当前不立即追求复杂模型，而是先验证：

\[
\text{数据可下载}
\rightarrow
\text{标签可对齐}
\rightarrow
\text{Baseline 可运行}
\rightarrow
\text{指标可计算}
\]

这四步全部通过后，再开始训练 LSTM/GRU Autoencoder 和制作正式实验结果。
