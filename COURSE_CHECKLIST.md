# 课程要求—当前证据—缺口—修复动作

依据03_Project.pdf（3页）、02_Project_Assessment_Rubrics.pdf（3页）和05_Report_Template.docx；2026-10-06直接读取并与原文件核对。旧 Report_v2.pdf 七页只是实验说明，不能因页数合适就称模板已完成。

| 课程要求 | 审计时的证据与缺口 | 本轮动作与验收证据 | 当前状态 |
|---|---|---|---|
| 至少一种 data mining task | NASA遥测正常训练→异常点检测；选题成立 | 保留 anomaly detection，正式定义输入/输出/目标与告警时点 | 已覆盖 |
| Data acquisition | 公开镜像完整，但下载程序硬编码本机/浮动main | 完整数据入包；固定revision下载、逐文件hash校验、动态项目路径 | 已覆盖 |
| Preprocessing | P-2冲突、旧稀疏评分、继承test缩放风险 | 80通道清单、P-2原行保留；3112旧评分检查；fit-only新增尺度、明确继承限制 | 已覆盖并披露 |
| Experimental design | v2确认暴露比README声称更广 | 保存新机制计划；22开发→哈希冻结→58确认一次；不按确认成绩换模型 | 已覆盖，确认不独立 |
| Comparison of methods | 多种本地方法但论文成绩协议不等价 | 阅读至少三类官方实现/摘要；旧本地同支持比较+四机制分支；监督分开 | 已覆盖 |
| Results and conclusions | MSL误报解释薄弱、候选被确认后突出 | 误报集中度、幅值/形状迁移、通道稳定性、bootstrap、正负案例、新失败结果 | 已覆盖 |
| Visualization | 旧图是实验网页插图 | 数据集分图、机制图、ECDF、MSL误报图、四案例及独立海报；页面PNG检查 | 已覆盖 |
| Report主文6–10页 | 旧7页含交付说明，模板结构欠缺 | Report.pdf按模板8页主文；Disclosures/References/Appendix分开，PDF检查记录 | 已覆盖 |
| Title & Group Members | 无真实身份 | 标题、Group XX和四个姓名/学号明确占位，绝不编造 | 必须人工填写 |
| Introduction/Problem/Methodology/Experiments/Conclusion | 旧说明稿缺正式研究叙述 | 重写正式章节，先结论再证据，贡献仅课程实现/审计 | 已覆盖，需学生审阅 |
| Disclosures及学术诚信 | 原文“不直接提交AI-generated work” | 单独披露AI参与调研、代码、实验、图文；学生核验/改写未假装完成 | 必须人工履行 |
| References | 旧网页有链接无统一文献段 | 真实原论文/实现链接、读取摘录、代码快照、license及hash；不填造论文分数 | 已覆盖 |
| Appendix | 无明确独立补充 | 协议公式/结果定位/监督补充、附CSV/配置/复现信息 | 已覆盖 |
| Group_XX_Project.zip | 旧包名字不符、未有标准根Report | 根Report.pdf/README/代码/完整数据、全部历史结果、manifest+testzip+SHA256 | 已覆盖，组号待填 |
| Poster.pdf单独提交 | 未见独立海报 | 制作可独立阅读的A1横版草稿并渲染；ZIP外单独保留 | 已生成，后续尺寸待核对 |
| Code功能/可读性 | 旧流程能跑但网格可误用确认标签 | 网格guard、冻结检查、正常无标签拟合、Fresh复现、单通道推理、协议测试 | 已覆盖 |
| Poster现场Q&A（30分） | 不能由文件代替真实理解 | README/报告给证据边界；需组员理解何为误报、事件召回、确认暴露 | 必须现场完成 |
| Peer assessment（20分） | 未提供组员或分工 | 不编造互评、合作或学生贡献 | 必须组员完成 |

## 本轮修复记录

1. `grid_subspace_v3.py` 原先遍历development和confirmation；保留原代码在results_v4/historical_code，现默认只开发且写新文件，禁止覆盖旧grid。它历史上确实存在确认暴露，不能通过修代码撤销。
2. 新实验API将拟合/预测与标签评价分离；确认要求冻结源码hash相同且一次性锁，避免继续同名目录挑配置。
3. 原候选重新计算与保存结果逐点一致；14个协议测试及80新模型/320分支预测重载与前缀检查通过。
4. 负结果保留。level_conditional是本轮开发选择，确认平均F1低于旧探索候选；不把robust SMAP的最好分数与另一方法的MSL分数拼接。
5. 主报告不写内部run编号、设备或batch细节，完整配置在README/config/provenance。

## 尚未能自动完成

真实姓名/学号/Canvas组号、学生原创修改与课程AIGC合规核对、未知外部人工协作者声明、后续海报尺寸与提交截止、Canvas上传、答辩与互评。完整技术交付不等于上述人工义务已完成。
