# AIAA 3111 项目进度与交付清单

更新日期：2026年10月8日。

当前已完成完整技术交付：数据、训练后的模型、实验结果、复现流程、报告和海报草稿均已公开上传。组员可以下载、核验和继续修改。正式课程提交仍需完成学生审阅、身份填写、提交要求核对及现场环节。

## 已完成的工作

| 项目 | 当前状态 | 文件或验收证据 |
|---|---|---|
| 选题与任务定义 | 已完成 NASA SMAP/MSL 遥测异常检测的任务定义、计划及研究材料整理 | [项目计划](NASA_SMAP_MSL_Project_Plan_and_Proposal.md)、[研究材料](research_v4/METHOD_REVIEW.md) |
| 完整训练和测试数据 | 已上传82份训练文件、82份测试文件及标签；评价纳入80通道，其中SMAP 53条、MSL 27条；P-2冲突标注和T-10无有效标注的处理已记录 | [数据目录](data/telemanom/)、[数据清单](results_v4/data_inventory.csv)、[准备及校验脚本](prepare_data_v4.py) |
| 模型和历史实验 | 已保留并上传v0至v4的实验；共964个模型文件，包含历史实验及复现副本，不代表964种独立方法 | [历史实验](results_v2/)、[监督实验](results_v3/)、[本轮实验](results_v4/) |
| 本轮四分支实验 | 已完成22条开发通道、配置冻结和58条确认通道评价；保存80个通道模型、320份分支输出及80份预选方法逐点CSV | [冻结及模型结果](results_v4/mechanism/)、[实验脚本](experiment_v4.py) |
| 结果分析 | 已完成方法比较、误报集中度、幅值与轨迹覆盖分析、通道稳定性、成对bootstrap及正负案例图 | [分析结果](results_v4/)、[分析脚本](analyze_v4.py) |
| 协议和推理检查 | 已通过14项协议测试、80个新模型导出重载检查及单通道推理检查；审计3,112份历史评分文件 | [验证记录](results_v4/verification.json)、[协议测试](test_protocol_v4.py) |
| 全流程复现 | 已在全新目录执行数据校验、开发、冻结、确认、分析和报告生成；320份评分输出在当前环境下完全一致，指标CSV字节一致 | [复现记录](results_v4/reproduction_validation.json)、[实际复现目录](results_v4/replication_delivery_check/)、[复现入口](reproduce_v4.ps1) |
| 正式结构报告草稿 | 已生成8页主文，另有披露、参考文献和附录，共11页；逐页检查已完成，仍需学生审阅修改 | [Report.pdf](submission/Report.pdf)、[报告源稿](submission/Report_source.md) |
| 独立海报草稿 | 已生成并检查A1横向海报；最终课程尺寸要求仍需核对 | [Poster.pdf](submission/Poster.pdf) |
| 提交包与下载检查 | 已生成约172MB提交包，含6,070个文件；压缩完整性和全部清单哈希通过；解压后的数据校验、推理和提交包检查通过 | [打包检查](submission/package_validation.json)、[解压运行检查](submission/zip_smoke_validation.json) |
| GitHub公开发布 | 完整源码、实际数据和模型均在公开仓库；三个大型ZIP、报告、海报及校验文件在Releases；已核对完整仓库文件并验证匿名访问和下载 | [公开仓库](https://github.com/Yomi017/AIAA-3111)、[全部附件](https://github.com/Yomi017/AIAA-3111/releases/tag/project-v4-complete)、[下载及校验清单](submission/github_release_manifest.json) |

## 当前实验结论

本轮在开发通道选择的 `level_conditional` 方法，在58条确认通道上的SMAP/MSL F1为 **0.4685 / 0.2679**。旧探索性候选PCA95加白化及StableLOF50为 **0.4871 / 0.2659**。本轮没有得到可靠提升，负结果及各分支输出已经保留。

旧实验曾接触官方测试数据和确认结果，因此这些结果属于探索性证据，不能声称是完全未接触的最终测试或SOTA。全流程复现证明当前计算可重复，不会消除历史测试暴露。数据还继承了原发布版本的测试min/max缩放限制。具体说明见[README](README.md)和报告。

## 仍需完成的清单

- [ ] 填写真实Canvas组号、组员姓名和学号，更新报告、海报及提交包文件名。
- [ ] 每位组员阅读代码和结果，理解方法、误报原因及实验限制；完成学生自己的审阅、修改和贡献记录。
- [ ] 根据课程AIGC及外部协作者要求核对披露。当前披露已写入，但学生核验及修改尚未完成。
- [ ] 核对Canvas后续公布的最终截止时间、海报尺寸和提交要求；当前海报为A1横向草稿。
- [ ] 完成修改后重新生成报告、海报和提交包，并再次检查内容、排版和包内文件。
- [ ] 由组员上传Canvas规定的项目ZIP和独立Poster.pdf。尚未进行Canvas提交。
- [ ] 准备并完成现场海报Q&A、分工记录和组员互评。这些环节不能由现有文件代替。

课程要求逐项对应的证据及待办见[COURSE_CHECKLIST.md](COURSE_CHECKLIST.md)。

## 给同学的下载入口

- [完整仓库](https://github.com/Yomi017/AIAA-3111)：点击 **Code → Download ZIP** 下载当前全部仓库文件，包含这份进度清单及复现实验。
- [课程项目提交包](https://github.com/Yomi017/AIAA-3111/releases/download/project-v4-complete/Group_XX_Project.zip)：约172MB，包含完整训练和测试数据、模型、代码、历史结果及报告；海报单独下载。
- [独立海报](https://github.com/Yomi017/AIAA-3111/releases/download/project-v4-complete/Poster.pdf)和[独立报告](https://github.com/Yomi017/AIAA-3111/releases/download/project-v4-complete/Report.pdf)。
- [历史大压缩包及校验文件](https://github.com/Yomi017/AIAA-3111/releases/tag/project-v4-complete)。

现有提交ZIP保持已验证的原始版本，SHA-256为 `e85fdef9080ed3ebbce89312cde5112a77327d17e4638182d65de5f0c2ec753f`。这次新增的进度清单在仓库和Release中单独提供；最新发布说明以仓库首页为准。
