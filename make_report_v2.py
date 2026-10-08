"""Scientific figures and local Chinese experiment report from saved evidence only."""
from experiment_v2 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html

NAMES={'zmax':'全维度 Z-score','ztarget':'目标值 Z-score','pca':'逐点 PCA','iforest':'Isolation Forest','subsequence_knn':'正常子序列近邻','gru_target':'单变量 GRU','gru_context':'指令条件 GRU','multi_target':'多步 GRU','multi_context':'指令条件多步 GRU','dae':'去噪自编码器','usad':'USAD 改编','hankel_pca_fixed':'时序 PCA 重构','trajectory_gaussian_fixed':'轨迹高斯距离','trajectory_lof20':'稳定轨迹 LOF-20','trajectory_lof50':'稳定轨迹 LOF-50','subspace_lof50':'正常子空间局部密度（候选）'}
BLUE='#2563a6';GOLD='#ce8b24';INK='#23364a'
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white'})

def table(df):
    df=df.copy().rename_axis(None,axis=1)
    if 'method' in df:df['method']=df.method.map(lambda s:NAMES.get(s,s))
    if 'dataset' in df:df['dataset']=df.dataset.replace({'equal_weight_mean':'两数据集等权平均'})
    df=df.rename(columns={'dataset':'数据集','method':'方法','precision':'精确率','recall':'召回率','f1':'逐点 F1','fpr':'正常点误报率','event_recall':'事件召回率','macro_ap':'通道平均 AP','delta_f1':'F1 差值','ci_low':'95% 区间下限','ci_high':'95% 区间上限','seed':'种子','channels':'通道数'})
    return df.to_html(index=False,border=0,escape=False,float_format=lambda v:f'{v:.4f}')

def main():
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);dest=OUT/'deliverables';dest.mkdir(exist_ok=True)
    d=pd.read_csv(OUT/'confirmation_frozen.csv');dev=pd.read_csv(OUT/'development_frozen.csv');ch=pd.read_csv(OUT/'confirmation_frozen_channels.csv')
    pivot=d.pivot(index='method',columns='dataset',values='f1');pivot['平均']=pivot.mean(axis=1)
    order=['zmax','ztarget','pca','iforest','subsequence_knn','gru_target','gru_context','multi_context','usad','dae','hankel_pca_fixed','trajectory_lof50','subspace_lof50']
    fig,axes=plt.subplots(1,2,figsize=(13,7.8),sharey=True)
    y=np.arange(len(order))
    for ax,ds in zip(axes,['SMAP','MSL']):
        vals=pivot.loc[order,ds];ax.barh(y,vals,color=[GOLD if m=='subspace_lof50' else BLUE for m in order],height=.64)
        ax.set_yticks(y,[NAMES[m] for m in order]);ax.invert_yaxis();ax.set_xlim(0,.6);ax.set_xlabel('逐点 F1（不做 point adjustment）');ax.set_title(f'{ds} · '+('39' if ds=='SMAP' else '19')+' 个确认通道',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
        for yy,val in zip(y,vals):ax.text(val+.008,yy,f'{val:.3f}',va='center',fontsize=9)
    fig.suptitle('固定参数的方法比较：候选提升平均 F1，但不在每个数据集上领先所有方法',x=.02,ha='left',fontsize=14,fontweight='bold')
    fig.text(.02,.015,'配置由开发通道确定。金色候选在确认结果比较后被重点报告；这是探索性证据，不是新的独立最终测试。',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(figdir/'comparison.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True);methods=['subsequence_knn','trajectory_lof50','subspace_lof50']
    for ax,ds in zip(axes,['SMAP','MSL']):
        for offset,frame,label,color in [(-.19,dev,'开发集',BLUE),(.19,d,'确认集',GOLD)]:
            vals=frame[frame.dataset==ds].set_index('method').loc[methods].f1
            ax.bar(np.arange(3)+offset,vals,.36,label=label,color=color)
            for i,v in enumerate(vals):ax.text(i+offset,v+.012,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_xticks(range(3),['子序列近邻','原始轨迹密度\n开发集预选','子空间密度\n探索性候选']);ax.set_ylim(0,.6);ax.set_title(ds,loc='left');ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    axes[0].set_ylabel('逐点 F1');axes[1].legend(frameon=False);fig.suptitle('开发／确认表现并不一致：不能只展示候选的有利划分',x=.03,ha='left',fontsize=14,fontweight='bold');fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(figdir/'split_comparison.png',dpi=180);plt.close(fig)

    a=ch[ch.method=='subspace_lof50'].copy();a['f1']=2*a.tp/(2*a.tp+a.fp+a.fn).clip(lower=1)
    b=ch[ch.method=='subsequence_knn'].set_index('channel');a['baseline_f1']=a.channel.map(2*b.tp/(2*b.tp+b.fp+b.fn).clip(lower=1));a['delta']=a.f1-a.baseline_f1
    cases=[]
    for ds in ['SMAP','MSL']:
        aa=a[a.dataset==ds];cases.extend([(aa.sort_values('delta',ascending=False).iloc[0],'改善案例'),(aa.sort_values('delta').iloc[0],'失败案例')])
    fig,axes=plt.subplots(4,2,figsize=(15,12),gridspec_kw={'width_ratios':[1,1]})
    source_rows=[]
    for row,(case,tag) in enumerate(cases):
        c=case.channel;z=np.load(OUT/'stable_confirm'/c/'subspace_lof50.npz');value=pd.read_parquet(DATA/'data/test'/f'{c}.parquet').value.to_numpy()[L:];x=np.arange(L,L+len(value));y=z['label'];p=z['pred'];t=float(z['threshold']);score=z['score']/t
        axes[row,0].plot(x,value,color=BLUE,lw=.8,label='目标遥测值')
        axes[row,1].plot(x,score,color=BLUE,lw=.8,label='分数／阈值');axes[row,1].axhline(1,color=INK,ls='--',lw=1,label='告警阈值');axes[row,1].set_yscale('symlog',linthresh=1)
        for ax in axes[row]:
            for start,end in segments(y):ax.axvspan(start+L,end+L,color=GOLD,alpha=.2)
            ax.set_xlim(L,x[-1]);ax.grid(alpha=.12);ax.set_xlabel('匿名采样点（非真实日期）')
        axes[row,0].set_title(f'{case.dataset} · {c} · {tag}   候选 F1 {case.f1:.3f} / 基线 {case.baseline_f1:.3f}',loc='left',fontsize=11)
        axes[row,1].set_title('异常分数；橙色背景＝真实异常；底部短线＝实际告警',loc='left',fontsize=10)
        axes[row,1].plot(x[p],np.zeros(p.sum()),'|',color=INK,markersize=3,alpha=.5)
        source_rows.append(dict(channel=c,dataset=case.dataset,selection=tag,candidate_f1=case.f1,baseline_f1=case.baseline_f1,delta=case.delta))
    axes[0,0].legend(frameon=False,loc='upper right');axes[0,1].legend(frameon=False,loc='upper right');fig.suptitle('按每个数据集 F1 增量最大／最小自动选例，同时展示收益与失败',x=.02,ha='left',fontsize=14,fontweight='bold');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(figdir/'cases.png',dpi=160);plt.close(fig)
    pd.DataFrame(source_rows).to_csv(OUT/'case_selection.csv',index=False)

    wide=pivot.sort_values('平均',ascending=False).reset_index();wide['method']=wide.method.map(NAMES);wide=wide.rename(columns={'method':'方法','平均':'两数据集等权平均 F1'})
    metrics=d[d.method.isin(['subspace_lof50','subsequence_knn','trajectory_lof50','hankel_pca_fixed'])][['dataset','method','precision','recall','f1','fpr','event_recall','macro_ap']].copy();metrics['method']=metrics.method.map(NAMES)
    ci=pd.read_csv(OUT/'paired_bootstrap.csv');ci=ci[ci.comparator=='subsequence_knn'][['dataset','delta_f1','ci_low','ci_high']]
    ab=pd.read_csv(OUT/'ablation.csv');ab=ab[(ab.split=='confirmation')&(ab.span==1)].pivot(index='method',columns='dataset',values='f1').reset_index();ab['method']=ab.method.map(NAMES)
    seeds=pd.read_csv(OUT/'seed_robustness.csv')
    full=pd.read_csv(OUT/'all80_descriptive.csv').pivot(index='method',columns='dataset',values='f1');full['平均 F1']=full.mean(1);full=full.sort_values('平均 F1',ascending=False).reset_index();full['method']=full.method.map(NAMES)
    sources='''<ol>
    <li><a href="https://github.com/khundman/telemanom">Telemanom 原仓库</a>；<a href="https://arxiv.org/abs/1802.04431">KDD 2018 论文</a>。已阅读 README、配置与阈值源代码。输入结构、预处理限制、多步预测和动态阈值的依据。</li>
    <li><a href="https://github.com/manigalati/usad">USAD 作者实现</a>；<a href="https://doi.org/10.1145/3394486.3403392">KDD 2020 论文</a>。已阅读编码器、双解码器、两阶段对抗损失。本项目使用线性输出适配标准化输入，属于改编实现。</li>
    <li><a href="https://github.com/imperial-qore/TranAD">TranAD，VLDB 2022</a>。已阅读仓库说明及模型代码；本轮没有训练 TranAD，不填造比较结果。</li>
    <li><a href="https://github.com/TheDatumOrg/TSB-AD">TSB-AD，NeurIPS 2024 基准</a>。已阅读方法目录，覆盖 Sub-LOF、FITS、M2N2 等；不把“新模型”直接视为更好。</li>
    <li><a href="https://scikit-learn.org/1.5/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor">scikit-learn LOF novelty 文档</a>。训练参考与未见数据分离；修正版保留局部可达密度定义，增加确定性距离计算和数值下限。</li>
    <li><a href="https://huggingface.co/datasets/appleparan/telemanom">数据镜像</a>。实际使用的文件哈希及镜像 revision 记录在 provenance.json。</li></ol>'''
    body=f'''<h1>NASA SMAP/MSL：方法优化与完整实验</h1>
    <p class="subtitle">AIAA 3111 · 异常检测 · v2 · 2026-10-06 · 80 个有效通道</p>
    <div class="lead"><b>本轮找到数值上超过已有基线的候选：正常子空间的稳定局部密度检测。</b><br>在 58 个确认通道上，SMAP／MSL 逐点 F1 为 <b>0.487／0.266</b>；较强子序列近邻基线为 <b>0.378／0.245</b>，两数据集等权平均从 <b>0.312 → 0.376</b>（+0.065，约 +20.9%）。这不是所有指标、所有划分全面领先。</div>
    <p><b>证据等级：</b>开发集预选的是原始轨迹密度，确认集平均 F1 只有 0.301，低于子序列近邻的 0.312。子空间候选虽然参数预先固定，但本报告是在比较确认结果后突出它；因此属于<b>探索性候选</b>，不能宣称已通过全新独立测试或达到 SOTA。此前 v0/v1 也看过这些官方测试数据。</p>
    <h2>1. 与课程要求的对应</h2><p>任务属于 <b>anomaly detection</b>：从真实航天器正常运行序列中挖掘反复出现的时序模式，检测测试运行中偏离正常模式的时间点。它不是天气预测，也不要求人工给数据补标签。官方提供异常区间标签，本轮只用它们进行开发阶段选型及结果评估；模型拟合与阈值校准使用原有正常训练数据。</p>
    <h2>2. 查阅了哪些方法，为什么没有简单拼接</h2>
    <table><tr><th>路线</th><th>机制与本轮处理</th><th>结果／取舍</th></tr>
    <tr><td>Telemanom</td><td>过去目标值和指令 → 未来目标值 → 预测残差；实现单步／16 步 GRU、简化无标签动态阈值。</td><td>补齐每个时间点评分后明显好于旧实现；多步及动态阈值未稳定胜出，保留负结果。</td></tr>
    <tr><td>USAD</td><td>共享编码器、两个解码器及对抗训练；同一个模型的训练目标。</td><td>改编实现平均 F1 0.317；不是原论文指标复现。</td></tr>
    <tr><td>去噪自编码器</td><td>给正常窗口加噪声／遮挡，再恢复原始轨迹。</td><td>平均 F1 0.335，强于部分预测方法，但低于本轮候选。</td></tr>
    <tr><td>时序 PCA、邻域与密度</td><td>学习正常轨迹的低维结构及邻域分布，比较形状和相对稀疏程度。</td><td>候选来自这一条路线；优势不依赖复杂网络或多模型加权。</td></tr>
    <tr><td>TranAD／FITS／M2N2</td><td>调研自条件 Transformer、频域插值、测试期正常样本适配。</td><td>本轮只调研，未训练；不拿论文表里的分数与当前协议直接比较。</td></tr></table>
    <h2>3. 最终候选具体在做什么</h2>
    <p>每个通道独立处理，最终候选使用目标遥测值，不使用指令字段。神经对照已比较是否使用指令。每个文件的其他维度主要是编码后的控制指令，不能描述成 25／55 个独立物理传感器。</p>
    <div class="formula">正常窗口 wₜ = [xₜ₋₆₃, …, xₜ]<br>正常训练窗口 → PCA（累计方差 95%）→ 白化坐标 zₜ = Λ⁻¹ᐟ² Uᵀ(wₜ − μ)<br>reachₖ(z,o) = max( ‖z−o‖₂, dₖ(o), ε )<br>lrdₖ(z) = 1 / meanₒ reachₖ(z,o)<br>score(z) = meanₒ [ lrdₖ(o) / lrdₖ(z) ]<br>告警 ⇔ score(zₜ) &gt; 正常校准分数的 99.5% 分位数</div>
    <p>保留 95% 正常方差压缩冗余轨迹，白化减轻某些高方差方向对距离的支配。局部密度比衡量“它是否比附近正常轨迹更稀疏”，而非只看离整体平均值多远。这里是<b>表示学习后进行一个密度判别</b>，没有混合 GRU、PCA、IForest 的异常分数，也没有投票。</p>
    <p>实现细节：64 点窗口；正常参考窗口步长 2，测试逐点评分；k=50（样本不足时 k=n−1）；ε=max(10⁻⁶, 正常正值第 k 邻居距离中位数×0.001)；特征四舍五入到 12 位小数；特征值下限 10⁻⁶；距离采用 Ball-tree 直接计算。候选不做额外分数平滑。数值下限是重复轨迹导致不稳定后的修复，并非根据测试 F1 搜索出来。</p>
    <h2>4. 统一实验协议及审计修正</h2>
    <ul><li>P-2 在标注表里有两份冲突区间，排除，不重复计数；T-10 有训练文件但不在有效评估标注清单。最终 53 个 SMAP、27 个 MSL 通道。</li>
    <li>按通道 ID 的 SHA-256 模 3 划分：开发 14 SMAP＋8 MSL，确认 39 SMAP＋19 MSL。没有把不同通道拼成长序列。</li>
    <li>正常序列原则上前 60% 拟合、中间 20% 选 checkpoint、末尾 20% 校准；短通道为后两段各留至少 96 点，其余用于拟合。D-12 仅 312 点，实际 120／96／96。这个修正来自长度检查。</li>
    <li>均值、标准差只在拟合段计算。训练、选择、校准窗口均不跨分段边界；所有方法统一丢弃测试最前面的 64 点作为 warmup，剩下每点都有分数。</li>
    <li>神经模型按正常选择段预测／重构误差选择 checkpoint。开发标签选择方法与平滑／阈值方案。各方法冻结的配置均为正常校准 99.5% 分位阈值；没有在确认标签上搜索最优阈值。</li>
    <li>测试标签区间按闭区间展开；逐点 Precision／Recall／F1，不做 point adjustment。事件召回另列，命中一个点不会把整个真实异常区间自动补成正确。</li>
    <li>原始公开 Telemanom 数据已用测试范围的 min/max 缩放，属于继承的基准预处理限制。不能声称数据来自完全无测试信息的原始工程流程。</li>
    <li>旧 LSTM 每隔 4／8 点预测、其余填零，旧基线还有重复标签和预处理不一致。本轮数据不能直接与旧表当成同协议提升；旧表保留但标记过时。</li></ul>
    <h2>5. 完整结果</h2><img src="../figures/comparison.png" alt="确认集各方法 F1 对比">{table(wide)}
    <p>“平均”是两个数据集的 F1 等权平均，不是把两者时间点混在一起后重算 F1。所有表里的基线都是本地重新运行的同协议结果，不是引用论文最优值。</p>
    <h3>误报、召回与排序能力</h3>{table(metrics)}
    <p><b>仍然明显的问题：</b>候选 MSL 正常点误报率约 32.9%，精确率仅 16.9%，还不适合直接上线；SMAP 事件召回约 66.0%，低于近邻基线的 84.9%。候选 SMAP macro-AP 也低于目标值 Z-score，因此不能说异常排序能力全面更强。时序 PCA 在 MSL 上 F1=0.304，比候选高。</p>
    <h3>划分敏感性与不确定性</h3><img src="../figures/split_comparison.png" alt="开发与确认分数对比">
    <p>候选在开发集 MSL 的 F1 只有 0.103，说明不同通道群之间差异很大。不能用一次划分的收益掩盖这个问题。下表以通道为单位做 5,000 次配对 bootstrap；比较对象为子序列近邻。它不是逐时间点独立抽样。</p>{table(ci)}
    <p>区间没有校正多候选筛选；不能将其解释成最终显著性结论。MSL 单独的增益区间跨 0；与去噪自编码器的平均增益区间也跨 0。完整对比见 paired_bootstrap.csv。</p>
    <h3>相同平滑配置下的消融</h3>{table(ab)}
    <p>此表两者均不平滑，避免将平滑差异误当成子空间表示的贡献。子空间密度相对原始轨迹密度提升约 0.042 SMAP、0.049 MSL。消融为描述性分析，不用于重新调参。</p>
    <h3>全 80 通道补充结果（包含开发集，仅描述性）</h3>{table(full)}<p>这里包含参与选型的开发通道，不能称作独立测试结果。候选等权平均 F1=0.343；本轮参考基线中 GRU-target 为 0.302、子序列近邻为 0.298。</p>
    <h3>神经基线重复种子</h3>{table(seeds)}<p>重复种子 7／42／2026，保持各自方法的冻结配置。确定性的密度方法无需用随机种子制造“重复实验”。</p>
    <h2>6. 成功与失败案例</h2><img src="../figures/cases.png" alt="每个数据集改善最多和恶化最多的案例">
    <p>每个数据集自动选 F1 增量最大及最小通道，不只挑成功图。阴影只用于展示真实标签，不参与分数计算和告警补齐。案例选择清单为 case_selection.csv。</p>
    <h2>7. 可直接复用的交付物</h2>
    <ul><li><a href="../../README_v2.md">README_v2.md</a>：重跑、单通道推理及文件说明。</li><li>results_v2/models/subspace_lof50/：80 个已拟合模型，直接读取对应通道数据即可评分。</li><li>results_v2/confirmation_frozen.csv、confirmation_frozen_channels.csv：汇总和逐通道结果。</li><li>results_v2/stable_confirm/ 与 stable_dev/：逐点评分、校准分数、标签、预测和阈值。</li><li>results_v2/frozen_selection.json、protocol_amendment.json、numerical_fix.json、provenance.json：选型、修订及来源记录。</li><li>results_v2/export_verification.csv：80 个通道的模型重载预测一致、前缀因果一致检查全部通过；8 个协议单元测试通过。</li></ul>
    <p><b>下一步研究建议：</b>当前值得保留的题目是“面向航天器遥测的正常轨迹表示与稳定局部密度异常检测”。课程报告可以比较预测、重构和邻域三条路线，重点解释为什么轨迹表示有效、为什么跨通道选择仍不稳定。若要提出可靠的泛化优势，需要预先锁定方法，再在新的通道群或新数据集验证，而不是继续对同一确认集挑参数。没有自动修改或提交 Canvas 提案。</p>
    <h2>8. 资料与复现声明</h2>{sources}<p>AIGC 使用：研究辅助、代码实现、检查、图表和文字草稿由 AI 辅助完成。实验数字来自本地实测文件；用户仍应核对课程的 AIGC 披露要求。没有伪造文献或人工制造异常标签。</p>'''
    css='''body{font-family:"Microsoft YaHei",system-ui,sans-serif;color:#23364a;background:#f4f7fb;margin:0;line-height:1.75}main{max-width:1180px;margin:30px auto;background:white;padding:44px 52px}h1{font-size:29px;line-height:1.4}h2{margin-top:40px;border-bottom:2px solid #e5edf6;padding-bottom:8px}h3{margin-top:25px}.subtitle{color:#62788e}.lead{background:#edf4fb;border-left:5px solid #2563a6;padding:20px}.formula{font-family:Consolas,"Microsoft YaHei",monospace;background:#f5f7fa;padding:20px;line-height:2}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}th{background:#edf3f9;text-align:left}th,td{padding:9px 10px;border-bottom:1px solid #e4eaf0}img{width:100%;height:auto;margin:16px 0}a{color:#2563a6}li{margin:7px 0}@media(max-width:750px){main{padding:20px;margin:0}table{display:block;overflow:auto}h1{font-size:24px}}@media print{body{background:white}main{padding:0;margin:0;max-width:none}img,table{break-inside:avoid}h2{break-after:avoid}}'''
    path=dest/'实验结果与方法说明.html';path.write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NASA SMAP/MSL v2 实验报告</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
    print(path)

if __name__=='__main__':main()
