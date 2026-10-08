"""Formal eight-page main report and separate poster, from measured CSV evidence.

This editable authoring source and emitted Markdown remain review drafts.
Uses stdlib + ReportLab; no ML dependency is needed to regenerate the documents.
"""
from pathlib import Path
import argparse,csv,json,html,re
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parent
BLUE=colors.HexColor('#245b7d');INK=colors.HexColor('#182e40');LIGHT=colors.HexColor('#edf3f7');GRAY=colors.HexColor('#506273')
FONT='Chinese';BOLD='ChineseBold'
def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def reg():
    pdfmetrics.registerFont(TTFont(FONT,'C:/Windows/Fonts/msyh.ttc'))
    pdfmetrics.registerFont(TTFont(BOLD,'C:/Windows/Fonts/msyhbd.ttc'))
    pdfmetrics.registerFontFamily(FONT,normal=FONT,bold=BOLD,italic=FONT,boldItalic=BOLD)
def fmt(x):return f'{float(x):.3f}'
NAMES={'subspace_lof50':'PCA 子空间局部密度（探索）','trajectory_lof50':'原始轨迹 LOF（此前预选）','subsequence_knn':'正常子序列近邻','dae':'去噪自编码器','usad':'USAD 改编','hankel_pca_fixed':'时序 PCA 重构','trajectory_gaussian_fixed':'轨迹高斯距离','gru_target':'单步 GRU','gru_context':'指令条件单步 GRU','multi_target':'多步 GRU','multi_context':'指令条件多步 GRU','ztarget':'目标值 Z-score','zmax':'目标与指令 Z-score','pca':'逐点 PCA','iforest':'Isolation Forest','trajectory_lof20':'原始轨迹 LOF20','subspace_reference':'PCA 密度参考','normal_envelope':'正常覆盖阈值','level_conditional':'水平分层阈值（本轮预选）','robust_trajectory':'稳健轨迹特征'}

def build(analysis,run,dest):
    reg();dest.mkdir(parents=True,exist_ok=True)
    s=getSampleStyleSheet()
    for name,size,leading,extra in [('body',10.1,16.1,dict(spaceAfter=9)),('heading',14,21,dict(fontName=BOLD,spaceBefore=8,spaceAfter=10,textColor=BLUE)),('subheading',11,17,dict(fontName=BOLD,spaceBefore=5,spaceAfter=6)),('caption',8.4,12.3,dict(textColor=GRAY,spaceAfter=10)),('cell',8.3,12.1,dict()),('title_cn',19,28,dict(fontName=BOLD,textColor=INK,spaceAfter=12)),('small',9,14,dict(spaceAfter=7))]:
        opts=dict(fontName=FONT,fontSize=size,leading=leading,wordWrap='CJK');opts.update(extra);s.add(ParagraphStyle(name,**opts))
    W,H=A4;width=W-144;story=[];source=[]
    def p(text,style='body'):
        # Microsoft YaHei has no Unicode subscript t/k. Use real PDF subscripts.
        submap=str.maketrans('₀₁₂₃₄₅₆₇₈₉ₜₖₒ₋','0123456789tko-')
        text=re.sub('[₀₁₂₃₄₅₆₇₈₉ₜₖₒ₋]+',lambda m:'<sub>'+m.group().translate(submap)+'</sub>',text)
        visible=html.unescape(re.sub('<[^>]+>','',text))
        missing={ch for ch in visible if not ch.isspace() and ord(ch) not in pdfmetrics.getFont(FONT).face.charToGlyph}
        if missing:raise ValueError('Unsupported PDF glyphs: '+str(missing))
        source.append(text);story.append(Paragraph(text,s[style]))
    def heading(text):p(text,'heading')
    def sub(text):p(text,'subheading')
    def page():story.append(PageBreak());source.append('\n--- PAGE BREAK ---\n')
    def table(rows,widths=None):
        source.append('\n'+'\n'.join('| '+' | '.join(str(c) for c in row)+' |' for row in rows)+'\n')
        cells=[[Paragraph(html.escape(str(c)),s['cell']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths or [width/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,BLUE),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d6e0e7')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,9))
    def pic(name,caption,w=width):
        f=analysis/'figures'/name;im=ImageReader(str(f));iw,ih=im.getSize();story.append(Image(str(f),width=w,height=w*ih/iw,hAlign='CENTER'));p(caption,'caption');source.append('Figure source: '+str(f))
    old=read(ROOT/'results_v2/confirmation_frozen.csv');dev=read(run/'development_summary.csv');conf=read(run/'confirmation_summary.csv');boot=read(analysis/'paired_bootstrap.csv');evidence=json.loads((analysis/'analysis_evidence.json').read_text());inv=read(analysis/'data_inventory.csv')
    def find(rows,ds,name):return next(r for r in rows if r['dataset']==ds and r['method']==name)
    selected=json.loads((run/'frozen_selection.json').read_text())['selected']
    newmean=sum(float(find(conf,ds,selected)['f1']) for ds in ['SMAP','MSL'])/2
    # Page 1: title, people, Introduction and formal task.
    p('航天器遥测中的正常轨迹覆盖与异常检测','title_cn')
    p('NASA SMAP/MSL 正常子空间、局部密度与阈值失配研究','subheading')
    p('AIAA 3111 Course Project  |  Group XX［待填真实 Canvas 组号］','small')
    table([['Group Members［待填写］','Student IDs［待填写］'],['［成员 1 姓名］ / ［成员 2 姓名］','［成员 1 学号］ / ［成员 2 学号］'],['［成员 3 姓名］ / ［成员 4 姓名］','［成员 3 学号］ / ［成员 4 学号］']],[width*.57,width*.43])
    heading('1. Introduction')
    p('本项目的主要发现是：正常轨迹表示可以改善部分遥测异常的逐点识别，但正常训练覆盖不足会使“偏离正常”与“真实故障”混在一起。PCA 白化后的局部密度候选在 SMAP 上优于正常子序列近邻，在 MSL 上仍有大量误报。三项面向误报的机制实验没有带来可靠的跨数据集提升；这个负结果说明，选择更复杂的表示或更细的阈值，并不能自动解决工况迁移。')
    p('SMAP 卫星与 MSL 火星车需要持续监测大量遥测流。Telemanom 的工作表明，历史值与指令上下文能帮助建模正常行为，但真实异常稀少、工况变化频繁，统一人工阈值难以维护 [1]。我们的目标是用公开、专家标注的数据比较预测、重构与邻域三类路线，并分析哪些正常行为会触发错误告警。贡献是课程项目中的可复现实验实现、评估审计和机制分析，不是新理论、完整工程部署或 SOTA 声明。')
    heading('2. Problem Formulation')
    p('每个匿名通道给定一条被基准视为正常的训练序列，以及待检测测试序列。目标值 xₜ 是一个遥测量，其他列是匿名指令指示。检测器在收到 xₜ 后输出异常分数 sₜ、正常数据确定的阈值 τₜ 与告警 aₜ=1[sₜ&gt;τₜ]。它使用当前及过去观测，是因果异常检测；只有预测模型的预测本身提前产生，异常分数仍须等待真实值。')
    p('主任务为正常训练的 anomaly detection。测试异常闭区间仅用于开发选型和最终评价；模型拟合及阈值校准不使用异常标签。工程目标是以可解释的误报负担发现异常事件，评价目标则同时考察逐点 F1、误报率、事件召回与延迟。公开标签用于评价，不足以判断告警对应的具体物理故障。')
    # Page 2 methodology.
    page();heading('3. Methodology');sub('3.1 Data preprocessing and workflow')
    p('数据获取后先核验字段、连续样本索引、有限值、标签长度与冲突通道。每条正常序列按时间划为拟合、正常选择、正常校准三段；新增均值/标准差只由拟合段估计，低方差通道使用尺度下限约定。窗口不跨段边界，不把不同匿名通道串成长序列。所有方法从测试索引 64 起逐点评分，避免稀疏预测或把未预测点填成零带来的不公平。')
    table([['数据与审计','正常学习','开发与冻结','确认与分析'],['公开训练/测试 + 闭区间标注','拟合表示和模型 → 正常选择 → 正常校准','开发通道比较 → 配置与源码哈希冻结','固定模型逐点推理 → 指标、通道分析、案例']], [width/4]*4)
    p('Figure 1. 端到端数据挖掘流程。异常标签流入开发选型和指标计算，模型拟合/校准仅接收正常数据。确认通道在本轮冻结后仅评价一次；历史暴露另外披露。','caption')
    sub('3.2 A normal subspace followed by local density')
    p('以目标值构造 64 点尾部窗口 wₜ=[xₜ₋₆₃,…,xₜ]。PCA 在正常拟合窗口上保留 95% 方差；白化将各保留方向按其正常标准差缩放，减轻高方差方向对距离的支配。只保留正常子空间也可能丢弃低方差异常方向，因此不能只看成功案例。正常参考窗口以步长 2 采样，测试窗口逐点构造。')
    p('局部密度检测沿用 LOF 的可达距离机制 [4]：对查询 z 与其正常邻居 o，reach(z,o)=max(‖z−o‖,dₖ(o),ε)；lrd(z)=1/mean reach(z,o)；s(z)=mean lrd(o)/lrd(z)，其中 k=50。这是先学一个表示再进行单一密度判别，没有不同模型分数拼接或投票。参考方法以正常校准分数的 99.5% 分位作为告警阈值。')
    p('重复或近恒定轨迹会产生零距离与退化协方差。实现使用直接距离的 Ball-tree、确定性舍入、参考数据确定的距离下限，以及 PCA 特征值下限；常数通道回退到以正常窗口均值为中心的坐标。修复后验证完整序列和前缀得分一致。数值稳定保证计算可复现，但不能保证训练覆盖了测试工况。')
    sub('3.3 Comparison families and primary sources')
    p('预测路线采用目标值/指令条件 GRU，并含因果多步残差；重构路线为去噪自编码器、USAD 改编与时序 PCA；邻域路线为正常子序列近邻和稳定局部密度。我们阅读 Telemanom、USAD、TranAD 与 TSB-AD 的原论文摘要或官方源代码 [1–5]。TranAD 与 M2N2 仅用于机制调研，未训练；所有本地表格都来自本项目实测，绝不引用论文分数充当对照。')
    p('GRU 以正常选择误差定模型状态，单步/多步分别比较是否加入指令；去噪自编码器从加噪/遮挡窗口恢复正常轨迹；USAD 以共享编码器和双解码器交替优化重构与对抗目标 [2]。时序 PCA 测量窗口离正常子空间的重构残差，子序列近邻测量到正常参考窗口的距离；局部密度则比较邻域疏密，三者回答的偏离问题不同。','small')
    # Page 3 dataset/protocol/audit.
    page();heading('4. Experiments');sub('4.1 Dataset')
    p('数据来自 Telemanom 公开 NASA SMAP/MSL 基准，通过固定版本镜像获取 [1,6]。每个文件是一个独立目标通道及其指令上下文；SMAP 为 25 输入维，MSL 为 55 输入维，不能把它们解释为可同步的 25/55 个物理传感器。时间、通道与指令语义已匿名化，因此延迟只以采样点表示。')
    table([['数据集','有效通道','正常训练点','测试点','异常点 / 事件'],['SMAP','53','135,183','427,617','54,696 / 67'],['MSL','27','58,317','73,729','7,766 / 36']],[60,55,85,80,width-280])
    p('P-2 的两条标注给出了冲突区间，均排除，不能重复计数；T-10 有数据但不在有效标注清单。共保留 80 通道。正常训练段的“正常”来自基准设定，并不是本项目对每个训练点独立审定。官方发布已经按测试 min/max 缩放，属于继承的测试信息，后续 fit-only 标准化无法撤销这一限制。')
    sub('4.2 Evaluation protocol')
    p('通道 ID 的固定哈希将 14 SMAP+8 MSL 分为开发，39 SMAP+19 MSL 分为确认。正常序列通常按 60%/20%/20% 划分，短通道给后两段各保留至少 96 点；最短 D-12 使用 120/96/96。正常选择段用于神经 checkpoint，校准段用于阈值；开发异常标签才用于方法选择。所有对照共享 t≥64 的支持与闭区间标签。丢弃的 warmup 中没有真实异常点。')
    p('主指标先在每个数据集汇总 TP/FP/FN，再算 F1=2TP/(2TP+FP+FN)，最后对 SMAP/MSL 等权平均。误报率为 FP/(FP+TN)。事件召回按真实事件至少被命中一点计算；命中一点评价事件成功，绝不把整个区间补为逐点正确，即不做 point adjustment。AP 为逐通道原始分数排序的宏平均；延迟只统计已发现事件，并另报事件召回以避免隐藏漏检。')
    sub('4.3 Audit and strength of evidence')
    p('历史 3,112 个评分文件全部通过长度、有限分数、闭区间标签和阈值判定检查；80 个新模型的重载与前缀推理通过。旧稀疏 LSTM 表格不能被当作同协议提升证据。更关键的是：较早实验已查看所有官方测试，且子空间参数网格曾同时评价两种划分。该脚本现限制为开发，并保存原代码，但历史暴露不可撤销。以下确认结果属于受污染的探索性评价；本轮冻结仅约束新的选择过程，不能创造全新独立测试。')
    # Page 4 historical method comparison.
    page();sub('4.4 Local comparison of prediction reconstruction and density')
    order=['subspace_lof50','trajectory_lof50','subsequence_knn','dae','usad','hankel_pca_fixed','trajectory_gaussian_fixed','gru_target','gru_context','multi_target','multi_context','ztarget','zmax','pca','iforest']
    rows=[['方法','SMAP F1','MSL F1','等权均值']]
    for name in order:
        a=find(old,'SMAP',name);b=find(old,'MSL',name);rows.append([NAMES[name],fmt(a['f1']),fmt(b['f1']),fmt((float(a['f1'])+float(b['f1']))/2)])
    table(rows,[width*.49,width*.17,width*.17,width*.17])
    p('Table 1. 58 确认通道的本地同支持比较。各旧方法的平滑配置按开发结果固定；不同方法的平滑不同，因此此表是算法配置比较，不能把差值全部归因于表示。原始轨迹 LOF 是此前开发预选方法；子空间密度是在确认比较后被突出报告的探索候选。','caption')
    p('子空间密度相对近邻提升主要来自 SMAP：F1 从 0.378 到 0.487，误报率从 32.1% 降到 11.9%，但事件召回从 84.9% 降到 66.0%。它倾向于更少地持续报警，却漏掉一些事件。MSL 的时序 PCA F1=0.304，高于密度候选 0.266；因此“密度全面领先”的结论不成立。')
    p('指令条件在本次设置下没有稳定改善逐点 F1：单步 GRU 的 SMAP 略好、MSL 较差；多步指令 GRU 与目标值版本差异也较小。指令可能解释正常执行行为，但匿名指令频率、稀疏性、训练覆盖和具体模型容量共同影响效果，不能从一个负结果推断指令没有价值。现有多步 GRU 依据过去输入产生多个未来预测，再在每个目标点聚合已产生的残差，没有稀疏填零。')
    p('相同的不平滑配置下，子空间密度对原始轨迹密度的确认 F1 增量约为 SMAP +0.042、MSL +0.049，支持表示会影响密度判别；它仍是描述性消融。额外 GRU 随机种子结果与完整表保留在附录数据，避免只展示最佳神经初始化。')
    # Page 5 new study.
    page();sub('4.5 Predeclared mechanism experiments')
    p('针对 MSL 误报，我们先提出机制假设，限定以下三个改变；窗口、参考采样、密度邻居数与目标任务保持一致。不搜索新的确认阈值，不进行投票，不给不同数据集挑不同赢家。')
    table([['假设与分支','正常数据确定的改变','预期及风险'],['正常覆盖不足','取正常选择/校准两段99.5%阈值的较大者','可降低误报，也可能损失真实异常召回'],['阈值随水平变化','拟合段窗口中位数四分位分层；正常校准逐层99.5%；不足30点回退全局','适配异方差；稀有工况可能无校准覆盖'],['波形表示过敏','8个稳健水平/变差/漂移特征；拟合段稳健尺度 + 同一LOF','降低维数，可能丢失细微形状异常']],[width*.24,width*.43,width*.33])
    p(f'四个分支先在开发通道评价；水平分层阈值以开发等权平均 F1=0.277 被选中，参考为0.245。配置、开发 CSV 和源代码哈希冻结后，一次性评价确认通道。表中的其他分支是预先声明的机制消融，不是可在确认后重新选择的菜单。本轮预选分支确认平均 F1={newmean:.3f}，低于原探索参考0.376。')
    pic('mechanism_f1.png','Figure 2. 开发收益没有稳定转移到确认。稳健轨迹在 SMAP 确认略好，但 MSL 退步；不能将不同分支的最好分数拼成一个方法。')
    p('正常覆盖阈值在开发 MSL 降低误报，但召回同步下降，确认平均 F1 变化仅约 +0.001。水平分层在确认 SMAP 增加事件召回，也增加大量误报；MSL 改善很小。稳健表示的失败提示，压缩成低维描述量会牺牲一些与异常有关的轨迹细节。这些都是应保留的结果，而非删除失败后另挑设置。')
    p('本轮预选分层方法的 SMAP 事件召回为75.5%，正常点FPR16.7%；MSL事件召回92.6%，FPR33.5%。相比参考，它在SMAP多命中5个事件，却增加12,856个误报点；MSL事件命中数保持25/27。事件覆盖改善仍须以误报负担为代价评估。','small')
    # Page 6 MSL diagnostics.
    page();sub('4.6 Why false alarms concentrate in MSL')
    p(f'原候选 MSL 的正常点误报率为32.9%，且三个通道 P-14、M-7、C-2 贡献 {evidence["msl_top3_fp_share"]*100:.1f}% 的误报。P-14 拟合标准差约0.00008，正常测试值全部超出拟合0.1%–99.9%范围；C-2 拟合值为常数。此时密度检测把未覆盖的测试轨迹判成稀疏，并不意味着它识别了物理故障。')
    pic('regime_shift.png','Figure 3. 左：MSL 误报点集中度。右：正常测试值超出拟合范围的比例与误报率；标签只用于事后分析，不参与模型。')
    p('MSL 超出拟合范围的正常点误报率92.8%，范围内为12.6%；这个关联支持工况覆盖失配。但 M-7 的正常值几乎都仍在幅值范围内，误报率却100%，说明静态幅值范围不能解释全部问题，训练轨迹中的转换与局部邻域覆盖同样重要。99.5% 校准分位并不意味着测试 FPR 应为0.5%，时间依赖与分布迁移破坏了这种解释。')
    pic('normal_test_score_shift.png','Figure 4. P-14 正常校准分数与标注正常测试分数明显分离；增加第二正常段的阈值覆盖也难以跨越未知工况。')
    p('指令变化附近的误报关联在两个数据集方向不同：MSL 的最近64点有指令变化/无变化正常点 FPR约34.0%/18.5%，SMAP约5.0%/19.1%。它受到通道组成与工况混杂，不是指令导致误报的因果证据。可行下一步是有真实工况标签的条件正常模型，而不是对当前匿名测试继续调参。','small')
    # Page 7 cases.
    page();sub('4.7 Success and failure cases')
    pic('success_failure_cases.png','Figure 5. 每个数据集按水平分层相对参考的通道 F1 变化，分别选最大改善和最大退步。左为遥测与真实异常阴影；右为同一密度分数、全局阈值与分层阈值。全序列展示保留误报背景，阴影只作事后解释。')
    table([['通道','参考 F1','分层 F1','解释边界']]+[[r['channel'],fmt(r['reference_f1']),fmt(r['selected_f1']),('最大改善' if r['selection']=='largest_gain' else '最大退步')+'，不代表总体'] for r in evidence['cases']],[60,65,65,width-190])
    p('F-2 的阈值变化把TP从91增加到1,528，FP仅从0增加到2，主要恢复了异常持续区间的识别，两种方法都已命中该事件。A-2 的TP从71增加到82，FP却从122增至479，较低局部阈值损害逐点平衡。MSL 两个极值的增减较小，且改善案例绝对 F1仍低。阈值分层不会改变原始分数排序，因此其宏 AP 与参考相同；更高事件召回不能被当成更好的异常排序。')
    # Page 8 uncertainty and conclusion.
    page();sub('4.8 Channel stability and uncertainty')
    chosenboot=[r for r in boot if r['method']==selected]
    table([['分层减参考','F1差值','95%通道 bootstrap区间']]+[[r['dataset'],f'{float(r["delta_f1"]):+.3f}',f'[{float(r["ci_low"]):+.3f}, {float(r["ci_high"]):+.3f}]'] for r in chosenboot],[width*.34,width*.2,width*.46])
    p('5,000 次重采样在每个数据集内成对抽取整条通道，保留通道内时间相关性；不按每点独立抽样。水平分层在 SMAP 的通道 F1 为11胜/6负/22平，MSL为2胜/2负/15平；中位通道增量均为0。微平均改善与“多数通道获益”是不同问题。全部分支的区间都包含0，历史选型偏差未校正，同一航天器的通道也未必独立，因此不作显著性或普适优势声明。')
    pic('channel_ecdf.png','Figure 6. 确认通道的 F1 分布。总体微平均会被长通道与多异常点通道影响，分布图揭示大量低分通道。',w=width)
    heading('5. Conclusion and Limitations')
    p('我们保留正常轨迹表示与稳定密度作为可运行的探索性参考，并诚实报告本轮开发选择没有可靠提升。实际 insight 是：误报既涉及静态范围未覆盖，也涉及范围内的时序轨迹未覆盖；局部阈值会改变召回/误报取舍，却不会自动改善异常排序。复杂神经模型、指令上下文与稳健特征都需要具体协议证据，不能用名称替代比较。')
    p('限制包括公开测试缩放、确认历史暴露、正常训练假设、匿名工况语义、通道划分敏感性以及高误报。监督跨通道实验使用了源通道异常标签，是不同信息条件，结果只作补充。后续应先锁定模型，再引入未接触的新数据或运行工况；采集正常工况覆盖与指令含义，验证固定或安全适配的条件模型，并以告警负担和漏事件共同评价。现有包可复现课程实验，但不适合直接上线或宣称故障诊断能力。')
    # Excluded material.
    page();heading('6. Disclosures');sub('External Collaborators')
    p('组员姓名、学号、分工及任何外部人工协作者均未提供，故本文不编造人名或贡献。提交前须由组员填写真实信息，并明确是否有外部协作者；如有，应列出其身份与实际贡献。公开算法作者的贡献在参考文献与第三方许可中注明，不算作本组原创方法。')
    sub('AIGC Use')
    p('OpenAI Codex（本机课程项目对话与本次继续工作）辅助完成课程文件核对、原论文/官方实现阅读、实验代码实现、协议测试、模型推理、统计分析、图表、报告草稿、海报和打包。实验数值由本地程序从公开数据实际计算，旧输出与失败结果保留。AI 未生成异常标签，也未编造组员身份或论文成绩。')
    p('自动检查已覆盖代码与数据一致性、未来扰动和重载、PDF页数/排版及ZIP完整性，但这些不是学生已完成理解和人工审阅的证明。此报告和海报仍是组员可核验、修改、署名的研究草稿。课程原文要求不要直接提交 AI-generated work；组员须亲自检查论证、修改文字、核对课程允许的使用方式，并保留本披露后才决定提交。')
    p('本项目贡献限定为课程实验性实现、比较和审计。PCA、LOF、GRU、自编码器以及USAD训练机制来自已有研究；数值修复与阈值/特征实验不被宣称为全新理论。代码和来源文件保留各自权利与许可。')
    sub('Evidence provenance')
    p('论文与实现链接、实际读取摘录、抓取时间和源文件哈希保存在研究来源清单。数据版本固定，训练/测试每个文件有 SHA-256。新实验的预声明、冻结配置、开发结果哈希、确认开始记录与逐点输出可审查。PDF引用的本地数值来自已保存CSV；论文表格分数没有参与本地结果排名。')
    p('报告主文为前8页；本页Disclosures、下一页References及最后Appendix不计入课程6–10页主文限制。组号/组员/学号仍是明确待填写项，海报后续格式和最终截止以课程通知为准。','small')
    page();heading('References')
    refs=[
      '[1] Hundman, K., Constantinou, V., Laporte, C., Colwell, I., and Soderstrom, T. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. KDD, 2018. DOI: 10.1145/3219819.3219845. https://arxiv.org/abs/1802.04431 ; official implementation https://github.com/khundman/telemanom .',
      '[2] Audibert, J., Michiardi, P., Guyard, F., Marti, S., and Zuluaga, M. A. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. KDD, 2020. DOI: 10.1145/3394486.3403392. Official implementation https://github.com/manigalati/usad .',
      '[3] Tuli, S., Casale, G., and Jennings, N. R. TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data. Proceedings of the VLDB Endowment 15(6), 1201–1214, 2022. https://arxiv.org/abs/2201.07284 ; https://github.com/imperial-qore/TranAD .',
      '[4] Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. LOF: Identifying Density-Based Local Outliers. SIGMOD, 2000. DOI: 10.1145/342009.335388. Novelty-mode implementation guidance: https://scikit-learn.org/1.5/modules/outlier_detection.html .',
      '[5] Liu, Q., and Paparrizos, J. The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark. NeurIPS Datasets and Benchmarks, 2024. Official benchmark, method implementations and citation: https://github.com/TheDatumOrg/TSB-AD .',
      '[6] Telemanom public data mirror, appleparan/telemanom. https://huggingface.co/datasets/appleparan/telemanom . Revision 2d22e1061be83a88b7b9e48df35163d5147adc9d. Accessed 2026-10-06. Original NASA dataset provenance is described in [1].']
    for r in refs:p(html.escape(r),'small')
    sub('How the sources were used')
    p('阅读范围有意限定为实际取回的摘要、README、配置与模型/评价源代码。Telemanom提供预测与指令机制及测试缩放事实；USAD提供双解码器对抗损失；TranAD提供自条件注意力机制与含point adjustment的评价代码；TSB-AD提供局部密度和M2N2测试期适配的机制对照。未声称通读所有论文正文，也未声称精确复现未训练模型。')
    p('来源快照及许可位于 research_v2、research_v4 与 THIRD_PARTY_v2.md；官方主分支源文件另附抓取SHA-256，避免将抓取后变化误认成当时所读版本。未许可文件不被赋予本项目的新版权声明。','small')
    page();heading('Appendix');sub('A. Separate supervised group-held-out study')
    sup=read(ROOT/'results_v3/group5/summary.csv');table([['方法','SMAP F1','MSL F1','信息条件']]+[[('HGB' if m=='hgb' else 'Logistic'),fmt(find(sup,'SMAP',m)['f1']),fmt(find(sup,'MSL',m)['f1']),'有标签源通道 + 目标正常适配'] for m in ['hgb','logistic']],[width*.2,width*.18,width*.18,width*.44])
    p('五个外层通道组分别作为目标，下一组作内层验证，其余三组训练；深度/阈值仅由内层组选择，然后用四个源组重拟合。每个目标通道的异常标签仅评价自身预测。不同外层折会轮流使用同一通道作为源，故它是group-held-out重采样而非全新数据。全80通道上的HGB平均F1=0.280，原密度参考为0.343；这不同于58通道主表，且不能视为无监督路线的等信息条件比较。')
    sub('B. Reproduction and supplemental evidence')
    table([['材料','内容'],['README + configuration','依赖、固定数据下载、正常准备、全流程和单通道推理'],['Per-channel and point CSV/NPZ','全部四分支结果、阈值、标签与告警；旧多方法/多种子结果'],['Bootstrap and channel stability','5000次分层配对通道抽样、胜负平与留一通道敏感性'],['Models and provenance','80个新拟合模型、旧参考模型、源代码/数据/来源哈希'],['Audit and checklist','P-2冲突行、3112评分支持检查、历史确认暴露与课程缺口']],[width*.36,width*.64])
    p('所有评分支持为t=64,…,n−1，64点检测窗口结束于t；预测窗口只包含预测目标之前的观测。LOF参考样本不含校准或测试观测。水平分层阈值按正常拟合窗口水平定边界，按正常校准得分定阈值，稀疏层回退全局，测试输入不改变这些边界。稳健描述量的尺度也仅由拟合段确定。')
    p('完整代码和依赖进入提交包；原始版本与失败结果在项目中保留。ZIP根目录为Report.pdf、README、代码和完整当前公开数据；Poster.pdf单独提交。学生身份、人工修订、海报格式核对、现场答辩、互评和Canvas上传属于尚需真实组员完成的步骤。')
    def foot(c,d):
        c.saveState();c.setFont(FONT,8);c.setFillColor(GRAY);c.drawString(72,38,'AIAA 3111 | NASA SMAP/MSL | Group XX');c.drawRightString(W-72,38,str(d.page));c.restoreState()
    SimpleDocTemplate(str(dest/'Report.pdf'),pagesize=A4,leftMargin=72,rightMargin=72,topMargin=60,bottomMargin=58,title='Normal trajectory coverage and spacecraft anomaly detection',author='Group XX - member information pending').build(story,onFirstPage=foot,onLaterPages=foot)
    (dest/'Report_source.md').write_text('\n\n'.join(source),encoding='utf-8')
    poster(analysis,run,dest,conf,newmean)
    print(dest/'Report.pdf');print(dest/'Poster.pdf')

def poster(analysis,run,dest,conf,newmean):
    # A1 landscape; later course poster requirements are not yet supplied.
    W,H=2383.94,1683.78;c=canvas.Canvas(str(dest/'Poster.pdf'),pagesize=(W,H));c.setTitle('NASA SMAP/MSL anomaly detection - Group XX')
    c.setFillColor(colors.white);c.rect(0,0,W,H,fill=1,stroke=0);c.setFillColor(INK);c.rect(0,H-265,W,265,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont(BOLD,62);c.drawString(75,H-105,'正常轨迹覆盖决定遥测异常检测的可靠性')
    c.setFont(FONT,29);c.drawString(78,H-167,'NASA SMAP/MSL  |  AIAA 3111  |  Group XX［待填］  |  姓名 / 学号［待填］')
    c.setFont(FONT,26);c.drawString(78,H-220,'研究结论：子空间密度改善部分识别；分层阈值没有带来可靠的跨数据集提升。')
    margin=72;gap=42;colw=(W-2*margin-2*gap)/3;tops=H-315
    styles=ParagraphStyle('poster',fontName=FONT,fontSize=24,leading=38,wordWrap='CJK',textColor=INK)
    title=ParagraphStyle('poster_heading',fontName=BOLD,fontSize=34,leading=44,textColor=BLUE,spaceAfter=16)
    def txt(x,y,text,head=False):
        p=Paragraph(text,title if head else styles);_,h=p.wrap(colw,y-95);p.drawOn(c,x,y-h);return y-h-22
    def img(x,y,name):
        f=analysis/'figures'/name;im=ImageReader(str(f));iw,ih=im.getSize();h=colw*ih/iw;c.drawImage(im,x,y-h,width=colw,height=h,preserveAspectRatio=True);return y-h-24
    x=margin;y=tops;y=txt(x,y,'问题与数据',True);y=txt(x,y,'在正常训练的遥测中发现异常时间点。公开数据包含53个SMAP和27个MSL通道；每条是目标值与匿名指令，不是同步物理传感器组。')
    y=txt(x,y,'正常学习与公平评价',True);y=txt(x,y,'正常拟合 → PCA95 / 白化 → LOF50 → 正常校准99.5%阈值。64点尾部窗口，逐点评分，闭区间标签，无point adjustment。开发22通道，确认58通道。')
    y=txt(x,y,'冻结后的机制检验',True);y=txt(x,y,'先声明三个假设：正常覆盖不足、阈值随水平变化、波形特征过敏。开发选择水平分层，冻结后一次性评价；不按确认分数换赢家。')
    y=img(x,y,'mechanism_f1.png');y=txt(x,y,'分层：SMAP/MSL F1=0.469/0.268；原参考=0.487/0.266。开发收益未稳定转移。')
    x=margin+colw+gap;y=tops;y=txt(x,y,'关键发现：MSL误报集中',True);y=txt(x,y,'原候选MSL误报率32.9%。P-14、M-7、C-2贡献62.3%的误报；幅值范围未覆盖与范围内轨迹未覆盖同时存在。')
    y=img(x,y,'regime_shift.png');y=img(x,y,'normal_test_score_shift.png');y=txt(x,y,'MSL范围外正常点FPR92.8%，范围内12.6%。P-14近恒定训练却有变化测试；M-7范围内也全部告警。正常分位阈值不保证测试误报率0.5%。')
    y=txt(x,y,'实际意义',True);y=txt(x,y,'需要补充正常工况和指令语义覆盖；告警应由工程人员核验。密度稀疏不等同于物理故障，不能直接上线。')
    x=margin+2*(colw+gap);y=tops;y=txt(x,y,'成功与失败都保留',True)
    # Crop no case evidence: use the complete four-case panel at its natural ratio.
    y=img(x,y,'success_failure_cases.png');y=txt(x,y,'每个数据集按F1增量自动选最好/最差案例。分层阈值恢复部分异常点，也增加正常告警；它不改变原始分数排序。')
    y=txt(x,y,'证据边界与下一步',True);y=txt(x,y,'全部官方测试此前已被查看，且公开数据继承测试min/max缩放。通道bootstrap未校正历史选型偏差；没有独立最终测试或SOTA声明。下一步锁定方法后使用新工况/新数据。')
    c.setFillColor(LIGHT);c.rect(0,0,W,98,fill=1,stroke=0);c.setFillColor(INK);c.setFont(FONT,18)
    c.drawString(72,62,'Sources: Hundman et al. KDD2018; Audibert et al. KDD2020; Tuli et al. PVLDB2022; Liu & Paparrizos NeurIPS2024; Breunig et al. SIGMOD2000.')
    c.drawString(72,30,'AIGC披露：Codex辅助调研、代码、实测分析与图文草稿；组员须核验、修改和填写身份后提交。A1横版草稿，最终海报格式待课程通知。')
    c.save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',default=str(ROOT/'results_v4'));p.add_argument('--run',default=str(ROOT/'results_v4/mechanism'));p.add_argument('--output',default=str(ROOT/'submission'));a=p.parse_args();build(Path(a.analysis),Path(a.run),Path(a.output))
