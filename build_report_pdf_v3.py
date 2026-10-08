from pathlib import Path
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

root=Path(__file__).resolve().parent; out=root/'results_v2'; src=out/'deliverables'/'实验结果与方法说明.html'; dest=out/'deliverables'/'Report_v2.pdf'
try: pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc')); font='CN'
except Exception: font='Helvetica'
s=getSampleStyleSheet(); s.add(ParagraphStyle('cnTitle',fontName=font,fontSize=20,leading=27,alignment=TA_CENTER,textColor=colors.HexColor('#23364a'),spaceAfter=16)); s.add(ParagraphStyle('cnH',fontName=font,fontSize=14,leading=20,textColor=colors.HexColor('#2563a6'),spaceBefore=14,spaceAfter=7)); s.add(ParagraphStyle('cnB',fontName=font,fontSize=8.8,leading=14,spaceAfter=6)); s.add(ParagraphStyle('cnS',fontName=font,fontSize=7,leading=9,textColor=colors.HexColor('#526777')))
def para(text,style='cnB'):
    text=text.replace('&nbsp;',' ').replace('→','-&gt;').replace('⇔','iff')
    return Paragraph(text,s[style])
def foot(c,d):
    c.saveState(); c.setFont(font,7); c.setFillColor(colors.HexColor('#62788e')); c.drawString(.65*inch,.4*inch,'AIAA 3111 - NASA SMAP/MSL anomaly detection'); c.drawRightString(7.85*inch,.4*inch,str(d.page)); c.restoreState()
doc=BeautifulSoup(src.read_text(encoding='utf-8'),'html.parser'); story=[]
for el in doc.select('main > *'):
    if el.name=='h1': story.append(para(el.get_text(' ',strip=True),'cnTitle'))
    elif el.name in ['h2','h3']: story.append(para(el.get_text(' ',strip=True),'cnH'))
    elif el.name=='p': story.append(para(el.decode_contents(),'cnB'))
    elif el.name=='div':
        text=el.get_text(' ',strip=True)
        if text: story.append(para(text,'cnB'))
    elif el.name=='img':
        p=out/'figures'/Path(el.get('src','')).name
        if p.exists(): story.append(Image(str(p),width=7.0*inch,height=4.0*inch)); story.append(para(el.get('alt',''),'cnS'))
    elif el.name=='table':
        rows=[]
        for tr in el.select('tr'):
            vals=[x.get_text(' ',strip=True) for x in tr.select('th,td')]
            if vals: rows.append(vals)
        if rows:
            from reportlab.platypus import Table,TableStyle
            rows=[[Paragraph(v,s['cnS']) for v in row] for row in rows]
            widths=[2.1*inch]+[(4.9*inch)/(len(rows[0])-1)]*(len(rows[0])-1) if len(rows[0])>1 else [7*inch]
            t=Table(rows,repeatRows=1,colWidths=widths); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf3f9')),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#d5e0ea')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)])); story.append(t); story.append(Spacer(1,6))
    elif el.name=='ul':
        for li in el.select(':scope > li'): story.append(para('- '+li.get_text(' ',strip=True)))
story += [PageBreak(),para('正式提交前检查','cnH'),para('课程要求：主文 6-10 页；包含标题、组号和组员、Introduction、Problem Formulation、Methodology、Experiments、Conclusion and Limitations、Disclosures、References；ZIP 包含 Report.pdf、代码、README、数据或下载说明；Poster.pdf 单独提交。当前报告已覆盖方法、数据、实验、限制、引用和 AIGC 披露。填写 Group XX、组员姓名/学号后即可导出最终提交版。')]
SimpleDocTemplate(str(dest),pagesize=letter,rightMargin=.65*inch,leftMargin=.65*inch,topMargin=.65*inch,bottomMargin=.65*inch).build(story,onFirstPage=foot,onLaterPages=foot); print(dest)
