"""PDF boundaries, glyphs, render QA artifacts and, when present, ZIP payload integrity."""
from pathlib import Path
import json,hashlib,zipfile,argparse
import fitz
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent
def pdf_checks(dest,render=True):
    out=ROOT/'results_v4/render_check';out.mkdir(exist_ok=True);record={};imgs=[]
    d=fitz.open(dest/'Report.pdf');assert len(d)==11
    assert '6. Disclosures' in d[8].get_text() and 'References' in d[9].get_text() and 'Appendix' in d[10].get_text()
    assert '5. Conclusion and Limitations' in d[7].get_text()
    for term in ['1. Introduction','2. Problem Formulation','3. Methodology','4. Experiments']:assert any(term in p.get_text() for p in d[:8])
    bounds=[]
    for i,p in enumerate(d):
        bad=[]
        for block in p.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0<0 or y0<0 or x1>p.rect.width+1 or y1>p.rect.height+1 or '\ufffd' in span['text']:bad.append(span)
        assert not bad,(i,bad);bounds.append(dict(page=i+1,characters=len(p.get_text()),out_of_page_spans=len(bad)))
        if render:
            f=out/f'report_page_{i+1:02}.png';p.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(f);imgs.append(f)
    record['report']=dict(total_pages=11,main_pages=8,excluded_pages=['Disclosures9','References10','Appendix11'],bounds=bounds)
    if not (dest/'Poster.pdf').exists():
        record['poster']=dict(status='separate submission; not included in course ZIP')
        return record
    poster=fitz.open(dest/'Poster.pdf');assert len(poster)==1;p=poster[0];assert abs(p.rect.width-2383.94)<1
    lower=[]
    for b in p.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox'];assert x0>=0 and x1<=p.rect.width+1 and y0>=0 and y1<=p.rect.height+1
                if y1>p.rect.height-98 and y0<p.rect.height-98:lower.append(span['text'])
    assert not lower,('Poster text crosses footer',lower)
    if render:
        p.get_pixmap(matrix=fitz.Matrix(.7,.7)).save(out/'poster.png')
        for start in [0,6]:
            chunk=imgs[start:start+6];sheet=Image.new('RGB',(3*397,2*589),(235,240,244));draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(chunk):
                im=Image.open(f);im.thumbnail((377,545));x=(j%3)*397+10;y=(j//3)*589+28;sheet.paste(im,(x,y));draw.text((x,y-20),f'Page {start+j+1}',fill=(30,40,50))
            sheet.save(out/f'contact_sheet_{start//6+1}.png')
    record['poster']=dict(pages=1,size='A1 landscape draft',footer_clear=True)
    return record
def zip_checks(f):
    with zipfile.ZipFile(f) as z:
        assert z.testzip() is None;names=set(z.namelist());assert {'Report.pdf','README.md','experiment_v4.py','data/telemanom/labeled_anomalies.csv','MANIFEST.json'}<=names
        assert 'Poster.pdf' not in names
        manifest=json.loads(z.read('MANIFEST.json'));assert set(manifest)==names-{'MANIFEST.json'}
        for name,r in manifest.items():
            b=z.read(name);assert len(b)==r['bytes'];assert hashlib.sha256(b).hexdigest()==r['sha256'],name
        return dict(zip=f.name,files=len(names),testzip='passed',manifest_hashes='all passed',sha256=hashlib.sha256(f.read_bytes()).hexdigest())
def main():
    default_dir=ROOT/'submission' if (ROOT/'submission/Report.pdf').exists() else ROOT
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(default_dir));p.add_argument('--skip-render',action='store_true');a=p.parse_args();dest=Path(a.directory)
    record=pdf_checks(dest,not a.skip_render)
    f=dest/'Group_XX_Project.zip'
    if f.exists():record['zip']=zip_checks(f)
    (ROOT/'results_v4/delivery_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
