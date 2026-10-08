"""Finalize source/data/artifact hashes after verification, retaining earlier provenance."""
from pathlib import Path
import json,hashlib,sys,platform,datetime,importlib.metadata as md,shutil
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'results_v4';current=out/'provenance.json';prior=out/'provenance_initial.json'
    if current.exists() and not prior.exists():shutil.copyfile(current,prior)
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.ps1'))+[ROOT/n for n in ['README.md','COURSE_CHECKLIST.md','THIRD_PARTY.md','requirements_v4.txt']]
    for folder in ['research_v2','research_v4','data/telemanom']:
        files.extend(f for f in (ROOT/folder).rglob('*') if f.is_file() and '/.cache/' not in f.as_posix())
    files.extend((out/'mechanism').glob('*.json'));files.append(out/'MECHANISM_PLAN.md')
    src={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
    artifacts={n:hashlib.sha256((ROOT/'submission'/n).read_bytes()).hexdigest() for n in ['Report.pdf','Poster.pdf','Report_source.md']}
    versions={n:md.version(n) for n in ['numpy','pandas','scikit-learn','pyarrow','torch','matplotlib','joblib','requests','reportlab','PyMuPDF','Pillow','beautifulsoup4']}
    record=dict(finalized_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),versions=versions,pdf_authoring_runtime='Codex bundled Python with ReportLab4.4.9; source/data experiments use listed Anaconda environment',data_revision=json.loads((ROOT/'results_v2/provenance.json').read_text())['huggingface_revision'],sha256=src,artifact_sha256=artifacts,prior_results_preserved=True,independence='All official test channels previously exposed; replication is numerical verification only')
    current.write_text(json.dumps(record,indent=2),encoding='utf-8');print('Final source/data/artifact hashes:',len(src),len(artifacts))
if __name__=='__main__':main()
