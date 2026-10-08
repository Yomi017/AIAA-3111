"""Package only explicit deliverables; manifest covers every payload and archive is tested."""
from pathlib import Path
import hashlib,json,zipfile
from validate_submission_v4 import pdf_checks,zip_checks
ROOT=Path(__file__).resolve().parent
def main():
    dest=ROOT/'submission';pdf_checks(dest,False);files={}
    for f in ROOT.glob('*.py'):files[f.name]=f
    for name in ['README.md','README_v2.md','COURSE_CHECKLIST.md','THIRD_PARTY.md','THIRD_PARTY_v2.md','requirements_v2.txt','requirements_v4.txt','reproduce_v2.ps1','reproduce_v4.ps1']:files[name]=ROOT/name
    for folder in ['data/telemanom','research_v2','research_v3','research_v4','results_v0','results_v1','results_v2','results_v3','results_v4']:
        for f in (ROOT/folder).rglob('*'):
            if not f.is_file():continue
            rel=f.relative_to(ROOT).as_posix()
            if '/render_check/' in rel or '/replication_' in rel or '__pycache__' in rel or '/.cache/' in rel:continue
            files[rel]=f
    for name in ['Report.pdf','Report_source.md']:files[name]=dest/name
    for name in ['03_Project.pdf','02_Project_Assessment_Rubrics.pdf','05_Report_Template.docx']:files['course_requirements/'+name]=ROOT/name
    manifest={name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for name,f in sorted(files.items())}
    archive=dest/'Group_XX_Project.zip'
    if archive.exists():raise RuntimeError('Preserve existing archive; archive or rename it explicitly before rebuilding')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,f in sorted(files.items()):z.write(f,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    result=zip_checks(archive);(dest/'Group_XX_Project.zip.sha256').write_text(result['sha256']+'  Group_XX_Project.zip\n');(dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (dest/'package_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
