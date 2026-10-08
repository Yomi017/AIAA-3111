"""Package the current reproducible experiment; never remove historical results."""
from pathlib import Path
import json,hashlib,zipfile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_v2'
archive=ROOT/'NASA_SMAP_MSL_v2_实验与模型.zip'
files=list(ROOT.glob('*v2.py'))+list(ROOT.glob('*v3.py'))+[ROOT/f for f in ['README_v2.md','THIRD_PARTY_v2.md','requirements_v2.txt','reproduce_v2.ps1','AIAA3111_requirements_checklist.md']]
files+=list((ROOT/'research_v2').glob('*'))
files+=list((ROOT/'research_v3').glob('*'))
files+=list((ROOT/'results_v3').rglob('*'))
files+=list((ROOT/'data/telemanom/data').rglob('*.parquet'))+[ROOT/'data/telemanom/labeled_anomalies.csv']
for directory in ['dev42','confirm42','confirm7','confirm2026','extra42','extra_confirm','recon42','recon_confirm','stable_dev','stable_confirm','models','figures','deliverables']:
    files.extend((OUT/directory).rglob('*'))
for name in ['frozen_selection.json','protocol_amendment.json','numerical_fix.json','provenance.json','confirmation_frozen.csv','confirmation_frozen_channels.csv','development_frozen.csv','development_frozen_channels.csv','paired_bootstrap.csv','ablation.csv','ablation_channels.csv','seed_robustness.csv','all80_descriptive.csv','case_selection.csv','export_verification.csv','reproduction_check.log']:
    files.append(OUT/name)
files.append(OUT/'deliverables'/'Report_v2.pdf')
files=sorted(set(f for f in files if f.is_file() and '__pycache__' not in f.parts))
manifest={str(f.relative_to(ROOT)).replace('\\','/'):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f in files:z.write(f,'NASA_SMAP_MSL_v2/'+str(f.relative_to(ROOT)))
    z.writestr('NASA_SMAP_MSL_v2/package_manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name,encoding='utf-8')
print(json.dumps(dict(archive=str(archive),files=len(files)+1,MB=round(archive.stat().st_size/1024**2,2),sha256=sha),ensure_ascii=True))
