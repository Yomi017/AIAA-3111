import json, urllib.request, time
from pathlib import Path
root=Path(r'E:\Class\AIAA 3111\Project\data\telemanom')
root.mkdir(parents=True, exist_ok=True)
api='https://huggingface.co/api/datasets/appleparan/telemanom/tree/main?recursive=true&expand=false'
with urllib.request.urlopen(api, timeout=60) as f: entries=json.load(f)
paths=[e['path'] for e in entries if e.get('type')=='file' and (e['path'].startswith('data/train/') or e['path'].startswith('data/test/') or e['path']=='labeled_anomalies.csv') and e['path'].endswith(('.parquet','.csv'))]
print('files',len(paths))
for i,p in enumerate(paths,1):
 out=root/p
 out.parent.mkdir(parents=True, exist_ok=True)
 if out.exists() and out.stat().st_size>0: continue
 url='https://huggingface.co/datasets/appleparan/telemanom/resolve/main/'+p+'?download=true'
 for attempt in range(4):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=120) as r, open(out,'wb') as w:
    while True:
     b=r.read(1024*1024)
     if not b: break
     w.write(b)
   break
  except Exception as e:
   print('retry',p,attempt,e); time.sleep(2)
 print(i,p,out.stat().st_size if out.exists() else 0)
print('done')
