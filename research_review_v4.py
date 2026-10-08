"""Snapshot primary source documents with retrieval timestamps and hashes."""
from pathlib import Path
import requests,json,hashlib,datetime
from bs4 import BeautifulSoup
from experiment_v2 import ROOT
def main():
    out=ROOT/'research_v4';out.mkdir(exist_ok=True);records=[]
    sources={
      'telemanom_paper':'https://arxiv.org/abs/1802.04431',
      'tranad_paper':'https://arxiv.org/abs/2201.07284',
      'usad_official':'https://raw.githubusercontent.com/manigalati/usad/master/usad.py',
      'telemanom_channel':'https://raw.githubusercontent.com/khundman/telemanom/master/telemanom/channel.py',
      'tsbad_lof':'https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/TSB_AD/models/LOF.py',
      'tsbad_m2n2':'https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/main/TSB_AD/models/M2N2.py',
      'tranad_pot':'https://raw.githubusercontent.com/imperial-qore/TranAD/main/src/pot.py'}
    for name,url in sources.items():
        try:
            r=requests.get(url,timeout=40);r.raise_for_status();suffix='.py' if url.endswith('.py') else '.html';f=out/(name+suffix);f.write_bytes(r.content)
            if suffix=='.html':(out/(name+'.txt')).write_text(BeautifulSoup(r.text,'html.parser').get_text(' ',strip=True),encoding='utf-8')
            records.append(dict(name=name,url=url,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=hashlib.sha256(r.content).hexdigest(),file=f.name,status='saved'))
        except Exception as e:records.append(dict(name=name,url=url,status='unavailable',error=str(e)))
    (out/'sources.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(records,indent=2))
if __name__=='__main__':main()
