"""Verify or download the exact, pinned mirror bytes; never silently overwrite data."""
from pathlib import Path
import json,hashlib,argparse,urllib.request
ROOT=Path(__file__).resolve().parent
def main(download=False):
    provenance=json.loads((ROOT/'results_v2/provenance.json').read_text());revision=provenance['huggingface_revision'];count=0
    for rel,digest in provenance['sha256'].items():
        rel=rel.replace('\\','/')
        if not rel.startswith('data/telemanom/'):continue
        f=ROOT/rel
        if not f.exists():
            if not download:raise FileNotFoundError(f'{f}: use --download')
            remote=rel[len('data/telemanom/'):];url=f'https://huggingface.co/datasets/appleparan/telemanom/resolve/{revision}/{remote}?download=true'
            f.parent.mkdir(parents=True,exist_ok=True);part=f.with_suffix(f.suffix+'.part')
            with urllib.request.urlopen(url,timeout=120) as r,part.open('wb') as w:
                while chunk:=r.read(1024*1024):w.write(chunk)
            if hashlib.sha256(part.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Download hash mismatch: {f}; partial retained')
            part.replace(f)
        if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Existing data hash mismatch: {f}; preserve and investigate')
        count+=1
    print('Verified pinned dataset files:',count,'revision:',revision)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args().download)
