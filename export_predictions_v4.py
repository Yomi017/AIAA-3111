"""Readable per-point CSV exports from fixed saved scores; no fitting or reselection."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_v2 import ROOT,DATA,L
def main(run):
    method=json.loads((run/'frozen_selection.json').read_text())['selected'];count=0
    for stage in ['development','confirmation']:
        for f in (run/stage).glob('*/model.joblib'):
            c=f.parent.name;z=np.load(f.parent/(method+'.npz'));values=pd.read_parquet(DATA/'data/test'/f'{c}.parquet').value.to_numpy()[L:]
            dest=run/'prediction_csv'/stage;dest.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(timestep=np.arange(L,L+len(values)),value=values,score=z['score'],threshold=z['threshold'],anomaly=z['pred'].astype(int),evaluation_label=z['label'])).to_csv(dest/(c+'.csv'),index=False);count+=1
    assert count==80;print('Exported',count,'fixed selected-arm prediction CSVs; evaluation_label is never an inference input')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(ROOT/'results_v4/mechanism'));main(Path(p.parse_args().run))
