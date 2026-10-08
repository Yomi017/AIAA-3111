"""Read-only historical audit; save evidence without rewriting earlier experiments."""
from pathlib import Path
import json, hashlib, ast, sys
import numpy as np
import pandas as pd
from experiment_v2 import ROOT, DATA, L, label_array, split_boundaries

OUT=ROOT/'results_v4'
def main():
    OUT.mkdir(exist_ok=True)
    labs=pd.read_csv(DATA/'labeled_anomalies.csv')
    eligible=labs[labs.chan_id!='P-2'].copy()
    records=[]
    for _,r in eligible.iterrows():
        tr=pd.read_parquet(DATA/'data/train'/f'{r.chan_id}.parquet')
        te=pd.read_parquet(DATA/'data/test'/f'{r.chan_id}.parquet')
        y=label_array(r,len(te));a,b=split_boundaries(len(tr))
        assert tr.columns.tolist()==te.columns.tolist() and tr.columns[1]=='value'
        assert np.array_equal(tr.timestep,np.arange(len(tr))) and np.array_equal(te.timestep,np.arange(len(te)))
        assert np.isfinite(tr.to_numpy()).all() and np.isfinite(te.to_numpy()).all()
        records.append(dict(channel=r.chan_id,dataset=r.spacecraft,train_n=len(tr),test_n=len(te),features=len(tr.columns)-1,fit_n=a,selection_n=b-a,calibration_n=len(tr)-b,anomaly_points=int(y.sum()),warmup_anomaly_points=int(y[:L].sum()),events=len(ast.literal_eval(r.anomaly_sequences)),split='development' if int(hashlib.sha256(r.chan_id.encode()).hexdigest(),16)%3==0 else 'confirmation'))
    pd.DataFrame(records).to_csv(OUT/'data_inventory.csv',index=False)
    look=eligible.set_index('chan_id');checks=[]
    for version in ['results_v2','results_v3']:
        for f in (ROOT/version).rglob('*.npz'):
            c=f.parent.name if version=='results_v2' else f.stem
            if c not in look.index: continue
            z=np.load(f)
            if 'score' not in z: continue
            row=look.loc[c];y=label_array(row,int(row.num_values))[L:]
            ok=len(z['score'])==len(y) and np.isfinite(z['score']).all() and np.array_equal(z['label'],y)
            threshold_ok=True
            if 'threshold' in z and 'pred' in z: threshold_ok=np.array_equal(z['pred'],z['score']>z['threshold'])
            checks.append(dict(path=str(f.relative_to(ROOT)),dense_aligned_inclusive=bool(ok),stored_threshold_matches_prediction=bool(threshold_ok)))
    pd.DataFrame(checks).to_csv(OUT/'historical_score_audit.csv',index=False)
    snapshot_files=list(ROOT.glob('*.py'))+list((ROOT/'results_v2').glob('*.csv'))+list((ROOT/'results_v3').rglob('done.json'))
    evidence=dict(p2_conflicts=labs[labs.chan_id=='P-2'].to_dict('records'),effective_counts=eligible.groupby('spacecraft').size().to_dict(),saved_score_files=len(checks),failed_support=sum(not x['dense_aligned_inclusive'] for x in checks),failed_prediction=sum(not x['stored_threshold_matches_prediction'] for x in checks),confirmation_exposure=['v0/v1 inspected official tests','v2 comparative analysis and post-confirmation emphasis of subspace candidate','grid_subspace_v3.py enumerates 4 variance levels x 4 neighbor counts on BOTH splits; grid_subspace.csv exists','v3 supervised grouped resampling uses all80 labeled tests as source/target across folds'],report_v2='Seven physical pages; experimental note, not template-compliant formal report. References/disclosures not separately paginated.',historical_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in snapshot_files})
    # Preserve the first audit; later scans are additional evidence, not a rewrite.
    target=OUT/'audit.json' if not (OUT/'audit.json').exists() else OUT/'audit_latest.json'
    target.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='historical_hashes'},indent=2))
if __name__=='__main__':main()
