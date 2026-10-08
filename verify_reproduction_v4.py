"""Compare an actual fresh replication against the original frozen study."""
from pathlib import Path
import json,hashlib
import numpy as np
from experiment_v4 import ROOT,METHODS
def main():
    original=ROOT/'results_v4/mechanism';fresh=ROOT/'results_v4/replication_delivery_check';count=0
    for stage in ['development','confirmation']:
        for suffix in ['channels.csv','summary.csv']:
            a=original/f'{stage}_{suffix}';b=fresh/a.name
            assert a.read_bytes()==b.read_bytes(),a.name
        for f in (original/stage).glob('*/model.joblib'):
            for method in METHODS:
                a=np.load(f.parent/(method+'.npz'));b=np.load(fresh/stage/f.parent.name/(method+'.npz'))
                for k in ['score','threshold','pred','label']:assert np.array_equal(a[k],b[k]),(f.parent.name,method,k)
                count+=1
    selected=json.loads((original/'frozen_selection.json').read_text())['selected'];assert selected==json.loads((fresh/'frozen_selection.json').read_text())['selected']
    record=dict(fresh_replication='results_v4/replication_delivery_check',selected=selected,compared_score_outputs=count,metric_csv='byte identical',all_point_scores_thresholds_labels_predictions='exactly identical under current environment',full_flow='data verification, audit, development, freeze, confirmation, analysis, PDF/poster generation executed; added model checks and CSV export executed on fresh outputs',independent_evidence=False,boundary='Replication checks computation; cannot undo historical test exposure',source_sha256=hashlib.sha256((ROOT/'experiment_v4.py').read_bytes()).hexdigest())
    (ROOT/'results_v4/reproduction_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
