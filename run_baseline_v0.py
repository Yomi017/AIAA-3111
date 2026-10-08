import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score, roc_auc_score

ROOT=Path(r'E:/Class/AIAA 3111/Project')
DATA=ROOT/'data/telemanom/data'
OUT=ROOT/'results_v0'
OUT.mkdir(exist_ok=True)
labels=pd.read_csv(DATA/'../labeled_anomalies.csv')

rows=[]
for _,r in labels[labels.spacecraft=='SMAP'].iterrows():
    chan=r.chan_id
    trf=DATA/'train'/f'{chan}.parquet'; tef=DATA/'test'/f'{chan}.parquet'
    if not trf.exists() or not tef.exists(): continue
    tr=pd.read_parquet(trf); te=pd.read_parquet(tef)
    feats=[c for c in tr.columns if c!='timestep']
    Xtr=tr[feats].to_numpy(dtype=np.float64); Xte=te[feats].to_numpy(dtype=np.float64)
    mu=np.nanmean(Xtr,axis=0); sd=np.nanstd(Xtr,axis=0); sd[sd<1e-8]=1.0
    # validation is the final 20% of the normal training sequence
    ztr=np.abs((Xtr-mu)/sd).max(axis=1); zte=np.abs((Xte-mu)/sd).max(axis=1)
    cut=max(1,int(0.8*len(ztr)))
    thr=float(np.quantile(ztr[cut:],0.995))
    seqs=json.loads(r.anomaly_sequences.replace("'",'"')) if isinstance(r.anomaly_sequences,str) else r.anomaly_sequences
    y=np.zeros(len(zte),dtype=np.int8)
    for a,b in seqs:
        a,b=int(a),min(int(b),len(y)-1)
        if a<len(y): y[a:b+1]=1
    pred=(zte>thr).astype(np.int8)
    rows.append({'channel':chan,'train_n':len(Xtr),'test_n':len(Xte),'dims':len(feats),'anomaly_points':int(y.sum()),'threshold':thr,'pred_points':int(pred.sum()),'tp':int(((pred==1)&(y==1)).sum()),'fp':int(((pred==1)&(y==0)).sum()),'fn':int(((pred==0)&(y==1)).sum()),'tn':int(((pred==0)&(y==0)).sum()),'roc_auc':float(roc_auc_score(y,zte)) if len(np.unique(y))>1 else None,'pr_auc':float(average_precision_score(y,zte)) if y.sum()>0 else None})
res=pd.DataFrame(rows)
TP,FP,FN,TN=[int(res[c].sum()) for c in ['tp','fp','fn','tn']]
precision=TP/(TP+FP) if TP+FP else 0; recall=TP/(TP+FN) if TP+FN else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
# event-level detection: interval hit if at least one predicted point lies inside
hit=total=0; delays=[]
for _,r in labels[labels.spacecraft=='SMAP'].iterrows():
    chan=r.chan_id; tef=DATA/'test'/f'{chan}.parquet';
    if not tef.exists(): continue
    rr=res[res.channel==chan]
    if rr.empty: continue
    te=pd.read_parquet(tef); feats=[c for c in te.columns if c!='timestep']; tr=pd.read_parquet(DATA/'train'/f'{chan}.parquet'); Xtr=tr[feats].to_numpy(float); Xte=te[feats].to_numpy(float); mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0);sd[sd<1e-8]=1; score=np.abs((Xte-mu)/sd).max(1); pred=score>float(rr.threshold.iloc[0])
    seqs=json.loads(r.anomaly_sequences.replace("'",'"')) if isinstance(r.anomaly_sequences,str) else r.anomaly_sequences
    for a,b in seqs:
        total+=1; ix=np.where(pred[int(a):min(int(b)+1,len(pred))])[0]
        if len(ix): hit+=1; delays.append(int(ix[0]))
res.to_csv(OUT/'smap_zscore_channel_results.csv',index=False)
summary={'dataset':'SMAP','method':'per-channel max absolute z-score','channels_evaluated':len(res),'train_points':int(res.train_n.sum()),'test_points':int(res.test_n.sum()),'dimensions':int(res.dims.iloc[0]),'anomaly_points':int(res.anomaly_points.sum()),'threshold_rule':'99.5th percentile of final 20% training scores','pointwise':{'precision':precision,'recall':recall,'f1':f1,'fpr':FP/(FP+TN),'tp':TP,'fp':FP,'fn':FN,'tn':TN,'roc_auc_micro':float(roc_auc_score(np.concatenate([np.zeros(0)]),[])) if False else None},'event_level':{'detected':hit,'total':total,'recall':hit/total if total else 0,'mean_delay_points':float(np.mean(delays)) if delays else None}}
# macro AUC across channels
summary['macro_roc_auc']=float(res.roc_auc.mean()); summary['macro_pr_auc']=float(res.pr_auc.mean())
(OUT/'smap_zscore_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
