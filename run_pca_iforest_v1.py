import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=Path(r'E:/Class/AIAA 3111/Project'); DATA=ROOT/'data/telemanom/data'; OUT=ROOT/'results_v1'; OUT.mkdir(exist_ok=True)
labels=pd.read_csv(DATA/'../labeled_anomalies.csv')
def eval_dataset(spacecraft):
 rows=[]; event={'pca':{'hit':0,'total':0,'delays':[]},'iforest':{'hit':0,'total':0,'delays':[]}}
 for _,r in labels[labels.spacecraft==spacecraft].iterrows():
  chan=r.chan_id; trf=DATA/'train'/f'{chan}.parquet'; tef=DATA/'test'/f'{chan}.parquet'
  if not trf.exists() or not tef.exists(): continue
  tr=pd.read_parquet(trf); te=pd.read_parquet(tef); feats=[c for c in tr.columns if c!='timestep']; Xtr=tr[feats].to_numpy(float); Xte=te[feats].to_numpy(float); mu=np.nanmean(Xtr,0);sd=np.nanstd(Xtr,0);sd[sd<1e-8]=1; Xtr=(np.nan_to_num(Xtr)-mu)/sd;Xte=(np.nan_to_num(Xte)-mu)/sd; cut=int(.8*len(Xtr)); fit=Xtr[:cut]; val=Xtr[cut:]
  seqs=json.loads(r.anomaly_sequences.replace("'",'"')); y=np.zeros(len(Xte),np.int8)
  for a,b in seqs:
   a,b=int(a),min(int(b),len(y)-1)
   if a<len(y): y[a:b+1]=1
  pca=PCA(n_components=0.95,svd_solver='full',random_state=42);pca.fit(fit); ps=((Xte-pca.inverse_transform(pca.transform(Xte)))**2).mean(1); pv=((val-pca.inverse_transform(pca.transform(val)))**2).mean(1); pt=float(np.quantile(pv,.995)); pp=ps>pt
  sample=fit[np.linspace(0,len(fit)-1,min(20000,len(fit))).astype(int)]; iso=IsolationForest(n_estimators=200,max_samples='auto',random_state=42,n_jobs=-1);iso.fit(sample); iv=-iso.decision_function(val); iscore=-iso.decision_function(Xte);it=float(np.quantile(iv,.995)); ip=iscore>it
  for name,score,pred in [('pca',ps,pp),('iforest',iscore,ip)]:
   tp=int(((pred==1)&(y==1)).sum());fp=int(((pred==1)&(y==0)).sum());fn=int(((pred==0)&(y==1)).sum());tn=int(((pred==0)&(y==0)).sum()); rows.append({'spacecraft':spacecraft,'channel':chan,'method':name,'dims':len(feats),'threshold':float(pt if name=='pca' else it),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'roc_auc':float(roc_auc_score(y,score)),'pr_auc':float(average_precision_score(y,score)),'n_components':int(pca.n_components_) if name=='pca' else None})
   for a,b in seqs:
    event[name]['total']+=1; a=int(a);b=min(int(b),len(pred)-1);ix=np.where(pred[a:b+1])[0]
    if len(ix):event[name]['hit']+=1;event[name]['delays'].append(int(ix[0]))
 out=pd.DataFrame(rows); summaries=[]
 for name in ['pca','iforest']:
  z=out[out.method==name];tp=int(z.tp.sum());fp=int(z.fp.sum());fn=int(z.fn.sum());tn=int(z.tn.sum());P=tp/(tp+fp) if tp+fp else 0;R=tp/(tp+fn) if tp+fn else 0;F=2*P*R/(P+R) if P+R else 0;e=event[name];summaries.append({'dataset':spacecraft,'method':name,'channels':len(z),'pointwise':{'precision':P,'recall':R,'f1':F,'fpr':fp/(fp+tn),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'macro_roc_auc':float(z.roc_auc.mean()),'macro_pr_auc':float(z.pr_auc.mean())},'event_level':{'detected':e['hit'],'total':e['total'],'recall':e['hit']/e['total'],'mean_delay_points':float(np.mean(e['delays'])) if e['delays'] else None}})
 out.to_csv(OUT/f'{spacecraft.lower()}_pca_iforest_channel_results.csv',index=False); (OUT/f'{spacecraft.lower()}_pca_iforest_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf8'); print(json.dumps(summaries,indent=2))
for sp in ['SMAP','MSL']: eval_dataset(sp)
