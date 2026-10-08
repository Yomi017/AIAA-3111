"""Predeclared mechanism study; development, freeze, then one confirmation pass.

No anomaly labels enter fit_model/predict. Earlier exposure prevents virgin-test claims.
All candidates are evaluated after freezing as mechanism ablations, not a test-time menu.
"""
from pathlib import Path
import argparse, json, hashlib, datetime, time
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from experiment_v2 import ROOT, DATA, L, windows, split_boundaries, label_array, measure
from stable_lof_v2 import StableLOF

METHODS=['subspace_reference','normal_envelope','level_conditional','robust_trajectory']
OUT=ROOT/'results_v4'
CONFIG=dict(window=64,fit_stride=2,pca_variance=.95,whitening_floor=1e-6,neighbors=50,quantile=.995,smoothing=1,normal_split='60/20/20; last two >=96',support='t=64..n-1',point_adjustment=False,level_bins='normal fit quartiles; duplicate boundaries removed',conditional_min_calibration=30,robust_features=['median','IQR','last','last-minus-first','median_abs_difference','q90_abs_difference','OLS_slope','mean_second_half-minus-first_half'],feature_scale='fit median and IQR/1.349 floor 0.1',selection='equal dataset weighted development micro F1; ties retain reference',confirmation_policy='all four frozen mechanism arms once; no reselection',history='all official channels previously exposed, including subspace grid confirmation scores')

def win(x):return windows(np.asarray(x)[:,None])[:,:,0][1:]
def descriptors(w):
    diff=np.abs(np.diff(w,axis=1));axis=np.arange(L)-(L-1)/2
    return np.column_stack([np.median(w,1),np.quantile(w,.75,axis=1)-np.quantile(w,.25,axis=1),w[:,-1],w[:,-1]-w[:,0],np.median(diff,1),np.quantile(diff,.9,axis=1),w@axis/(axis@axis),w[:,L//2:].mean(1)-w[:,:L//2].mean(1)])

def represent(m,w,kind):
    if kind=='robust_trajectory':return (descriptors(w)-m['feature_center'])/m['feature_scale']
    pc=m['pca'];return pc.transform(w)/np.sqrt(np.maximum(pc.explained_variance_,1e-6)) if pc is not None else w-m['window_mean']

def fit_model(x):
    x=np.asarray(x,dtype=np.float64);a,b=split_boundaries(len(x));mu=x[:a].mean();sd=x[:a].std();sd=sd if sd>=.01 else 1.
    z=(x-mu)/sd;fw=windows(z[:a,None])[:,:,0][::2]
    pc=PCA(n_components=.95,svd_solver='full').fit(fw) if np.var(fw,axis=0).sum()>1e-12 else None
    feat=descriptors(fw);center=np.median(feat,axis=0);spread=np.maximum((np.quantile(feat,.75,axis=0)-np.quantile(feat,.25,axis=0))/1.349,.1)
    m=dict(mu=mu,sd=sd,pca=pc,window_mean=fw.mean(0),feature_center=center,feature_scale=spread,n1=a,n2=b,config=CONFIG)
    m['subspace']=StableLOF(50).fit(represent(m,fw,'subspace_reference'))
    m['robust']=StableLOF(50).fit(represent(m,fw,'robust_trajectory'))
    cw=win(z[b:]);sw=win(z[a:b]);cs=-m['subspace'].score_samples(represent(m,cw,'subspace_reference'));ss=-m['subspace'].score_samples(represent(m,sw,'subspace_reference'))
    ref=max(float(np.quantile(cs,.995)),1e-6);envelope=max(ref,float(np.quantile(ss,.995)))
    edges=np.unique(np.quantile(np.median(fw,axis=1),[.25,.5,.75]));bins=np.searchsorted(edges,np.median(cw,axis=1),side='right')
    local=[];counts=[]
    for k in range(len(edges)+1):
        vals=cs[bins==k];counts.append(len(vals));local.append(max(float(np.quantile(vals,.995)),1e-6) if len(vals)>=30 else ref)
    robust_cal=-m['robust'].score_samples(represent(m,cw,'robust_trajectory'))
    m.update(thresholds=dict(subspace_reference=ref,normal_envelope=envelope,robust_trajectory=max(float(np.quantile(robust_cal,.995)),1e-6)),edges=edges,local_thresholds=np.array(local),local_counts=counts)
    m['normal_diagnostics']=dict(calibration_n=len(cs),reference_threshold=ref,selection_threshold=float(np.quantile(ss,.995)),envelope_ratio=envelope/ref,selection_exceedance=float(np.mean(ss>ref)),calibration_exceedance=float(np.mean(cs>ref)),pca_rank=int(pc.n_components_) if pc else 0,reference_floor=m['subspace'].floor)
    return m

def predict(m,x,method):
    if method not in METHODS:raise ValueError(method)
    if len(x)<=L:raise ValueError('Need more than 64 samples')
    w=win((np.asarray(x,dtype=np.float64)-m['mu'])/m['sd'])
    learner=m['robust'] if method=='robust_trajectory' else m['subspace']
    score=-learner.score_samples(represent(m,w,method))
    threshold=m['local_thresholds'][np.searchsorted(m['edges'],np.median(w,axis=1),side='right')] if method=='level_conditional' else np.full(len(score),m['thresholds'][method])
    return score,threshold,score>threshold

def aggregate(rows):
    df=pd.DataFrame(rows);result=[]
    for (ds,method),g in df.groupby(['dataset','method']):
        c=g[['tp','fp','fn','tn','events','hits','alerts','hit_alerts','delay_sum']].sum().to_dict()
        result.append(dict(dataset=ds,method=method,channels=len(g),f1=2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn']),precision=c['tp']/max(1,c['tp']+c['fp']),recall=c['tp']/max(1,c['tp']+c['fn']),fpr=c['fp']/max(1,c['fp']+c['tn']),event_recall=c['hits']/max(1,c['events']),macro_ap=g.ap.mean(),median_channel_f1=float(np.median(2*g.tp/np.maximum(1,2*g.tp+g.fp+g.fn))),delay_detected=c['delay_sum']/max(1,c['hits']),**c))
    return pd.DataFrame(result)

def initialize(out):
    out.mkdir(parents=True,exist_ok=True);f=out/'predeclared_config.json'
    if f.exists():assert json.loads(f.read_text())['config']==CONFIG
    else:f.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=CONFIG,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

def run(stage,out):
    initialize(out)
    if stage=='confirmation':
        frozen=json.loads((out/'frozen_selection.json').read_text());assert frozen['config']==CONFIG
        assert frozen['code_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Code changed since freeze'
        lock=out/'confirmation_started.json'
        if lock.exists():raise RuntimeError('Confirmation already started. Preserve it; use a distinct output directory for a declared replication.')
        lock.write_text(json.dumps(dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_sha256=hashlib.sha256((out/'frozen_selection.json').read_bytes()).hexdigest()),indent=2))
    labs=pd.read_csv(OUT/'data_inventory.csv');labs=labs[labs.split==stage];labels=pd.read_csv(DATA/'labeled_anomalies.csv').set_index('chan_id')
    assert len(labs)==(22 if stage=='development' else 58)
    rows=[];normal=[]
    for _,r in labs.iterrows():
        started=time.time();c=r.channel;dest=out/stage/c;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():
            rows.extend(json.loads((dest/'metrics.json').read_text()));normal.append(json.loads((dest/'normal.json').read_text()));continue
        x=pd.read_parquet(DATA/'data/train'/f'{c}.parquet').value.to_numpy(np.float64)
        m=fit_model(x);joblib.dump(m,dest/'model.joblib',compress=3)
        xt=pd.read_parquet(DATA/'data/test'/f'{c}.parquet').value.to_numpy(np.float64)
        # Labels loaded only after fitting; never passed to fitting or prediction.
        y=label_array(labels.loc[c],len(xt))[L:];channel_rows=[]
        for name in METHODS:
            s,t,p=predict(m,xt,name);assert len(s)==len(y) and np.isfinite(s).all() and np.isfinite(t).all()
            np.savez_compressed(dest/(name+'.npz'),score=s,threshold=t,pred=p,label=y)
            channel_rows.append(dict(channel=c,dataset=r.dataset,split=stage,method=name,**measure(y,s,p)))
        info=dict(channel=c,dataset=r.dataset,split=stage,**m['normal_diagnostics']);normal.append(info);(dest/'normal.json').write_text(json.dumps(info,indent=2));(dest/'metrics.json').write_text(json.dumps(channel_rows,indent=2));rows.extend(channel_rows)
        print(stage,c,round(time.time()-started,1),flush=True)
    pd.DataFrame(rows).to_csv(out/(stage+'_channels.csv'),index=False);pd.DataFrame(normal).to_csv(out/(stage+'_normal.csv'),index=False);s=aggregate(rows);s.to_csv(out/(stage+'_summary.csv'),index=False);print(s[['dataset','method','f1','fpr','event_recall']].to_string(index=False))

def freeze(out):
    initialize(out);f=out/'frozen_selection.json'
    if f.exists():raise RuntimeError('Selection already frozen; do not overwrite')
    df=pd.read_csv(out/'development_summary.csv');assert len(df)==8
    rank=df.groupby('method').f1.mean().sort_values(ascending=False);chosen=rank.index[0]
    f.write_text(json.dumps(dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=chosen,development_mean_f1=rank.to_dict(),config=CONFIG,development_csv_sha256=hashlib.sha256((out/'development_channels.csv').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(f.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','freeze','confirmation','predict']);p.add_argument('--out',default=str(OUT/'mechanism'));p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');p.add_argument('--method',choices=METHODS,default='subspace_reference');a=p.parse_args()
    if a.stage=='freeze':freeze(Path(a.out))
    elif a.stage=='predict':
        from detector_v2 import series
        m=joblib.load(a.model);x=series(a.input);s,t,b=predict(m,x,a.method);pd.DataFrame(dict(timestep=np.arange(L,len(x)),value=x[L:],score=s,threshold=t,anomaly=b.astype(int))).to_csv(a.output,index=False)
    else:run(a.stage,Path(a.out))
