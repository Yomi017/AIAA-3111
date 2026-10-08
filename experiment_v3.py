"""Cross-channel supervised anomaly detection with nested group-held-out evaluation.

No target-channel anomaly labels enter fitting, hyperparameter or threshold selection.
All 80 channels were explored in v2: this is grouped resampling, NOT a virgin test.
Target-channel NORMAL fit data calibrates physical scales (allowed adaptation).
"""
from pathlib import Path
import argparse,ast,hashlib,json,time,platform
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import joblib
from experiment_v2 import DATA,L,ROOT,split_boundaries,label_array,measure

OUT=ROOT/'results_v3'

def fold_id(channel):return int(hashlib.sha256(('v3:'+channel).encode()).hexdigest(),16)%5

def descriptors(x):
    s=pd.Series(np.asarray(x,dtype=np.float64));diff=s.diff().fillna(0);f={}
    f['level']=s.to_numpy();f['step']=diff.to_numpy();f['abs_step']=diff.abs().to_numpy()
    for w in [8,32,64]:
        roll=s.rolling(w,min_periods=w)
        mean=roll.mean();std=roll.std(ddof=0);lo=roll.min();hi=roll.max()
        f[f'mean_{w}']=mean.to_numpy();f[f'std_{w}']=std.to_numpy();f[f'range_{w}']=(hi-lo).to_numpy()
        f[f'detrended_{w}']=(s-mean).to_numpy();f[f'drift_{w}']=(s-s.shift(w-1)).to_numpy()
        f[f'roughness_{w}']=diff.abs().rolling(w,min_periods=w).mean().to_numpy()
    return pd.DataFrame(f).iloc[L:].reset_index(drop=True)

def features(normal,test):
    # Calibration uses only earliest normal fit segment, not target labels or test distribution.
    n1,_=split_boundaries(len(normal));normal=normal[:n1]
    center=float(np.median(normal));scale=max(float(np.std(normal)),.01)
    a=descriptors((normal-center)/scale);b=descriptors((test-center)/scale)
    med=a.median().to_numpy();spread=np.maximum((a.quantile(.75)-a.quantile(.25)).to_numpy()/1.349,.1)
    z=(b.to_numpy()-med)/spread
    # Signed logarithm bounds scale mismatches without removing anomaly direction.
    raw=np.sign(b.to_numpy())*np.log1p(np.abs(b.to_numpy()))
    relative=np.sign(z)*np.log1p(np.abs(z))
    X=np.concatenate([raw,relative],axis=1).astype(np.float32)
    assert np.isfinite(X).all()
    names=['raw_'+c for c in b]+['normal_relative_'+c for c in b]
    return X,names

def load_channels():
    labs=pd.read_csv(DATA/'labeled_anomalies.csv');labs=labs[labs.chan_id!='P-2'];channels={}
    for _,r in labs.iterrows():
        normal=pd.read_parquet(DATA/'data/train'/f'{r.chan_id}.parquet').value.to_numpy()
        test=pd.read_parquet(DATA/'data/test'/f'{r.chan_id}.parquet').value.to_numpy()
        x,names=features(normal,test);y=label_array(r,len(test))[L:]
        channels[r.chan_id]=dict(X=x,y=y,dataset=r.spacecraft,fold=fold_id(r.chan_id),names=names)
    return channels

def stack(channels,ids,stride=2):
    xx=[];yy=[];ww=[]
    for c in ids:
        z=channels[c];x=z['X'][::stride];y=z['y'][::stride]
        # Each channel has equal total fitting weight, then global classes balanced.
        xx.append(x);yy.append(y);ww.append(np.full(len(y),1/len(y)))
    X=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww)
    for k in [0,1]:w[y==k]*=.5/max(w[y==k].sum(),1e-10)
    w*=len(w)
    return X,y,w

def model(kind,depth=3):
    if kind=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=400,random_state=42))
    return HistGradientBoostingClassifier(max_iter=150,learning_rate=.08,max_leaf_nodes=2**depth,max_depth=depth,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=42)

def fit_model(m,X,y,w):
    if hasattr(m,'steps'):m.fit(X,y,logisticregression__sample_weight=w)
    else:m.fit(X,y,sample_weight=w)
    return m

def metric_aggregate(items):
    rows=[]
    for ds in ['SMAP','MSL']:
        z=[r for r in items if r['dataset']==ds]
        if not z:continue
        v={k:sum(r[k] for r in z) for k in ['tp','fp','fn','tn','events','hits','alerts','hit_alerts']}
        p=v['tp']/max(1,v['tp']+v['fp']);r=v['tp']/max(1,v['tp']+v['fn'])
        rows.append(dict(dataset=ds,channels=len(z),f1=2*p*r/max(1e-12,p+r),precision=p,recall=r,fpr=v['fp']/max(1,v['fp']+v['tn']),event_recall=v['hits']/max(1,v['events']),macro_ap=float(np.mean([r['ap'] for r in z])),**v))
    return rows

def choose_threshold(predictions,channels):
    # Finite predeclared grid on inner validation channels only.
    results=[]
    for t in [.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]:
        rows=[]
        for c,s in predictions.items():
            y=channels[c]['y'];p=s>t
            rows.append(dict(dataset=channels[c]['dataset'],tp=int(((y==1)&p).sum()),fp=int(((y==0)&p).sum()),fn=int(((y==1)&~p).sum()),tn=int(((y==0)&~p).sum())))
        f=[]
        for ds in ['SMAP','MSL']:
            z=[r for r in rows if r['dataset']==ds];tp=sum(r['tp'] for r in z);fp=sum(r['fp'] for r in z);fn=sum(r['fn'] for r in z)
            f.append(2*tp/max(1,2*tp+fp+fn))
        results.append(dict(threshold=t,inner_mean_f1=float(np.mean(f))))
    return max(results,key=lambda r:r['inner_mean_f1']),results

def run(args):
    out=OUT/args.run;out.mkdir(parents=True,exist_ok=True);channels=load_channels();ids=sorted(channels)
    config=dict(task='supervised cross-channel anomaly detection',outer_folds=5,fold_hash='sha256(v3:+channel)%5',inner_validation='(outer+1)%5; other three folds train',fit_stride=2,window=64,depth_grid=[3,5],threshold_grid=[.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95],normal_adaptation='first normal training segment only',features=channels[ids[0]]['names'],test_label_policy='outer target labels used only to report metrics',previous_exposure='v2 inspected all80 official test channels; not independent final evidence',seed=42)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    pd.DataFrame([dict(channel=c,dataset=channels[c]['dataset'],fold=channels[c]['fold']) for c in ids]).to_csv(out/'folds.csv',index=False)
    for outer in range(5):
        target=[c for c in ids if channels[c]['fold']==outer];inner=[c for c in ids if channels[c]['fold']==(outer+1)%5];train=[c for c in ids if c not in target+inner];source=train+inner
        assert not set(source)&set(target) and not set(train)&set(inner)
        for kind in ['logistic','hgb']:
            dest=out/f'fold{outer}_{kind}';dest.mkdir(exist_ok=True)
            if (dest/'done.json').exists():continue
            started=time.time();X,y,w=stack(channels,train);choices=[]
            for depth in ([3,5] if kind=='hgb' else [0]):
                m=fit_model(model(kind,depth),X,y,w);scores={c:m.predict_proba(channels[c]['X'])[:,1] for c in inner};best,grid=choose_threshold(scores,channels);choices.append(dict(depth=depth,**best,grid=grid))
            selected=max(choices,key=lambda r:r['inner_mean_f1']);X,y,w=stack(channels,source);m=fit_model(model(kind,selected['depth']),X,y,w)
            joblib.dump(dict(model=m,threshold=selected['threshold'],config=config,source_channels=source,target_channels=target),dest/'model.joblib',compress=3)
            rows=[]
            for c in target:
                s=m.predict_proba(channels[c]['X'])[:,1];p=s>selected['threshold'];y=channels[c]['y']
                np.savez_compressed(dest/(c+'.npz'),score=s,pred=p,label=y,threshold=selected['threshold'])
                rows.append(dict(channel=c,dataset=channels[c]['dataset'],fold=outer,method=kind,threshold=selected['threshold'],**measure(y,s,p)))
            record=dict(outer=outer,kind=kind,train_inner=train,validation_inner=inner,source_final=source,target=target,selected=selected,choices=choices,seconds=time.time()-started,metrics=rows)
            (dest/'done.json').write_text(json.dumps(record,indent=2));print('fold',outer,kind,'seconds',round(time.time()-started,1),'threshold',selected['threshold'],'depth',selected['depth'],flush=True)
    rows=[]
    for f in out.glob('*/done.json'):rows.extend(json.loads(f.read_text())['metrics'])
    df=pd.DataFrame(rows);df.to_csv(out/'per_channel.csv',index=False);agg=[]
    for name,g in df.groupby('method'):
        for r in metric_aggregate(g.to_dict('records')):agg.append(dict(method=name,**r))
    a=pd.DataFrame(agg);a.to_csv(out/'summary.csv',index=False);print(a[['method','dataset','f1','precision','recall','fpr']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='group5');run(p.parse_args())
