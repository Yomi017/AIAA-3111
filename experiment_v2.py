"""Channel-isolated, train-only models; frozen calibration; dense causal scores.

Official test labels are only consumed by evaluation, never by model training.
P-2 is excluded due to contradictory duplicate annotations (OmniAnomaly convention).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import argparse, ast, hashlib, json, time, copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/telemanom'
OUT = ROOT/'results_v2'
L = 64
torch.set_num_threads(4)

def split_boundaries(n):
    # Preserve useful normal calibration on short channels; uses lengths only.
    n1=min(int(n*.6),n-192);n2=min(int(n*.8),n-96)
    assert n1>L+16 and n2-n1>=96 and n-n2>=96
    return n1,n2

def smooth(a, span=16):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()

def windows(x, length=L):
    return np.lib.stride_tricks.sliding_window_view(x, length, axis=0).transpose(0,2,1).copy()

def label_array(row, n):
    assert n == int(row.num_values), (row.chan_id,n,row.num_values)
    y=np.zeros(n,dtype=np.int8)
    for a,b in ast.literal_eval(row.anomaly_sequences):
        assert 0 <= a <= b < n
        y[a:b+1]=1
    return y

def segments(y):
    edges=np.diff(np.r_[0,y.astype(int),0]); return list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]-1))

def measure(y,s,p):
    tp=int(np.sum((y==1)&p));fp=int(np.sum((y==0)&p));fn=int(np.sum((y==1)&~p));tn=int(np.sum((y==0)&~p))
    ev=segments(y); delays=[int(np.flatnonzero(p[a:b+1])[0]) for a,b in ev if p[a:b+1].any()]
    alerts=segments(p)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,events=len(ev),hits=len(delays),delay_sum=sum(delays),
                alerts=len(alerts),hit_alerts=sum(bool(y[a:b+1].any()) for a,b in alerts),
                auroc=float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None,
                ap=float(average_precision_score(y,s)) if y.any() else None)

class Predictor(nn.Module):
    def __init__(self,d):
        super().__init__(); self.gru=nn.GRU(d,32,batch_first=True); self.head=nn.Linear(32,1)
    def forward(self,x):
        _,h=self.gru(x);return self.head(h[-1]).squeeze(-1)

def neural(train, select, cal, test, seed, output, use_commands=True, epochs=20):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    d=train.shape[1] if use_commands else 1
    def pairs(x):
        w=windows(x[:,:d])[:-1]; return w,x[L:,0].copy()
    a,b=pairs(train); a,b=a[::2],b[::2]
    v,vy=pairs(select)
    m=Predictor(d).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    lossfn=nn.MSELoss(); best=float('inf');state=None;history=[]
    def infer(w):
        with torch.no_grad():
            return np.concatenate([m(torch.from_numpy(w[i:i+512]).to(dev)).cpu().numpy() for i in range(0,len(w),512)])
    for ep in range(epochs):
        m.train();order=rng.permutation(len(a));ls=0
        for idx in np.array_split(order,max(1,int(np.ceil(len(a)/128)))):
            xx=torch.from_numpy(a[idx]).to(dev);yy=torch.from_numpy(b[idx]).to(dev)
            opt.zero_grad();loss=lossfn(m(xx),yy);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();ls+=loss.item()*len(idx)
        m.eval();vl=float(np.mean((infer(v)-vy)**2)); history.append([ep+1,ls/len(a),vl])
        if vl<best:best=vl;state=copy.deepcopy(m.state_dict())
    m.load_state_dict(state);m.eval()
    preds=[]
    for x in [cal,test]:
        w,y=pairs(x);pred=infer(w);preds.append((np.abs(y-pred),pred))
    torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},d=d,seed=seed,history=history),output)
    return preds,history

def run(args):
    out=OUT/args.run;out.mkdir(parents=True,exist_ok=True)
    labs=pd.read_csv(DATA/'labeled_anomalies.csv')
    labs=labs[labs.chan_id!='P-2'].copy()
    # Hash split determined before running new models. Earlier v0/v1 inspected all tests:
    # final split is a prospective held-out evaluation for v2, not virgin data.
    labs['split']=[('development' if int(hashlib.sha256(c.encode()).hexdigest(),16)%3==0 else 'confirmation') for c in labs.chan_id]
    labs.to_csv(out/'channel_split.csv',index=False)
    config=dict(seed=args.seed,L=L,epochs=args.epochs,train_fractions=[.6,.2,.2],quantile=.995,
      warmup=L,score_smoothing=16,point_adjustment=False,excluded=['P-2'],
      reason='P-2 duplicate conflicting labels; no entity concatenation; inputs are target plus command indicators')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for _,row in labs.iterrows():
        c=row.chan_id
        if args.split!='all' and row.split!=args.split:continue
        dest=out/c;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():continue
        t=time.time();tr=pd.read_parquet(DATA/'data/train'/f'{c}.parquet');te=pd.read_parquet(DATA/'data/test'/f'{c}.parquet')
        assert np.array_equal(tr.timestep,np.arange(len(tr))) and np.array_equal(te.timestep,np.arange(len(te)))
        cols=[x for x in tr if x!='timestep'];assert cols[0]=='value' and cols==[x for x in te if x!='timestep']
        x=tr[cols].to_numpy(np.float32);xt=te[cols].to_numpy(np.float32)
        assert np.isfinite(x).all() and np.isfinite(xt).all()
        n1,n2=split_boundaries(len(x))
        mu=x[:n1].mean(0);sd=x[:n1].std(0);sd[sd<.01]=1
        x=(x-mu)/sd;xt=(xt-mu)/sd
        fit,sel,cal=x[:n1],x[n1:n2],x[n2:]
        assert min(map(len,[fit,sel,cal,xt]))>L
        y=label_array(row,len(xt))[L:]
        scores={}
        if args.mode in ['all','classical']:
            scores['zmax']=(np.abs(cal).max(1)[L:],np.abs(xt).max(1)[L:])
            scores['ztarget']=(np.abs(cal[:,0])[L:],np.abs(xt[:,0])[L:])
            active=np.std(fit,0)>1e-6
            if active.any():
                pc=PCA(n_components=.95,svd_solver='full').fit(fit[:,active])
                def pe(a):
                    a=a[:,active];return np.mean((a-pc.inverse_transform(pc.transform(a)))**2,1)[L:]
                scores['pca']=(pe(cal),pe(xt))
            else:scores['pca']=(np.mean(cal**2,1)[L:],np.mean(xt**2,1)[L:])
            iso=IsolationForest(n_estimators=100,random_state=args.seed,n_jobs=4).fit(fit)
            scores['iforest']=(-iso.score_samples(cal)[L:],-iso.score_samples(xt)[L:])
            # Distance to closest normal subsequence, an AB-join/discord-style baseline.
            fw=windows(fit[:,:1])[:,:,0][::4]
            nearest=NearestNeighbors(n_neighbors=1,n_jobs=4).fit(fw)
            def knn(a):return nearest.kneighbors(windows(a[:,:1])[:,:,0][1:],return_distance=True)[0][:,0]/np.sqrt(L)
            scores['subsequence_knn']=(knn(cal),knn(xt))
        if args.mode in ['all','neural']:
            for commands in [False,True]:
                name='gru_context' if commands else 'gru_target'
                pairs,hist=neural(fit,sel,cal,xt,args.seed,dest/(name+'.pt'),commands,args.epochs)
                scores[name]=(pairs[0][0],pairs[1][0]);np.save(dest/(name+'_prediction.npy'),pairs[1][1])
        np.savez_compressed(dest/'normalization.npz',mean=mu,std=sd)
        results=[]
        for method,(v,s) in scores.items():
            assert len(s)==len(y) and np.isfinite(s).all() and np.isfinite(v).all()
            for span in [1,16]:
                vv=smooth(v,span);ss=smooth(s,span)
                threshold=max(float(np.quantile(vv,.995)),1e-6)
                pred=ss>threshold;name=method+('_smooth' if span>1 else '')
                np.savez_compressed(dest/(name+'.npz'),score=ss,calibration=vv,label=y,pred=pred,threshold=threshold)
                results.append(dict(dataset=row.spacecraft,channel=c,split=row.split,method=name,threshold=threshold,**measure(y,ss,pred)))
        (dest/'done.json').write_text(json.dumps(dict(metrics=results,train_n=len(x),test_n=len(xt),d=len(cols),seconds=time.time()-t),indent=2))
        print(c,row.spacecraft,row.split,'seconds',round(time.time()-t,1),flush=True)
    summarize(out)

def summarize(out):
    rows=[]
    for f in out.glob('*/done.json'):rows.extend(json.loads(f.read_text())['metrics'])
    df=pd.DataFrame(rows);df.to_csv(out/'per_channel.csv',index=False);agg=[]
    for (ds,split,method),z in df.groupby(['dataset','split','method']):
        s=z[['tp','fp','fn','tn','hits','events','alerts','hit_alerts','delay_sum']].sum().to_dict()
        p=s['tp']/max(1,s['tp']+s['fp']);r=s['tp']/max(1,s['tp']+s['fn'])
        agg.append(dict(dataset=ds,split=split,method=method,channels=len(z),precision=p,recall=r,f1=2*p*r/max(1e-15,p+r),fpr=s['fp']/max(1,s['fp']+s['tn']),event_recall=s['hits']/max(1,s['events']),alert_precision=s['hit_alerts']/max(1,s['alerts']),delay_detected=s['delay_sum']/s['hits'] if s['hits'] else None,macro_ap=z.ap.mean(),**s))
    pd.DataFrame(agg).to_csv(out/'summary.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='pilot');p.add_argument('--split',default='development');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=20);p.add_argument('--mode',default='all');a=p.parse_args();run(a)
