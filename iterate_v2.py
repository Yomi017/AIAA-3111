"""Mechanism experiments: direct multi-horizon forecasting and normal subspace.
Only development entities are used to choose the model/configuration.
"""
from experiment_v2 import *

class MultiPredictor(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.gru=nn.GRU(d,32,batch_first=True);self.head=nn.Linear(32,h)
    def forward(self,x):
        _,h=self.gru(x);return self.head(h[-1])

def multi(fit,sel,cal,test,seed,dest,h=16,commands=True):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    d=fit.shape[1] if commands else 1
    def pairs(x):
        return windows(x[:,:d])[:-h],windows(x[L:,:1],h)[:,:,0]
    a,b=pairs(fit);a,b=a[::2],b[::2];v,vy=pairs(sel)
    m=MultiPredictor(d,h).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001);best=np.inf;hist=[]
    def infer(w):
        with torch.no_grad():return np.concatenate([m(torch.from_numpy(w[i:i+512]).to(device)).cpu().numpy() for i in range(0,len(w),512)])
    for ep in range(30):
        m.train();order=rng.permutation(len(a));total=0
        for ix in np.array_split(order,max(1,int(np.ceil(len(a)/128)))):
            opt.zero_grad();loss=((m(torch.from_numpy(a[ix]).to(device))-torch.from_numpy(b[ix]).to(device))**2).mean()
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=loss.item()*len(ix)
        m.eval();vl=float(((infer(v)-vy)**2).mean());hist.append([ep+1,total/len(a),vl])
        if vl<best:best=vl;state=copy.deepcopy(m.state_dict())
    m.load_state_dict(state);m.eval();scores=[]
    for x in [cal,test]:
        # Predict even near the end: forecasts only use past inputs.
        forecasts=infer(windows(x[:,:d])[:-1]);n=len(x)-L
        err=np.zeros(n);counts=np.zeros(n)
        for k in range(h):
            err[k:]+=np.abs(x[L+k:,0]-forecasts[:n-k,k]);counts[k:]+=1
        scores.append(err/counts)
    torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},d=d,h=h,seed=seed,history=hist),dest)
    return tuple(scores)

def extra(args):
    out=OUT/args.run;out.mkdir(exist_ok=True,parents=True)
    labs=pd.read_csv(OUT/args.base/'channel_split.csv')
    labs=labs if args.split=='all' else labs[labs.split==args.split]
    for _,r in labs.iterrows():
        dest=out/r.chan_id;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():continue
        ts=time.time();cols=lambda a:a.drop(columns='timestep').to_numpy(np.float32)
        x=cols(pd.read_parquet(DATA/'data/train'/f'{r.chan_id}.parquet'));xt=cols(pd.read_parquet(DATA/'data/test'/f'{r.chan_id}.parquet'))
        n1,n2=split_boundaries(len(x));mu=x[:n1].mean(0);sd=x[:n1].std(0);sd[sd<.01]=1
        x=(x-mu)/sd;xt=(xt-mu)/sd;fit,sel,cal=x[:n1],x[n1:n2],x[n2:];y=label_array(r,len(xt))[L:]
        scores={}
        if args.models in ['all','multi']:
            for commands in [False,True]:
                name='multi_context' if commands else 'multi_target'
                scores[name]=multi(fit,sel,cal,xt,args.seed,dest/(name+'.pt'),commands=commands)
        if args.models in ['all','subspace']:
            fw=windows(fit[:,:1])[:,:,0][::2];pc=PCA(n_components=.95,svd_solver='full').fit(fw)
            def score(z):
                w=windows(z[:,:1])[:,:,0][1:];return np.mean((w-pc.inverse_transform(pc.transform(w)))**2,axis=1)
            scores['hankel_pca']=(score(cal),score(xt))
        rows=[]
        for name,(v,s) in scores.items():
            assert len(s)==len(y) and np.isfinite(s).all()
            threshold=max(float(np.quantile(v,.995)),1e-6);p=s>threshold
            np.savez_compressed(dest/(name+'.npz'),score=s,calibration=v,label=y,pred=p,threshold=threshold)
            rows.append(dict(dataset=r.spacecraft,channel=r.chan_id,split=r.split,method=name,threshold=threshold,**measure(y,s,p)))
        (dest/'done.json').write_text(json.dumps(dict(metrics=rows,seconds=time.time()-ts),indent=2))
        print(r.chan_id,'extra',round(time.time()-ts,1),flush=True)
    summarize(out)

def dynamic(s):
    """Simplified Telemanom-inspired sequence-level threshold, no ground truth.
    Offline/transductive: uses the entire unlabeled evaluation score sequence.
    No label-based filling, buffering or test-F1 optimization.
    """
    mean=float(np.mean(s));sd=float(np.std(s));best=-np.inf;threshold=mean+12*sd
    for z in np.arange(2.5,12,.5):
        t=mean+z*sd;p=s>t;pr=s[~p];k=int(p.sum());e=len(segments(p))
        if not k or k>=len(s)*.5 or not len(pr):continue
        value=((mean-np.mean(pr))/max(mean,1e-12)+(sd-np.std(pr))/max(sd,1e-12))/(e*e+k)
        if value>best:best=value;threshold=t
    return max(float(threshold),1e-6)

def compare(runs,output,split='development'):
    rows=[]
    for run in runs:
        out=OUT/run
        for f in out.glob('*/done.json'):
            info=json.loads(f.read_text())
            for meta in info['metrics']:
                if meta['split']!=split or meta['method'].endswith('_smooth'):continue
                z=np.load(f.parent/(meta['method']+'.npz'));y=z['label']
                for span in [1,16,64]:
                    s=smooth(z['score'],span);v=smooth(z['calibration'],span)
                    for policy in ['cal995','dynamic']:
                        t=max(float(np.quantile(v,.995)),1e-6) if policy=='cal995' else dynamic(s)
                        row=dict(dataset=meta['dataset'],channel=meta['channel'],split=split,method=meta['method'],span=span,policy=policy,threshold=t,**measure(y,s,s>t));rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(OUT/(output+'_channels.csv'),index=False);agg=[]
    for keys,g in df.groupby(['dataset','method','span','policy']):
        s=g[['tp','fp','fn','tn','hits','events']].sum();p=s.tp/max(1,s.tp+s.fp);r=s.tp/max(1,s.tp+s.fn)
        agg.append(dict(zip(['dataset','method','span','policy'],keys))|dict(f1=2*p*r/max(1e-12,p+r),precision=p,recall=r,fpr=s.fp/max(1,s.fp+s.tn),event_recall=s.hits/max(1,s.events),macro_ap=g.ap.mean()))
    a=pd.DataFrame(agg);a.to_csv(OUT/(output+'.csv'),index=False)
    rank=a.groupby(['method','span','policy']).f1.mean().sort_values(ascending=False)
    print(rank.head(18).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='extra42');p.add_argument('--base',default='dev42');p.add_argument('--split',default='development');p.add_argument('--seed',type=int,default=42);p.add_argument('--models',default='all');p.add_argument('--compare',action='store_true');a=p.parse_args()
    if a.compare:compare([a.base,a.run],a.run+'_comparison',a.split)
    else:extra(a)
