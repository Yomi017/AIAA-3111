"""Normal-trajectory models; window scores assigned to their right endpoint.
USAD-inspired adversarial training adapted to standardized unbounded target inputs.
"""
from experiment_v2 import *
from sklearn.covariance import LedoitWolf

class WindowAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(L,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        def dec():return nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,32),nn.ReLU(),nn.Linear(32,L))
        self.decoder1=dec();self.decoder2=dec()
    def forward(self,x):return self.decoder1(self.encoder(x))

def ae(fit,sel,cal,test,seed,path,kind):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    w=lambda z:windows(z[:,:1])[:,:,0]
    a=torch.from_numpy(w(fit)[::2]).to(device);v=torch.from_numpy(w(sel)).to(device)
    m=WindowAE().to(device);opt1=torch.optim.Adam(list(m.encoder.parameters())+list(m.decoder1.parameters()),lr=.001)
    opt2=torch.optim.Adam(list(m.encoder.parameters())+list(m.decoder2.parameters()),lr=.001);best=np.inf;hist=[]
    for epoch in range(1,51):
        m.train();loss_sum=0
        for ix in np.array_split(rng.permutation(len(a)),max(1,int(np.ceil(len(a)/128)))):
            x=a[ix];m.zero_grad(set_to_none=True)
            if kind=='dae':
                corrupted=x+.1*torch.randn_like(x);corrupted=corrupted.masked_fill(torch.rand_like(x)<.15,0)
                loss=((m(corrupted)-x)**2).mean()
            else:
                w1=m(x);w3=m.decoder2(m.encoder(w1))
                loss=((w1-x)**2).mean()/epoch+(1-1/epoch)*((w3-x)**2).mean()
            loss.backward();opt1.step();loss_sum+=loss.item()*len(ix)
            if kind=='usad':
                m.zero_grad(set_to_none=True);w1=m(x);w2=m.decoder2(m.encoder(x));w3=m.decoder2(m.encoder(w1))
                loss2=((w2-x)**2).mean()/epoch-(1-1/epoch)*((w3-x)**2).mean();loss2.backward();opt2.step()
        m.eval()
        with torch.no_grad():vl=float(((m(v)-v)**2).mean().item())
        hist.append([epoch,loss_sum/len(a),vl])
        if vl<best:best=vl;state=copy.deepcopy(m.state_dict())
    m.load_state_dict(state);m.eval();res=[]
    with torch.no_grad():
        for z in [cal,test]:
            inp=w(z)[1:];parts=[]
            for start in range(0,len(inp),512):
                xx=torch.from_numpy(inp[start:start+512]).to(device);p=m(xx);s=((p-xx)**2).mean(1)
                if kind=='usad':s=.5*s+.5*((m.decoder2(m.encoder(p))-xx)**2).mean(1)
                parts.append(s.cpu().numpy())
            res.append(np.concatenate(parts))
    torch.save(dict(state_dict={k:v.cpu() for k,v in state.items()},seed=seed,kind=kind,history=hist),path)
    return tuple(res)

def run_extra(args):
    out=OUT/args.run;out.mkdir(exist_ok=True,parents=True);labs=pd.read_csv(OUT/'dev42/channel_split.csv');labs=labs[labs.split==args.split]
    for _,r in labs.iterrows():
        dest=out/r.chan_id;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():continue
        start=time.time();rd=lambda p:pd.read_parquet(p).drop(columns='timestep').to_numpy(np.float32)
        x=rd(DATA/'data/train'/f'{r.chan_id}.parquet');xt=rd(DATA/'data/test'/f'{r.chan_id}.parquet');n1,n2=split_boundaries(len(x))
        mu=x[:n1].mean(0);sd=x[:n1].std(0);sd[sd<.01]=1;x=(x-mu)/sd;xt=(xt-mu)/sd;fit,sel,cal=x[:n1],x[n1:n2],x[n2:];y=label_array(r,len(xt))[L:]
        scores={}
        for kind in ['dae','usad']:scores[kind]=ae(fit,sel,cal,xt,args.seed,dest/(kind+'.pt'),kind)
        fw=windows(fit[:,:1])[:,:,0][::2];cov=LedoitWolf().fit(fw)
        # Constant channels need a variance floor, not a zero pseudoinverse.
        eig,u=np.linalg.eigh(cov.covariance_);precision=(u/np.maximum(eig,1e-6))@u.T
        def score(z):
            delta=windows(z[:,:1])[:,:,0][1:]-cov.location_
            return np.maximum(np.einsum('ni,ij,nj->n',delta,precision,delta),0)
        scores['trajectory_gaussian']=(score(cal),score(xt))
        rows=[]
        for name,(v,s) in scores.items():
            assert len(s)==len(y) and np.isfinite(s).all();t=max(float(np.quantile(v,.995)),1e-6);p=s>t
            np.savez_compressed(dest/(name+'.npz'),score=s,calibration=v,label=y,pred=p,threshold=t)
            rows.append(dict(dataset=r.spacecraft,channel=r.chan_id,split=r.split,method=name,threshold=t,**measure(y,s,p)))
        (dest/'done.json').write_text(json.dumps(dict(metrics=rows,seconds=time.time()-start),indent=2));print(r.chan_id,'recon',round(time.time()-start,1),flush=True)
    summarize(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='recon42');p.add_argument('--split',default='development');p.add_argument('--seed',type=int,default=42);run_extra(p.parse_args())
