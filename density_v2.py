"""Local normal trajectory density; corrected covariance and rank-zero handling."""
from experiment_v2 import *
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import LedoitWolf
from stable_lof_v2 import StableLOF

def run_density(split,run):
    out=OUT/run;out.mkdir(exist_ok=True,parents=True);labs=pd.read_csv(OUT/'dev42/channel_split.csv');labs=labs[labs.split==split]
    for _,r in labs.iterrows():
        dest=out/r.chan_id;dest.mkdir(exist_ok=True)
        if (dest/'done.json').exists():continue
        rd=lambda p:pd.read_parquet(p).value.to_numpy(np.float64)[:,None]
        x=rd(DATA/'data/train'/f'{r.chan_id}.parquet');xt=rd(DATA/'data/test'/f'{r.chan_id}.parquet');n1,n2=split_boundaries(len(x))
        mu=x[:n1].mean(0);sd=x[:n1].std(0);sd[sd<.01]=1;x=(x-mu)/sd;xt=(xt-mu)/sd
        fw=windows(x[:n1])[:,:,0][::2];v=windows(x[n2:])[:,:,0][1:];t=windows(xt)[:,:,0][1:];y=label_array(r,len(xt))[L:]
        scores={}
        for k in [20,50]:
            model=StableLOF(n_neighbors=k).fit(fw)
            scores['trajectory_lof'+str(k)]=(-model.score_samples(v),-model.score_samples(t))
        # Normal-subspace density: learn a temporal representation before density.
        # Eigenvalue floor prevents constant channels from producing NaNs.
        pc_lat=PCA(n_components=.95,svd_solver='full').fit(fw) if np.var(fw,axis=0).sum()>1e-12 else None
        def latent(w):
            return pc_lat.transform(w)/np.sqrt(np.maximum(pc_lat.explained_variance_,1e-6)) if pc_lat is not None else w-fw.mean(0)
        model=StableLOF(n_neighbors=50).fit(latent(fw))
        scores['subspace_lof50']=(-model.score_samples(latent(v)),-model.score_samples(latent(t)))
        cov=LedoitWolf().fit(fw);e,u=np.linalg.eigh(cov.covariance_);precision=(u/np.maximum(e,1e-6))@u.T
        def gs(w):
            d=w-cov.location_;return np.maximum(np.einsum('ni,ij,nj->n',d,precision,d),0)
        scores['trajectory_gaussian_fixed']=(gs(v),gs(t))
        if np.var(fw,axis=0).sum()<1e-12:
            ps=lambda w:np.mean((w-fw.mean(0))**2,axis=1)
        else:
            pc=PCA(n_components=.95,svd_solver='full').fit(fw)
            ps=lambda w:np.mean((w-pc.inverse_transform(pc.transform(w)))**2,axis=1)
        scores['hankel_pca_fixed']=(ps(v),ps(t));rows=[]
        for name,(a,b) in scores.items():
            assert np.isfinite(a).all() and np.isfinite(b).all();threshold=max(float(np.quantile(a,.995)),1e-6)
            np.savez_compressed(dest/(name+'.npz'),score=b,calibration=a,label=y,pred=b>threshold,threshold=threshold)
            rows.append(dict(dataset=r.spacecraft,channel=r.chan_id,split=split,method=name,threshold=threshold,**measure(y,b,b>threshold)))
        (dest/'done.json').write_text(json.dumps(dict(metrics=rows),indent=2));print(r.chan_id,'density',flush=True)
    summarize(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='density');p.add_argument('--split',default='development');a=p.parse_args();run_density(a.split,a.run)
