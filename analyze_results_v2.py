"""Fixed comparisons, paired channel bootstrap, same-smoothing ablations, provenance."""
from experiment_v2 import *
import platform,sys,sklearn,requests

def f1_from_counts(a):
    return 2*a[...,0]/np.maximum(1,2*a[...,0]+a[...,1]+a[...,2])

def main():
    df=pd.read_csv(OUT/'confirmation_frozen_channels.csv');summary=pd.read_csv(OUT/'confirmation_frozen.csv');rng=np.random.default_rng(20261006)
    all80=pd.concat([pd.read_csv(OUT/'development_frozen_channels.csv'),df]);all_counts=all80.groupby(['dataset','method'])[['tp','fp','fn']].sum()
    all_counts['f1']=f1_from_counts(all_counts[['tp','fp','fn']].to_numpy());all_counts.reset_index().to_csv(OUT/'all80_descriptive.csv',index=False)
    baseline=['zmax','ztarget','pca','iforest','subsequence_knn','gru_target','gru_context']
    best=summary[summary.method.isin(baseline)].groupby('method').f1.mean().idxmax();comparisons=[]
    for competitor in [best,'dae','hankel_pca_fixed','trajectory_lof50']:
        changes=[];observed=[]
        for ds in ['SMAP','MSL']:
            a=df[(df.dataset==ds)&(df.method=='subspace_lof50')].set_index('channel').sort_index()
            b=df[(df.dataset==ds)&(df.method==competitor)].set_index('channel').reindex(a.index)
            ca=a[['tp','fp','fn']].to_numpy();cb=b[['tp','fp','fn']].to_numpy();ix=rng.integers(0,len(a),(5000,len(a)))
            delta=f1_from_counts(ca[ix].sum(1))-f1_from_counts(cb[ix].sum(1));obs=float(f1_from_counts(ca.sum(0))-f1_from_counts(cb.sum(0)))
            changes.append(delta);observed.append(obs);lo,hi=np.quantile(delta,[.025,.975])
            comparisons.append(dict(candidate='subspace_lof50',comparator=competitor,dataset=ds,delta_f1=obs,ci_low=lo,ci_high=hi,bootstrap_n=5000))
        delta=np.mean(changes,0);lo,hi=np.quantile(delta,[.025,.975]);comparisons.append(dict(candidate='subspace_lof50',comparator=competitor,dataset='equal_weight_mean',delta_f1=np.mean(observed),ci_low=lo,ci_high=hi,bootstrap_n=5000))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_bootstrap.csv',index=False)
    # Deliberately descriptive: candidates were screened, these CIs do not correct model-selection bias.
    ablation=[]
    for split,run in [('development','stable_dev'),('confirmation','stable_confirm')]:
        for f in (OUT/run).glob('*/done.json'):
            for r in json.loads(f.read_text())['metrics']:
                if r['method'] not in ['trajectory_lof50','subspace_lof50']:continue
                z=np.load(f.parent/(r['method']+'.npz'))
                for span in [1,64]:
                    s=smooth(z['score'],span);v=smooth(z['calibration'],span);t=max(float(np.quantile(v,.995)),1e-6)
                    ablation.append(dict(dataset=r['dataset'],split=split,method=r['method'],span=span,**measure(z['label'],s,s>t)))
    a=pd.DataFrame(ablation);a.to_csv(OUT/'ablation_channels.csv',index=False)
    agg=[]
    for keys,g in a.groupby(['dataset','split','method','span']):
        counts=g[['tp','fp','fn']].sum().to_numpy();agg.append(dict(zip(['dataset','split','method','span'],keys))|dict(f1=float(f1_from_counts(counts))))
    pd.DataFrame(agg).to_csv(OUT/'ablation.csv',index=False)
    # Evaluate fixed GRU configurations at additional seeds; no re-selection.
    seed_rows=[];cfg={r['method']:r for r in json.loads((OUT/'frozen_selection.json').read_text())['models']}
    for seed in [42,7,2026]:
        for f in (OUT/('confirm'+str(seed))).glob('*/done.json'):
            for r in json.loads(f.read_text())['metrics']:
                if r['method'] not in ['gru_target','gru_context']:continue
                z=np.load(f.parent/(r['method']+'.npz'));span=int(cfg[r['method']]['span']);s=smooth(z['score'],span);v=smooth(z['calibration'],span);t=max(float(np.quantile(v,.995)),1e-6)
                seed_rows.append(dict(dataset=r['dataset'],method=r['method'],seed=seed,**measure(z['label'],s,s>t)))
    sr=pd.DataFrame(seed_rows);agg=[]
    for keys,g in sr.groupby(['dataset','method','seed']):agg.append(dict(zip(['dataset','method','seed'],keys))|dict(channels=len(g),f1=float(f1_from_counts(g[['tp','fp','fn']].sum().to_numpy()))))
    pd.DataFrame(agg).to_csv(OUT/'seed_robustness.csv',index=False)
    metadata=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    try:metadata['huggingface_revision']=requests.get('https://huggingface.co/api/datasets/appleparan/telemanom',timeout=30).json().get('sha')
    except Exception as e:metadata['hf_revision_error']=str(e)
    files=list((DATA/'data').rglob('*.parquet'))+[DATA/'labeled_anomalies.csv']+list(ROOT.glob('*v2.py'))+list((ROOT/'research_v2').glob('*'))
    metadata['sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2))
    print('Strongest reference baseline by confirmation equal-weight F1:',best)
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(pd.DataFrame(agg).to_string(index=False))

if __name__=='__main__':main()
