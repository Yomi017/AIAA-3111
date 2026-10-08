"""Inspectable channel analysis, stratified paired bootstrap and honest case selection."""
from pathlib import Path
import json,hashlib,sys,platform,importlib.metadata as md
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_v2 import ROOT,DATA,L,segments
from experiment_v4 import OUT,METHODS,aggregate

RUN=OUT/'mechanism';FIG=OUT/'figures'
DISPLAY=dict(subspace_reference='PCA + local density',normal_envelope='Normal envelope',level_conditional='Level conditional (selected)',robust_trajectory='Robust descriptors')

def f1(c):return 2*c[...,0]/np.maximum(1,2*c[...,0]+c[...,1]+c[...,2])
def main():
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    df=pd.concat([pd.read_csv(RUN/f'{s}_channels.csv') for s in ['development','confirmation']],ignore_index=True)
    df['channel_f1']=f1(df[['tp','fp','fn']].to_numpy());df['channel_fpr']=df.fp/np.maximum(1,df.fp+df.tn)
    df.to_csv(OUT/'all_channel_results.csv',index=False)
    inv=pd.read_csv(OUT/'data_inventory.csv');normal=pd.concat([pd.read_csv(RUN/f'{s}_normal.csv') for s in ['development','confirmation']])
    diagnosis=[]
    for _,r in inv.iterrows():
        c=r.channel;tr=pd.read_parquet(DATA/'data/train'/f'{c}.parquet');te=pd.read_parquet(DATA/'data/test'/f'{c}.parquet');fit=tr.value.to_numpy()[:r.fit_n]
        z=np.load(RUN/r.split/c/'subspace_reference.npz');y=z['label'].astype(bool);pred=z['pred'];v=te.value.to_numpy()[L:]
        lo,hi=np.quantile(fit,[.001,.999]);outside=(v<lo)|(v>hi);is_normal=~y
        commands=te.drop(columns=['timestep','value']).to_numpy();change=np.r_[False,np.any(commands[1:]!=commands[:-1],axis=1)]
        recent=pd.Series(change.astype(int)).rolling(L,min_periods=1).max().to_numpy()[L:]>0
        def rate(mask):return float(pred[mask].mean()) if mask.any() else None
        row=dict(channel=c,dataset=r.dataset,split=r.split,normal_points=int(is_normal.sum()),false_positives=int((pred&is_normal).sum()),fpr=rate(is_normal),outside_fit_normal_fraction=float(outside[is_normal].mean()),outside_fit_fpr=rate(is_normal&outside),inside_fit_fpr=rate(is_normal&~outside),outside_normal_n=int((is_normal&outside).sum()),inside_normal_n=int((is_normal&~outside).sum()),normal_recent_command_fraction=float(recent[is_normal].mean()),command_recent_fpr=rate(is_normal&recent),no_recent_command_fpr=rate(is_normal&~recent),command_recent_n=int((is_normal&recent).sum()),no_recent_command_n=int((is_normal&~recent).sum()),fit_value_std=float(fit.std()),normal_test_value_std=float(v[is_normal].std()),fit_low=float(lo),fit_high=float(hi))
        diagnosis.append(row)
    d=pd.DataFrame(diagnosis).merge(normal,on=['channel','dataset','split']);d.to_csv(OUT/'regime_diagnostics.csv',index=False)
    rng=np.random.default_rng(20261006);boot=[];stability=[]
    conf=df[df.split=='confirmation']
    for method in METHODS[1:]:
        deltas=[];obs=[]
        for ds in ['SMAP','MSL']:
            a=conf[(conf.dataset==ds)&(conf.method==method)].set_index('channel').sort_index();b=conf[(conf.dataset==ds)&(conf.method==METHODS[0])].set_index('channel').reindex(a.index)
            ca=a[['tp','fp','fn']].to_numpy();cb=b[['tp','fp','fn']].to_numpy();ix=rng.integers(len(a),size=(5000,len(a)));delta=f1(ca[ix].sum(1))-f1(cb[ix].sum(1));change=float(f1(ca.sum(0))-f1(cb.sum(0)));lo,hi=np.quantile(delta,[.025,.975]);deltas.append(delta);obs.append(change)
            boot.append(dict(method=method,comparator=METHODS[0],dataset=ds,delta_f1=change,ci_low=lo,ci_high=hi,n_bootstrap=5000))
            per=f1(ca)-f1(cb);loo=[]
            for k in range(len(a)):loo.append(float(f1(ca.sum(0)-ca[k])-f1(cb.sum(0)-cb[k])))
            stability.append(dict(method=method,dataset=ds,channel_wins=int((per>1e-12).sum()),channel_losses=int((per< -1e-12).sum()),channel_ties=int((np.abs(per)<=1e-12).sum()),median_delta=float(np.median(per)),leave_one_channel_out_min=min(loo),leave_one_channel_out_max=max(loo)))
        delta=np.mean(deltas,axis=0);lo,hi=np.quantile(delta,[.025,.975]);boot.append(dict(method=method,comparator=METHODS[0],dataset='equal_weight_mean',delta_f1=float(np.mean(obs)),ci_low=lo,ci_high=hi,n_bootstrap=5000))
    pd.DataFrame(boot).to_csv(OUT/'paired_bootstrap.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'channel_stability.csv',index=False)
    # Cases selected by signed F1 change, retain both success and failure per dataset.
    cases=[]
    for ds in ['SMAP','MSL']:
        a=conf[(conf.dataset==ds)&(conf.method=='level_conditional')].set_index('channel');b=conf[(conf.dataset==ds)&(conf.method==METHODS[0])].set_index('channel').reindex(a.index);delta=a.channel_f1-b.channel_f1
        for kind,c in [('largest_gain',delta.idxmax()),('largest_loss',delta.idxmin())]:cases.append(dict(dataset=ds,channel=c,selection=kind,delta_f1=float(delta[c]),reference_f1=float(b.loc[c,'channel_f1']),selected_f1=float(a.loc[c,'channel_f1'])))
    pd.DataFrame(cases).to_csv(OUT/'case_selection.csv',index=False)
    summary=pd.concat([pd.read_csv(RUN/f'{s}_summary.csv').assign(split=s) for s in ['development','confirmation']])
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    for ax,ds in zip(axs,['SMAP','MSL']):
        pos=np.arange(4)
        for off,sp,col in [(-.18,'development','#98afc4'),(.18,'confirmation','#246a9a')]:
            zz=summary[(summary.dataset==ds)&(summary.split==sp)].set_index('method').reindex(METHODS);ax.bar(pos+off,zz.f1,.35,label=sp,color=col)
        ax.set_xticks(pos,['PCA density','Envelope','Level bins','Robust'],rotation=15);ax.set_title(ds);ax.set_ylim(0,.6);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Micro point F1 (no adjustment)');axs[1].legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'mechanism_f1.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.6))
    for ax,ds in zip(axs,['SMAP','MSL']):
        g=conf[conf.dataset==ds]
        for name in METHODS:
            v=g[g.method==name].channel_f1.to_numpy();v.sort();ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=DISPLAY[name])
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(ds);ax.set_xlabel('Per-channel point F1');ax.grid(alpha=.2)
    axs[0].set_ylabel('Fraction of channels');axs[1].legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(FIG/'channel_ecdf.png');plt.close(fig)
    g=d[(d.dataset=='MSL')&(d.split=='confirmation')].sort_values('false_positives',ascending=False)
    fig,axs=plt.subplots(1,2,figsize=(10,3.5));axs[0].bar(np.arange(len(g)),g.false_positives,color='#246a9a');axs[0].set_xticks(np.arange(len(g)),g.channel,rotation=65,fontsize=8);axs[0].set_ylabel('False-positive points');axs[0].set_title('MSL confirmation: concentration')
    for ds,col in [('SMAP','#246a9a'),('MSL','#d27c44')]:
        q=d[(d.dataset==ds)&(d.split=='confirmation')];axs[1].scatter(q.outside_fit_normal_fraction,q.fpr,label=ds,c=col,alpha=.8)
    axs[1].set_xlabel('Normal test points outside fit range');axs[1].set_ylabel('False-positive rate');axs[1].legend(frameon=False);axs[1].set_xlim(-.02,1.02);axs[1].set_ylim(-.02,1.02);fig.tight_layout();fig.savefig(FIG/'regime_shift.png');plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(11,8.5),gridspec_kw={'width_ratios':[1.2,1]})
    for i,r in enumerate(cases):
        c=r['channel'];x=pd.read_parquet(DATA/'data/test'/f'{c}.parquet').value.to_numpy()[L:];ref=np.load(RUN/'confirmation'/c/'subspace_reference.npz');new=np.load(RUN/'confirmation'/c/'level_conditional.npz');ts=np.arange(L,L+len(x));y=ref['label'];ax=axs[i,0];ax.plot(ts,x,color='#246a9a',lw=.65)
        for a,b in segments(y):ax.axvspan(ts[a],ts[b],color='#e68b73',alpha=.3)
        ax.set_title(f"{r['dataset']} {c}: {r['selection'].replace('_',' ')}, delta F1={r['delta_f1']:+.3f}",fontsize=10);ax.set_ylabel('Value')
        ax=axs[i,1];ax.plot(ts,np.log10(np.maximum(ref['score'],1e-6)),color='#246a9a',lw=.65,label='score');ax.plot(ts,np.log10(new['threshold']),color='#d27c44',lw=.7,label='level threshold');ax.axhline(np.log10(ref['threshold'][0]),color='#8494a1',ls='--',label='global threshold')
        for a,b in segments(y):ax.axvspan(ts[a],ts[b],color='#e68b73',alpha=.3)
        ax.set_ylabel('log10 score / threshold');ax.set_title('Fixed score, changing normal threshold',fontsize=10)
    axs[0,1].legend(fontsize=7,frameon=False);axs[-1,0].set_xlabel('Anonymous sample index');axs[-1,1].set_xlabel('Anonymous sample index');fig.tight_layout();fig.savefig(FIG/'success_failure_cases.png');plt.close(fig)
    # Compare a high-FP channel's NORMAL calibration and known-normal test scores.
    c=g.iloc[0].channel;old=np.load(ROOT/'results_v2/stable_confirm'/c/'subspace_lof50.npz');fig,ax=plt.subplots(figsize=(7,3));bins=np.linspace(-1,6,70)
    for vals,label,col in [(old['calibration'],'Normal calibration','#8494a1'),(old['score'][old['label']==0],'Labeled-normal test','#d27c44'),(old['score'][old['label']==1],'Labeled anomaly','#246a9a')]:
        ax.hist(np.log10(np.maximum(vals,1e-6)),bins=bins,density=True,histtype='step',lw=1.5,label=label,color=col)
    ax.axvline(np.log10(float(old['threshold'])),color='black',ls='--',lw=1,label='Calibration99.5%');ax.set_title(f'MSL {c}: normal calibration does not cover test scores');ax.set_xlabel('log10 density score');ax.set_ylabel('Density');ax.legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(FIG/'normal_test_score_shift.png');plt.close(fig)
    evidence=dict(msl_top_fp_channels=g.head(5)[['channel','false_positives','fpr','outside_fit_normal_fraction']].to_dict('records'),msl_top3_fp_share=float(g.head(3).false_positives.sum()/g.false_positives.sum()),score_shift_case=c,selected='level_conditional',cases=cases,uncertainty='Stratified paired channel percentile bootstrap; temporal dependence kept within channel; no correction for model/history selection; channels may not be independent spacecraft replicates',warmup_dropped=int(inv.warmup_anomaly_points.sum()))
    (OUT/'analysis_evidence.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2));print(pd.DataFrame(boot).to_string(index=False));print(pd.DataFrame(stability).to_string(index=False))
    versions={n:md.version(n) for n in ['numpy','pandas','scikit-learn','pyarrow','torch','matplotlib','joblib','requests']}
    files=list(ROOT.glob('*.py'))+list((ROOT/'research_v4').glob('*'))+list(DATA.rglob('*.parquet'))+[DATA/'labeled_anomalies.csv']+list(RUN.glob('*.json'))
    (OUT/'provenance.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),versions=versions,data_revision=json.loads((ROOT/'results_v2/provenance.json').read_text())['huggingface_revision'],sha256={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()},prior_results_preserved=True),indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(RUN));p.add_argument('--out',default=str(OUT));a=p.parse_args()
    RUN=Path(a.run);OUT=Path(a.out);OUT.mkdir(parents=True,exist_ok=True);FIG=OUT/'figures'
    # Inventory is an audited input shared by declared replications.
    if not (OUT/'data_inventory.csv').exists():
        import shutil
        shutil.copyfile(ROOT/'results_v4/data_inventory.csv',OUT/'data_inventory.csv')
    main()
