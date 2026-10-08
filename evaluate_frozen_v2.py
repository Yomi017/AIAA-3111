"""Evaluate only configurations selected on development; never search on confirmation."""
from experiment_v2 import *
from iterate_v2 import dynamic

def evaluate(runs,split):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());config={c['method']:c for c in frozen['models']};rows=[]
    for run in runs:
        for f in (OUT/run).glob('*/done.json'):
            for meta in json.loads(f.read_text())['metrics']:
                method=meta['method']
                if method not in config or meta['split']!=split:continue
                c=config[method];z=np.load(f.parent/(method+'.npz'));score=smooth(z['score'],int(c['span']));cal=smooth(z['calibration'],int(c['span']))
                threshold=max(float(np.quantile(cal,.995)),1e-6) if c['policy']=='cal995' else dynamic(score)
                rows.append(dict(dataset=meta['dataset'],channel=meta['channel'],split=split,method=method,span=c['span'],policy=c['policy'],threshold=threshold,**measure(z['label'],score,score>threshold)))
    df=pd.DataFrame(rows);assert not df.duplicated(['dataset','channel','method']).any()
    df.to_csv(OUT/(split+'_frozen_channels.csv'),index=False);agg=[]
    for (ds,m),g in df.groupby(['dataset','method']):
        s=g[['tp','fp','fn','tn','hits','events','alerts','hit_alerts','delay_sum']].sum().to_dict();p=s['tp']/max(1,s['tp']+s['fp']);r=s['tp']/max(1,s['tp']+s['fn'])
        agg.append(dict(dataset=ds,method=m,channels=len(g),f1=2*p*r/max(1e-12,p+r),precision=p,recall=r,fpr=s['fp']/max(1,s['fp']+s['tn']),event_recall=s['hits']/max(1,s['events']),alert_precision=s['hit_alerts']/max(1,s['alerts']),delay_detected=s['delay_sum']/max(1,s['hits']),macro_ap=g.ap.mean(),**s))
    a=pd.DataFrame(agg);a.to_csv(OUT/(split+'_frozen.csv'),index=False)
    print(a.pivot(index='method',columns='dataset',values='f1').round(4).to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='confirmation');p.add_argument('--runs',nargs='+',default=['confirm42','stable_confirm','extra_confirm','recon_confirm']);a=p.parse_args();evaluate(a.runs,a.split)
