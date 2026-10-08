"""Fit/export and infer normal-trajectory LOF detectors without any anomaly labels.
Examples:
 python detector_v2.py fit --method subspace_lof50
 python detector_v2.py predict --model results_v2/models/subspace_lof50/P-1.joblib --input data/telemanom/data/test/P-1.parquet --output predictions.csv
"""
from experiment_v2 import *
from sklearn.neighbors import LocalOutlierFactor
import joblib
from stable_lof_v2 import StableLOF

def series(path):
    path=Path(path)
    if path.suffix=='.npy':
        a=np.load(path);return a[:,0] if a.ndim==2 else a
    f=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
    return f['value'].to_numpy(np.float64)

def feature(model,x):
    w=windows(((np.asarray(x,dtype=np.float64)-model['mu'])/model['sd'])[:,None])[:,:,0][1:]
    if model['method']=='subspace_lof50':
        pc=model['pca'];w=pc.transform(w)/np.sqrt(np.maximum(pc.explained_variance_,1e-6)) if pc is not None else w-model['window_mean']
    return w

def predict(model,x):
    if len(x)<=L:raise ValueError('Need more than 64 samples.')
    s=smooth(-model['lof'].score_samples(feature(model,x)),model['span'])
    return s,s>model['threshold']

def fit_all(method):
    dest=OUT/'models'/method;dest.mkdir(parents=True,exist_ok=True)
    eligible=set(pd.read_csv(OUT/'dev42/channel_split.csv').chan_id)
    for f in sorted((DATA/'data/train').glob('*.parquet')):
        if f.stem not in eligible:continue
        x=series(f);n1,n2=split_boundaries(len(x));mu=x[:n1].mean();sd=x[:n1].std();sd=sd if sd>=.01 else 1
        z=(x-mu)/sd;fw=windows(z[:n1,None])[:,:,0][::2];pc=None
        if method=='subspace_lof50' and np.var(fw,axis=0).sum()>1e-12:
            pc=PCA(n_components=.95,svd_solver='full').fit(fw)
        m=dict(method=method,L=L,mu=mu,sd=sd,pca=pc,window_mean=fw.mean(0),span=1 if method=='subspace_lof50' else 64,n1=n1,n2=n2)
        train=fw
        if method=='subspace_lof50':train=pc.transform(fw)/np.sqrt(np.maximum(pc.explained_variance_,1e-6)) if pc is not None else fw-fw.mean(0)
        m['lof']=StableLOF(n_neighbors=50).fit(train)
        cal=smooth(-m['lof'].score_samples(feature(m,x[n2:])),m['span']);m['threshold']=max(float(np.quantile(cal,.995)),1e-6)
        joblib.dump(m,dest/(f.stem+'.joblib'),compress=3)
    print('Exported',len(list(dest.glob('*.joblib'))),'channel models to',dest)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fit');f.add_argument('--method',choices=['subspace_lof50','trajectory_lof50'],default='subspace_lof50')
    a=sub.add_parser('predict');a.add_argument('--model',required=True);a.add_argument('--input',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='fit':fit_all(args.method)
    else:
        m=joblib.load(args.model);x=series(args.input);s,pred=predict(m,x)
        pd.DataFrame(dict(timestep=np.arange(L,len(x)),value=x[L:],score=s,threshold=m['threshold'],anomaly=pred.astype(int))).to_csv(args.output,index=False)
        print('Scored',len(s),'samples. First64 are warmup.')
