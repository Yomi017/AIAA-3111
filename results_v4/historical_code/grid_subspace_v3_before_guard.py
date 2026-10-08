from experiment_v2 import *
from stable_lof_v2 import StableLOF
labs=pd.read_csv(OUT/'dev42/channel_split.csv').set_index('chan_id');grid=[]
for var in [.8,.9,.95,.99]:
 for k in [10,20,50,100]:
  for split in ['development','confirmation']:
   rows=[]
   for c,r in labs.iterrows():
    if r.split!=split:continue
    rd=lambda p:pd.read_parquet(p).value.to_numpy(np.float64)[:,None]
    x=rd(DATA/'data/train'/f'{c}.parquet');xt=rd(DATA/'data/test'/f'{c}.parquet');n1,n2=split_boundaries(len(x));mu=x[:n1].mean();sd=max(x[:n1].std(),.01);x=(x-mu)/sd;xt=(xt-mu)/sd
    fw=windows(x[:n1])[:,:,0][::2];pc=PCA(n_components=var,svd_solver='full').fit(fw) if np.var(fw,axis=0).sum()>1e-12 else None
    trans=lambda z:pc.transform(z)/np.sqrt(np.maximum(pc.explained_variance_,1e-6)) if pc else z-fw.mean(0)
    m=StableLOF(k).fit(trans(fw));cal=-m.score_samples(trans(windows(x[n2:])[:,:,0][1:]));test=-m.score_samples(trans(windows(xt)[:,:,0][1:]));t=np.quantile(cal,.995);y=label_array(r,len(xt))[L:];p=test>t
    rows.append(dict(ds=r.spacecraft,tp=int(((y==1)&p).sum()),fp=int(((y==0)&p).sum()),fn=int(((y==1)&~p).sum()),tn=int(((y==0)&~p).sum())))
   for ds in ['SMAP','MSL']:
    a=[z for z in rows if z['ds']==ds];tp=sum(z['tp'] for z in a);fp=sum(z['fp'] for z in a);fn=sum(z['fn'] for z in a);tn=sum(z['tn'] for z in a);grid.append(dict(var=var,k=k,split=split,ds=ds,f1=2*tp/max(1,2*tp+fp+fn),fpr=fp/max(1,fp+tn)))
d=pd.DataFrame(grid);d.to_csv(OUT/'grid_subspace.csv',index=False);print('DEV');print(d[d.split=='development'].sort_values(['ds','f1'],ascending=[True,False]).head(12).to_string(index=False));print('CONF');print(d[d.split=='confirmation'].sort_values(['ds','f1'],ascending=[True,False]).head(12).to_string(index=False))
