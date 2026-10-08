from detector_v2 import *
eligible=pd.read_csv(OUT/'dev42/channel_split.csv');rows=[]
for _,r in eligible.iterrows():
    c=r.chan_id;m=joblib.load(OUT/'models/subspace_lof50'/f'{c}.joblib');x=series(DATA/'data/test'/f'{c}.parquet')
    s,p=predict(m,x);run='stable_dev' if r.split=='development' else 'stable_confirm';old=np.load(OUT/run/c/'subspace_lof50.npz')
    assert np.allclose(s,old['score'],rtol=1e-5,atol=1e-6),c
    assert np.allclose(m['threshold'],old['threshold'],rtol=1e-5,atol=1e-6),c
    assert np.array_equal(p,old['pred']),c
    # Prefix invariance verifies no future test observations enter this detector.
    prefix,_=predict(m,x[:max(128,len(x)//2)]);assert np.allclose(prefix,s[:len(prefix)]),c
    rows.append(dict(channel=c,split=r.split,n=len(s),prediction_mismatches=0,causal_prefix_pass=True))
pd.DataFrame(rows).to_csv(OUT/'export_verification.csv',index=False)
print('Verified saved detector parity and causal prefix invariance:',len(rows),'channels')
