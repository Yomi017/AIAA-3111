"""Meaningful invariants: future perturbations, calibration independence and exports."""
import unittest,json
import numpy as np
import pandas as pd
import joblib
from experiment_v4 import ROOT,OUT,DATA,METHODS,fit_model,predict,L
RUN=OUT/'mechanism'

class ProtocolTests(unittest.TestCase):
    def test_normal_model_causal_future_perturbation(self):
        rng=np.random.default_rng(4);x=np.sin(np.arange(700)/18)+rng.normal(0,.02,700)
        m=fit_model(x);a=np.cos(np.arange(250)/20);b=a.copy();b[170:]+=100
        for name in METHODS:
            s,t,p=predict(m,a,name);ss,tt,pp=predict(m,b,name)
            np.testing.assert_allclose(s[:170-L],ss[:170-L],rtol=1e-10,atol=1e-9)
            np.testing.assert_array_equal(t[:170-L],tt[:170-L]);np.testing.assert_array_equal(p[:170-L],pp[:170-L])
    def test_constant_model_and_short_normal(self):
        m=fit_model(np.zeros(312))
        for name in METHODS:
            s,t,p=predict(m,np.zeros(200),name);self.assertTrue(np.isfinite(s).all());self.assertFalse(p.any())
        self.assertGreaterEqual(min(m['thresholds'].values()),1e-6)
    def test_calibration_cannot_change_fitted_representation(self):
        x=np.sin(np.arange(700)/18);m=fit_model(x);b=x.copy();b[m['n2']:]+=100;n=fit_model(b)
        self.assertEqual(m['mu'],n['mu']);self.assertEqual(m['sd'],n['sd'])
        np.testing.assert_array_equal(m['pca'].components_,n['pca'].components_)
        np.testing.assert_array_equal(m['feature_center'],n['feature_center'])
        q=np.arange(250)/100
        np.testing.assert_array_equal(predict(m,q,'subspace_reference')[0],predict(n,q,'subspace_reference')[0])
        self.assertNotEqual(m['thresholds']['subspace_reference'],n['thresholds']['subspace_reference'])
    def test_development_reference_matches_preserved_scores(self):
        for f in (RUN/'development').glob('*/subspace_reference.npz'):
            old=np.load(ROOT/'results_v2/stable_dev'/f.parent.name/'subspace_lof50.npz');new=np.load(f)
            np.testing.assert_allclose(old['score'],new['score'],rtol=1e-8,atol=1e-8)
            np.testing.assert_array_equal(old['pred'],new['pred'])
    def test_all_saved_models_reload_and_prefix(self):
        count=0
        for f in RUN.glob('*/*/model.joblib'):
            c=f.parent.name;m=joblib.load(f);x=pd.read_parquet(DATA/'data/test'/f'{c}.parquet').value.to_numpy(np.float64);end=min(len(x)-1,500)
            for name in METHODS:
                stored=np.load(f.parent/(name+'.npz'));s,t,p=predict(m,x,name);ss,tt,pp=predict(m,x[:end],name)
                np.testing.assert_allclose(s,stored['score'],rtol=0,atol=0);np.testing.assert_array_equal(p,stored['pred'])
                np.testing.assert_allclose(ss,s[:end-L],rtol=1e-10,atol=1e-9);np.testing.assert_array_equal(pp,p[:end-L]);np.testing.assert_array_equal(tt,t[:end-L])
            count+=1
        self.assertGreaterEqual(count,22)
    def test_supervised_group_separation(self):
        seen={'hgb':set(),'logistic':set()}
        for f in (ROOT/'results_v3/group5').glob('*/done.json'):
            r=json.loads(f.read_text());self.assertFalse(set(r['source_final'])&set(r['target']));self.assertFalse(set(r['train_inner'])&set(r['validation_inner']))
            self.assertFalse(seen[r['kind']]&set(r['target']));seen[r['kind']].update(r['target'])
        self.assertEqual(len(seen['hgb']),80);self.assertEqual(len(seen['logistic']),80)
if __name__=='__main__':
    import argparse,sys
    from pathlib import Path
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--run',default=str(RUN));args,remaining=parser.parse_known_args();RUN=Path(args.run)
    unittest.main(argv=[sys.argv[0]]+remaining,verbosity=2)
