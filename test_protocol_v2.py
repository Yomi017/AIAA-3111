import unittest
import numpy as np
import pandas as pd
from experiment_v2 import windows,label_array,measure,L,segments,OUT,split_boundaries
from iterate_v2 import dynamic
from stable_lof_v2 import StableLOF

class ProtocolTests(unittest.TestCase):
    def test_density_duplicates_and_batches(self):
        x=np.vstack([np.zeros((100,64)),np.ones((100,64))]);m=StableLOF().fit(x)
        q=np.vstack([np.zeros((10,64)),np.ones((10,64)),np.ones((10,64))*3])
        s=m.score_samples(q);self.assertTrue(np.isfinite(s).all())
        self.assertTrue(np.array_equal(s[:15],m.score_samples(q[:15])))
        self.assertGreater(-s[-1],-s[0]);self.assertAlmostEqual(-s[0],1)
    def test_short_channel_split(self):
        a,b=split_boundaries(312);self.assertEqual((a,b),(120,216))
        self.assertGreater(a,L+16);self.assertGreaterEqual(b-a,96);self.assertGreaterEqual(312-b,96)
    def test_window_alignment(self):
        x=np.arange(100,dtype=np.float32)[:,None]
        w=windows(x)
        self.assertEqual(w[0,-1,0],63)
        self.assertEqual(w[1,-1,0],64)
        self.assertEqual(len(w[:-1]),len(x[L:]))
        self.assertTrue(np.array_equal(w[:-1,-1,0]+1,x[L:,0]))
    def test_inclusive_labels(self):
        r=pd.Series(dict(num_values=10,chan_id='test',anomaly_sequences='[[2, 4], [8, 8]]'))
        y=label_array(r,10);self.assertEqual(np.flatnonzero(y).tolist(),[2,3,4,8])
    def test_no_point_adjustment(self):
        y=np.array([0,1,1,1,0]);p=np.array([0,1,0,0,0],dtype=bool)
        r=measure(y,p.astype(float),p);self.assertEqual((r['tp'],r['fn'],r['hits']),(1,2,1))
    def test_constant_dynamic(self):
        self.assertTrue(np.isfinite(dynamic(np.zeros(200))))
        self.assertFalse((np.zeros(200)>dynamic(np.zeros(200))).any())
    def test_forecast_aggregation(self):
        # Perfect forecasts at every horizon must score exactly zero, including both edges.
        n=100;h=16;truth=np.arange(n);pred=np.arange(n)[:,None]+np.arange(h)[None,:]
        error=np.zeros(n);count=np.zeros(n)
        for k in range(h):error[k:]+=abs(truth[k:]-pred[:n-k,k]);count[k:]+=1
        self.assertTrue(np.all(error/count==0));self.assertTrue(np.all(count>0))
    def test_saved_support(self):
        for f in (OUT/'dev42').glob('*/zmax.npz'):
            base=np.load(f);self.assertGreater(len(base['label']),0)
            for other in f.parent.glob('*.npz'):
                if other.stem=='normalization':continue
                z=np.load(other);self.assertTrue(np.array_equal(base['label'],z['label']),str(other))
                self.assertEqual(len(z['score']),len(base['label']))
                self.assertTrue(np.isfinite(z['score']).all())

if __name__=='__main__':unittest.main(verbosity=2)
