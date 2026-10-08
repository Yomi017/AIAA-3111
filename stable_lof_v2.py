"""Regularized novelty LOF with direct-distance neighbors for duplicate trajectories.
No test observations or labels affect the normal reference distribution.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

class StableLOF:
    def __init__(self,n_neighbors=50):self.n_neighbors=n_neighbors
    def fit(self,x):
        x=np.round(np.asarray(x,dtype=np.float64),12)
        self.k=min(self.n_neighbors,len(x)-1)
        # Ball-tree evaluates Euclidean distances directly; avoids cancellation in x.x+y.y-2x.y.
        self.neighbors=NearestNeighbors(n_neighbors=self.k,algorithm='ball_tree',n_jobs=4).fit(x)
        distances,indices=self.neighbors.kneighbors(None)
        self.kdist=distances[:,-1]
        positive=self.kdist[self.kdist>1e-8]
        self.floor=max(1e-6,.001*float(np.median(positive))) if len(positive) else 1e-6
        reach=np.maximum(np.maximum(distances,self.kdist[indices]),self.floor)
        self.lrd=1/np.mean(reach,axis=1)
        return self
    def score_samples(self,x):
        x=np.round(np.asarray(x,dtype=np.float64),12)
        distances,indices=self.neighbors.kneighbors(x)
        reach=np.maximum(np.maximum(distances,self.kdist[indices]),self.floor)
        lof=np.mean(self.lrd[indices],axis=1)*np.mean(reach,axis=1)
        return -np.round(lof,12)
