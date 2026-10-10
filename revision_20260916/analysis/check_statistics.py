"""Validate weighted bootstrap calculations against explicit resampling."""
import numpy as np
import pandas as pd
from revised_statistics import boot,estimates

d=pd.DataFrame({'abstract_id':['a','a','b','b','c','c','c'],'label':[0,1,0,1,0,1,1],'probability':[.2,.8,.8,.8,.2,.4,.9],'prediction':[0,1,1,1,0,0,1]})
draws=np.array([[1,1,1],[2,0,1],[0,1,2],[1,2,0]])
actual=boot(d,draws,np.array(['a','b','c']))
for i,c in enumerate(draws):
    expanded=pd.concat([d[d.abstract_id==pid] for j,pid in enumerate(['a','b','c']) for _ in range(c[j])],ignore_index=True)
    np.testing.assert_allclose(actual[i],estimates(expanded),rtol=1e-12,atol=1e-12)
print('PASS: weighted clustered resampling equals explicit duplicated abstracts for all nine metrics, including tied-score ROC AUC and average precision.')

# Check the reporting helper against explicit nonempty-bin calculations at
# decimal boundaries, including probability one and multiple empty bins.
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy_src'))
from src.evaluation.metrics import expected_calibration_error
y=np.array([0,1,0,1,0]);p=np.array([0.,.1,.1,.3,1.])
expected=(abs(0.-0.)+2*abs(.5-.1)+abs(1.-.3)+abs(0.-1.))/5
np.testing.assert_allclose(expected_calibration_error(y,p,10),expected,rtol=1e-14)
print('PASS: ECE bin boundaries, final endpoint, weights and empty-bin treatment.')
