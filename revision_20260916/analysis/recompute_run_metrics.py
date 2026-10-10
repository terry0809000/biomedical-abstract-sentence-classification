"""Unify metrics on the exact archived probabilities, without refitting models.

The historical helper clipped all probabilities to [1e-7, 1-1e-7]. The revision
uses full saved precision for evaluation and clips only when taking logits for
the descriptive calibration intercept/slope. This cannot change any model fit.
"""
from pathlib import Path
from evidence_io import read_csv
import json, sys
import numpy as np
import pandas as pd
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/legacy_src'))
from src.evaluation.metrics import binary_metrics

expected=['majority_42','tfidf_lr_42','tfidf_svm_42','position_lr_42','unigram_lr_42','word12_lr_42','text_position_lr_42','matched120k_lr_42']+[f'{m}_{s}' for m in ['ffnn','lstm','bilstm','bilstm_dropout'] for s in [42,2026,3407]]
assert all((ROOT/'runs'/name/'COMPLETE').exists() for name in expected)
audit=[]
for name in expected:
    run=ROOT/'runs'/name;cfg=json.loads((run/'config.json').read_text())
    model,seed=name.rsplit('_',1);rows=[]
    for split in ['selection','calibration','threshold','test','external']:
        d=read_csv(run/f'predictions_{split}.csv')
        np.testing.assert_array_equal((d.probability>=cfg['threshold']).astype(int),d.prediction)
        metrics=binary_metrics(d.label,d.probability,cfg['threshold'])
        if model not in ['majority','tfidf_svm']:
            pre=binary_metrics(d.label,expit(d.raw_score),.5)
            metrics.update(pre_brier=pre['brier_score'],pre_ece=pre['ece_10bin'],
                           delta_brier=metrics['brier_score']-pre['brier_score'],
                           delta_ece=metrics['ece_10bin']-pre['ece_10bin'])
        rows.append({'model':model,'seed':int(seed),'split':split,**metrics})
        audit.append({'run':name,'split':split,'n':len(d),
                      'outside_old_clipping_bounds':int(((d.probability<1e-7)|(d.probability>1-1e-7)).sum()),
                      'predictions_match_full_precision_threshold':True})
    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
(ROOT/'tables/full_precision_metric_audit.json').write_text(json.dumps(audit,indent=2))
print('PASS: all twenty run metrics evaluated on exact saved probabilities; only calibration logits are clipped.')
