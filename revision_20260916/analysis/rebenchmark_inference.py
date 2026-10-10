"""Serial inference remeasurement after every fitting process has completed."""
from pathlib import Path
from evidence_io import read_csv
import json,time,gc
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from predict_saved_model import predict,ARTIFACT_CACHE

ROOT=Path(__file__).resolve().parents[1]
expected=['majority_42','tfidf_lr_42','tfidf_svm_42','position_lr_42','unigram_lr_42','word12_lr_42','text_position_lr_42','matched120k_lr_42']+[f'{m}_{s}' for m in ['ffnn','lstm','bilstm','bilstm_dropout'] for s in [42,2026,3407]]
assert all((ROOT/'runs'/name/'COMPLETE').exists() for name in expected), 'All twenty runs must be complete'
data=read_csv(ROOT/'data/test.csv',dtype={'abstract_id':str}).iloc[:1000]
# The loader caches artifacts on the first warmup. All timed repetitions then
# use in-memory weights and preprocessing; no artifact or sentence-file IO.
results=[]
with threadpool_limits(limits=4):
    for run in sorted((ROOT/'runs').iterdir()):
        if not (run/'COMPLETE').exists():raise RuntimeError('All model fits must complete before serial benchmarking')
        predict(ROOT,run.name,data);predict(ROOT,run.name,data);times=[]
        for _ in range(5):
            start=time.perf_counter();raw,prob,classes=predict(ROOT,run.name,data);times.append(time.perf_counter()-start)
        ref=read_csv(run/'predictions_test.csv').iloc[:1000]
        np.testing.assert_allclose(raw,ref.raw_score,rtol=1e-5,atol=2e-5);np.testing.assert_array_equal(classes,ref.prediction)
        path=run/'computational_cost.json';cost=json.loads(path.read_text())
        cost['original_artifact_bytes_excluding_config']=cost['artifact_bytes']
        artifact_files=set(p.replace('\\','/') for p in cost['artifact_files'])
        artifact_files.update([f'runs/{run.name}/config.json',f'runs/{run.name}/calibrator.json'])
        cost['artifact_files']=sorted(artifact_files)
        cost['artifact_bytes']=sum((ROOT/p).stat().st_size for p in artifact_files)
        cost['original_warm_inference_seconds_repetitions']=cost['inference_seconds_repetitions']
        cost['original_warm_inference_seconds_median']=cost['inference_seconds_median']
        cost['inference_seconds_repetitions']=times;cost['inference_seconds_median']=float(np.median(times))
        cost['inference_scope']='Serial after all training; 1000 in-memory sentence inputs to probabilities with cached models and preprocessing; 2 warmups and 5 repeats; artifact loading and CSV reading excluded'
        cost['benchmark_label']='Warm end-to-end prediction latency measured serially after fitting'
        path.write_text(json.dumps(cost,indent=2),encoding='utf-8')
        results.append({'run':run.name,'seconds_median':float(np.median(times)),'prediction_classes_match':True,'max_abs_raw_difference':float(np.max(np.abs(raw-ref.raw_score)))})
        print(run.name,results[-1]['seconds_median'],flush=True);ARTIFACT_CACHE.clear();gc.collect()
(ROOT/'tables/serial_inference_verification.json').write_text(json.dumps(results,indent=2))
