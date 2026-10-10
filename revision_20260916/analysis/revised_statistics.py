"""Statistics from fixed revision predictions; no model fitting or test tuning."""
from pathlib import Path
from evidence_io import read_csv
import json, sys
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';OUT=ROOT/'tables'
SEED=20260916;B=2000
NAMES=['roc_auc','average_precision','brier','ece','f1','sensitivity','specificity','precision','accuracy']

def estimates(d):
    y=d.label.to_numpy();p=d.probability.to_numpy();r=d.prediction.to_numpy()
    tp=((y==1)&(r==1)).sum();tn=((y==0)&(r==0)).sum();fp=((y==0)&(r==1)).sum();fn=((y==1)&(r==0)).sum()
    bins=np.minimum((p*10).astype(int),9)
    return np.array([roc_auc_score(y,p),average_precision_score(y,p),np.mean((y-p)**2),np.abs(np.bincount(bins,weights=y-p,minlength=10)).sum()/len(y),2*tp/(2*tp+fp+fn),tp/(tp+fn),tn/(tn+fp),tp/(tp+fp) if tp+fp else 0.,(tp+tn)/len(y)])

def boot(d,draws,ids):
    g=pd.Categorical(d.abstract_id,categories=ids).codes;assert (g>=0).all()
    y=d.label.to_numpy();p=d.probability.to_numpy();r=d.prediction.to_numpy();G=len(ids)
    aggregate=lambda v:np.bincount(g,weights=v,minlength=G)
    P=draws@aggregate(y);N=draws@aggregate(1-y);n=P+N
    tp=draws@aggregate((y==1)&(r==1));tn=draws@aggregate((y==0)&(r==0));fp=N-tn;fn=P-tp
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(np.abs(draws@aggregate((y-p)*(bins==b))) for b in range(10))/n
    order=np.argsort(p,kind='stable');ys=y[order];gs=g[order];ps=p[order];starts=np.r_[0,np.flatnonzero(np.diff(ps))+1]
    auc=[];ap=[]
    for i,c in enumerate(draws):
        w=c[gs];pos=np.add.reduceat(w*ys,starts);neg=np.add.reduceat(w*(1-ys),starts)
        auc.append(np.sum(pos*(np.cumsum(neg)-.5*neg))/(P[i]*N[i]) if P[i]*N[i] else np.nan)
        cp=np.cumsum(pos[::-1]);ct=np.cumsum((pos+neg)[::-1]);precision=np.divide(cp,ct,out=np.zeros_like(cp,dtype=float),where=ct>0)
        ap.append(np.sum(precision*pos[::-1])/P[i] if P[i] else np.nan)
    with np.errstate(divide='ignore',invalid='ignore'):
        out=np.column_stack([auc,ap,(draws@aggregate((y-p)**2))/n,ece,2*tp/(2*tp+fp+fn),tp/P,tn/N,np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0),(tp+tn)/n])
    return out

def getdraws(ids):
    rng=np.random.default_rng(SEED);n=len(ids)
    return np.array([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(B)],dtype=np.int32)

def main():
    expected=['majority_42','tfidf_lr_42','tfidf_svm_42','position_lr_42','unigram_lr_42','word12_lr_42','text_position_lr_42','matched120k_lr_42']+[f'{m}_{s}' for m in ['ffnn','lstm','bilstm','bilstm_dropout'] for s in [42,2026,3407]]
    missing=[x for x in expected if not (RUNS/x/'COMPLETE').exists()]
    if missing:raise RuntimeError('Incomplete runs: '+', '.join(missing))
    allmetrics=pd.concat([read_csv(RUNS/x/'metrics.csv') for x in expected],ignore_index=True)
    allmetrics.to_csv(OUT/'all_metrics.csv',index=False)
    mainmodels=['majority','tfidf_lr','tfidf_svm','ffnn','lstm','bilstm','bilstm_dropout']
    # Selection uses raw margins/logits, regardless of post-hoc calibration slope.
    select=[]
    for name in expected:
        model,seed=name.rsplit('_',1)
        if model not in mainmodels:continue
        d=read_csv(RUNS/name/'predictions_selection.csv')
        select.append({'model':model,'seed':int(seed),'selection_auc':roc_auc_score(d.label,d.raw_score)})
    selection=pd.DataFrame(select);selection.to_csv(OUT/'model_selection.csv',index=False)
    means=selection.groupby('model').selection_auc.mean().sort_values(ascending=False)
    selected=means.index[0];print('SELECTED',selected,means.to_dict(),flush=True)
    (OUT/'selected_model.json').write_text(json.dumps({'model':selected,'representative_seed':42,'criterion':'Mean raw-score selection ROC AUC across seeds; baseline and classical models have one fixed run','selection_mean_auc':means.to_dict()},indent=2))
    seedsummary=allmetrics.groupby(['model','split']).agg(n_seeds=('seed','size'),auc_mean=('roc_auc','mean'),auc_sd=('roc_auc','std'),f1_mean=('f1','mean'),f1_sd=('f1','std'),brier_mean=('brier_score','mean'),brier_sd=('brier_score','std'),ece_mean=('ece_10bin','mean'),ece_sd=('ece_10bin','std')).reset_index()
    seedsummary.to_csv(OUT/'seed_summary.csv',index=False)
    cis=[];paired=[];curves=[];deltas=[];subgroups=[];errors=[];examples=[];ecechecks=[]
    train=read_csv(ROOT/'data/train.csv');cuts=np.quantile(train.word_count,[.25,.5,.75])
    normalize=lambda s:s.astype(str).str.lower().str.split().str.join(' ')
    development_text=set(normalize(train.text));del train
    for name in ['selection','calibration','threshold']:
        development_text.update(normalize(read_csv(ROOT/'data'/f'{name}.csv',usecols=['text']).text))
    duplicate_sensitivity=[]
    for split in ['test','external']:
        anchor=read_csv(RUNS/f'{selected}_42'/f'predictions_{split}.csv',dtype={'abstract_id':str})
        ids=np.unique(anchor.abstract_id);draws=getdraws(ids);arrays={};points={}
        duplicate_mask=normalize(anchor.text).isin(development_text)
        nonduplicate=anchor[~duplicate_mask]
        duplicate_arr=boot(nonduplicate,draws,ids);duplicate_point=estimates(nonduplicate)
        lo,hi=np.nanpercentile(duplicate_arr[:,0],[2.5,97.5])
        duplicate_sensitivity.append({'split':split,'model':selected,'excluded_sentences':int(duplicate_mask.sum()),'retained_sentences':len(nonduplicate),'retained_abstracts':nonduplicate.abstract_id.nunique(),'auc':duplicate_point[0],'ci_low':lo,'ci_high':hi,'rule':'Sensitivity only: exclude every evaluation sentence whose lowercase whitespace-normalized text occurs in training or any development subset; no label or score used.'})
        for name in [x for x in expected if x.endswith('_42')]:
            d=read_csv(RUNS/name/f'predictions_{split}.csv',dtype={'abstract_id':str})
            assert d[['abstract_id','sentence_index']].equals(anchor[['abstract_id','sentence_index']])
            arr=boot(d,draws,ids);point=estimates(d);arrays[name]=arr;points[name]=point
            for j,metric in enumerate(NAMES):
                lo,hi=np.nanpercentile(arr[:,j],[2.5,97.5]);cis.append(dict(run=name,split=split,metric=metric,estimate=point[j],ci_low=lo,ci_high=hi,valid_replicates=int(np.isfinite(arr[:,j]).sum())))
            print('BOOTSTRAP',split,name,flush=True)
        for other in arrays:
            if other==f'{selected}_42':continue
            diff=arrays[f'{selected}_42'][:,0]-arrays[other][:,0];lo,hi=np.nanpercentile(diff,[2.5,97.5])
            paired.append(dict(split=split,reference=f'{selected}_42',comparator=other,delta_auc=points[f'{selected}_42'][0]-points[other][0],ci_low=lo,ci_high=hi))
        g=pd.Categorical(anchor.abstract_id,categories=ids).codes;agg=lambda v:np.bincount(g,weights=v,minlength=len(ids))
        y=anchor.label.to_numpy();p=anchor.probability.to_numpy();raw=expit(anchor.raw_score.to_numpy());n=draws@agg(np.ones(len(y)))
        diff=(draws@agg((p-y)**2-(raw-y)**2))/n
        deltas.append({'split':split,'model':selected,'metric':'post minus pre Brier','estimate':np.mean((p-y)**2-(raw-y)**2),'ci_low':np.percentile(diff,2.5),'ci_high':np.percentile(diff,97.5)})
        for bins_n in [5,10,20]:
            for label,probs in [('pre',raw),('post',p)]:
                bins=np.minimum((probs*bins_n).astype(int),bins_n-1)
                value=np.abs(np.bincount(bins,weights=y-probs,minlength=bins_n)).sum()/len(y)
                ecechecks.append({'split':split,'model':selected,'probability':label,'n_bins':bins_n,'ece':value})
                if bins_n==10:
                    for b in range(10):
                        m=bins==b;den=draws@agg(m);num=draws@agg(y*m)
                        frac=np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)
                        if m.any():curves.append({'split':split,'probability':label,'bin':b,'n':int(m.sum()),'mean_probability':probs[m].mean(),'observed_fraction':y[m].mean(),'ci_low':np.nanpercentile(frac,2.5),'ci_high':np.nanpercentile(frac,97.5),'valid_replicates':int(np.isfinite(frac).sum())})
        bins=np.digitize(anchor.word_count.to_numpy(),cuts,right=True)
        for b in range(4):
            d=anchor[bins==b];arr=boot(d,draws,ids);point=estimates(d)
            for j,metric in enumerate(NAMES):
                lo,hi=np.nanpercentile(arr[:,j],[2.5,97.5]);subgroups.append({'split':split,'quartile':b+1,'metric':metric,'sentences':len(d),'abstracts':d.abstract_id.nunique(),'estimate':point[j],'ci_low':lo,'ci_high':hi,'valid_replicates':int(np.isfinite(arr[:,j]).sum())})
        for label,d in anchor.groupby('source_label'):
            fp=d[(d.label==0)&(d.prediction==1)];fn=d[(d.label==1)&(d.prediction==0)]
            errors.append({'split':split,'source_label':label,'n':len(d),'false_positive':len(fp),'false_negative':len(fn),'error_rate':(len(fp)+len(fn))/len(d)})
            for kind,err in [('FP',fp),('FN',fn)]:
                if err.empty:continue
                # Deterministic representative near the median confidence within stratum.
                err=err.sort_values(['probability','abstract_id','sentence_index']);chosen=err.iloc[len(err)//2]
                examples.append({'split':split,'error':kind,**chosen.to_dict(),'selection_rule':'Middle probability-ranked error in this corpus/source-label/direction stratum'})
    for filename,rows in [('cluster_intervals',cis),('paired_auc_differences',paired),('calibration_curve_intervals',curves),('paired_calibration_differences',deltas),('length_subgroups',subgroups),('errors_by_source_label',errors),('illustrative_errors',examples),('ece_binning_sensitivity',ecechecks)]:pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    pd.DataFrame(duplicate_sensitivity).to_csv(OUT/'sentence_overlap_sensitivity.csv',index=False)
    costs=[]
    for name in expected:
        c=json.loads((RUNS/name/'computational_cost.json').read_text());costs.append({'run':name,**{k:c[k] for k in ['training_seconds','calibration_threshold_seconds','inference_seconds_median','artifact_bytes','peak_process_working_set_bytes']}})
    pd.DataFrame(costs).to_csv(OUT/'computational_costs.csv',index=False)
    # Coefficients of word features are associations in the fitted lexical model.
    import joblib
    if (ROOT/'shared_artifacts/wordchar_vectorizer.joblib').exists() and (RUNS/'tfidf_lr_42/classifier.joblib').exists():
        vec=joblib.load(ROOT/'shared_artifacts/wordchar_vectorizer.joblib');clf=joblib.load(RUNS/'tfidf_lr_42/classifier.joblib')
        terms=vec.get_feature_names_out();coef=clf.coef_[0];mask=np.array([t.startswith('word__') for t in terms]);ii=np.flatnonzero(mask);order=ii[np.argsort(coef[ii])]
        pd.DataFrame({'feature':terms[np.r_[order[:20],order[-20:]]],'coefficient':coef[np.r_[order[:20],order[-20:]]]}).to_csv(OUT/'lexical_coefficients.csv',index=False)
    elif not (OUT/'lexical_coefficients.csv').exists():
        raise FileNotFoundError('Refit logistic model or supply the archived lexical-coefficient table.')
    (OUT/'statistics_manifest.json').write_text(json.dumps({'bootstrap_replicates':B,'seed':SEED,'unit':'abstract PMID','ci_method':'percentile 2.5 and 97.5; conditional on fitted models','seed_sd':'sample standard deviation ddof=1; not a confidence interval','length_cutoffs_training_words':cuts.tolist(),'paired_draws':True,'one_class_replicates':'excluded for AUC; valid counts saved','scope':'All revised metrics, except explicit historical audit, use filtered test sets'},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
