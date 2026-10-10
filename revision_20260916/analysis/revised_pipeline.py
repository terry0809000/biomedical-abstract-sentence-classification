"""Revision pipeline with disjoint selection/calibration/threshold documents.
Scientific runtime: Python with numpy pandas scipy sklearn torch joblib psutil.
All outputs are new files; the original study is never overwritten.
"""
from pathlib import Path
import argparse, copy, gc, hashlib, json, os, platform, random, sys, time
import joblib
import numpy as np
import pandas as pd
import psutil
import scipy
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import expit
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROJECT=ROOT.parent/'Healthcare_NLP_CPU_Project'
sys.path.insert(0,str(HERE/'legacy_src'))
from src.data.parsers import parse_pubmed_pico,parse_pubmed_rct,add_text_features
from src.features.neural_tokenizer import NeuralTokenizer
from src.evaluation.metrics import binary_metrics,optimize_threshold

SEEDS=[42,2026,3407]
SPLIT_SEED=20260916
THREADS=4
DATA=ROOT/'data'; RUNS=ROOT/'runs'; TABLES=ROOT/'tables'
for p in [DATA,RUNS,TABLES]:p.mkdir(exist_ok=True,parents=True)

def savejson(p,obj):p.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def abstract_hashes(d):
    return d.groupby('abstract_id',sort=False).text.agg(lambda x:hashlib.sha256('\n'.join(' '.join(str(t).lower().split()) for t in x).encode()).hexdigest())
def add_position(d):
    d=d.copy(); n=d.groupby('abstract_id').sentence_index.transform('size')
    d['position']=np.where(n>1,d.sentence_index/np.maximum(n-1,1),0.)
    return d
def prepare():
    raw=PROJECT/'data/raw'
    ds={k:add_text_features(parse_pubmed_pico(raw/f'pubmed_pico_{f}.txt',k)) for k,f in [('train','train'),('validation','dev'),('test','test')]}
    ds['external']=add_text_features(parse_pubmed_rct(raw/'pubmed_20k_rct_test.txt','external'))
    for k,d in ds.items():
        refpath=PROJECT/'data/processed'/('external_pubmed_rct20k_test.csv' if k=='external' else f'primary_{k}.csv')
        if refpath.exists():
            ref=pd.read_csv(refpath,dtype={'abstract_id':str})
            assert all(d[c].reset_index(drop=True).equals(ref[c].reset_index(drop=True)) for c in ['abstract_id','sentence_index','source_label','label','text'])
    h={k:abstract_hashes(d) for k,d in ds.items()}; exclusions=[]
    seen_hash=set(h['train'])|set(h['validation'])
    remove=set(h['test'][h['test'].isin(seen_hash)].index)
    for pid in sorted(remove):exclusions.append({'split':'test','abstract_id':pid,'reason':'exact normalized whole abstract already in training/development'})
    ds['test']=ds['test'][~ds['test'].abstract_id.isin(remove)].copy()
    allprimary=set().union(*(set(h[k].index) for k in ['train','validation','test']))
    allhash=set().union(*(set(h[k]) for k in ['train','validation','test']))
    remove=set(ds['external'].abstract_id)&allprimary
    removehash=set(h['external'][h['external'].isin(allhash)].index)
    for pid in sorted(remove|removehash):exclusions.append({'split':'external','abstract_id':pid,'reason':'PMID overlaps primary corpus' if pid in remove else 'exact normalized whole abstract overlaps primary corpus'})
    ds['external']=ds['external'][~ds['external'].abstract_id.isin(remove|removehash)].copy()
    ids=np.array(sorted(ds['validation'].abstract_id.unique())); np.random.default_rng(SPLIT_SEED).shuffle(ids)
    roles={'selection':set(ids[:1200]),'calibration':set(ids[1200:1800]),'threshold':set(ids[1800:])}
    assignments=[]
    for role,values in roles.items():
        ds[role]=ds['validation'][ds['validation'].abstract_id.isin(values)].copy()
        assignments.extend({'abstract_id':x,'role':role} for x in sorted(values))
    del ds['validation']
    frac=120000/len(ds['train'])
    subidx=np.concatenate([g.sample(max(1,int(round(len(g)*frac))),random_state=42).index.to_numpy() for _,g in ds['train'].groupby('label')])
    subidx=np.random.default_rng(42).permutation(subidx)
    pd.DataFrame({'train_row':subidx}).to_csv(DATA/'neural_training_rows.csv',index=False)
    for k,d in ds.items():
        d=add_position(d).reset_index(drop=True); ds[k]=d; d.to_csv(DATA/f'{k}.csv',index=False)
    for a in ds:
        for b in ds:
            if a!=b:assert not set(ds[a].abstract_id)&set(ds[b].abstract_id),(a,b)
    pd.DataFrame(exclusions).to_csv(DATA/'excluded_abstracts.csv',index=False)
    pd.DataFrame(assignments).to_csv(DATA/'development_assignments.csv',index=False)
    counts=[{'split':k,'sentences':len(d),'abstracts':d.abstract_id.nunique(),'positive':int(d.label.sum()),'negative':int((1-d.label).sum()),'mean_words':d.word_count.mean(),'median_words':d.word_count.median()} for k,d in ds.items()]
    pd.DataFrame(counts).to_csv(TABLES/'dataset_counts.csv',index=False)
    savejson(DATA/'manifest.json',{'split_seed':SPLIT_SEED,'source_hashes':{p.name:hashfile(p) for p in raw.glob('*.txt')},'outputs':{p.name:hashfile(p) for p in DATA.glob('*.csv')},'neural_training_sentences':len(subidx),'excluded':exclusions})
    print('PREPARED',counts,flush=True)

def load():return {k:pd.read_csv(DATA/f'{k}.csv',dtype={'abstract_id':str}) for k in ['train','selection','calibration','threshold','test','external']}
def memory():
    m=psutil.Process().memory_info()
    return {'rss_bytes_at_measurement':m.rss,'peak_process_working_set_bytes':getattr(m,'peak_wset',None),'memory_scope':'Whole current process including shared feature matrices; not isolated model-only allocation'}
def environment():
    return {'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'ram_bytes':psutil.virtual_memory().total,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threads':THREADS}
def finalize(name,seed,scores,frames,run,config,training_seconds,artifact_paths,infer):
    tic=time.perf_counter()
    if name=='majority':
        prob={k:v for k,v in scores.items()}; cal={'kind':'none'}; threshold=.5
    elif config['family']=='neural':
        x=scores['calibration']; y=frames['calibration'].label.to_numpy()
        f=lambda l: np.mean(np.logaddexp(0,x/np.exp(l))-y*x/np.exp(l))
        result=minimize_scalar(f,bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-8})
        if not result.success:raise RuntimeError(result.message)
        temp=float(np.exp(result.x)); cal={'kind':'temperature','temperature':temp,'objective':'unweighted binary log loss'}
        prob={k:expit(v/temp) for k,v in scores.items()}
        threshold=optimize_threshold(frames['threshold'].label,prob['threshold'])
    else:
        c=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000).fit(scores['calibration'].reshape(-1,1),frames['calibration'].label)
        cal={'kind':'sigmoid','coefficient':float(c.coef_[0,0]),'intercept':float(c.intercept_[0]),'C':1.}
        prob={k:c.predict_proba(v.reshape(-1,1))[:,1] for k,v in scores.items()}
        threshold=optimize_threshold(frames['threshold'].label,prob['threshold'])
    caltime=time.perf_counter()-tic
    savejson(run/'calibrator.json',cal); savejson(run/'config.json',{**config,'seed':seed,'threshold':threshold,'environment':environment()})
    rows=[]
    for split,d in frames.items():
        if split=='train':continue
        pred=d[['abstract_id','sentence_index','source_label','text','label','word_count','position']].copy()
        pred['raw_score']=scores[split]; pred['probability']=prob[split]; pred['prediction']=(prob[split]>=threshold).astype(int)
        pred.to_csv(run/f'predictions_{split}.csv',index=False)
        m=binary_metrics(pred.label.to_numpy(),pred.probability.to_numpy(),threshold)
        if name!='majority' and name!='tfidf_svm':
            raw=expit(scores[split]); pre=binary_metrics(d.label.to_numpy(),raw,.5)
            m.update(pre_brier=pre['brier_score'],pre_ece=pre['ece_10bin'],delta_brier=m['brier_score']-pre['brier_score'],delta_ece=m['ece_10bin']-pre['ece_10bin'])
        rows.append({'model':name,'seed':seed,'split':split,**m})
    pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False)
    # Same 1,000 ordered internal test sentences, preprocessing and calibration included.
    bench=frames['test'].iloc[:1000]
    def prediction():
        v=infer(bench)
        if cal['kind']=='sigmoid':return expit(v*cal['coefficient']+cal['intercept'])
        if cal['kind']=='temperature':return expit(v/cal['temperature'])
        return v
    prediction();prediction();times=[]
    for _ in range(5):
        t=time.perf_counter();prediction();times.append(time.perf_counter()-t)
    savejson(run/'computational_cost.json',{'training_seconds':training_seconds,'calibration_threshold_seconds':caltime,'inference_n':len(bench),'inference_seconds_repetitions':times,'inference_seconds_median':float(np.median(times)),'inference_scope':'End-to-end in-memory sentence text to calibrated probability; 2 warmups and 5 repeats; no disk IO','artifact_bytes':sum(p.stat().st_size for p in artifact_paths)+(run/'calibrator.json').stat().st_size,'artifact_files':[str(p.relative_to(ROOT)) for p in artifact_paths],'environment':environment(),**memory()})
    (run/'COMPLETE').write_text('Complete with metrics, predictions, configuration and costs.\n')
    print('COMPLETE',name,seed,[(r['split'],r['roc_auc']) for r in rows],flush=True)

def positions(d):
    x=d.position.to_numpy();return np.column_stack([x,x*x,x*x*x])

def classical():
    ds=load(); train=ds['train']; y=train.label.to_numpy(); subidx=pd.read_csv(DATA/'neural_training_rows.csv').train_row.to_numpy()
    evals={k:v for k,v in ds.items() if k!='train'}
    run=RUNS/'majority_42';run.mkdir(exist_ok=True)
    if not (run/'COMPLETE').exists():
        prior=float(y.mean());savejson(run/'prior.json',{'prior':prior})
        finalize('majority',42,{k:np.full(len(v),prior) for k,v in evals.items()},evals,run,{'family':'baseline'},0,[run/'prior.json'],lambda d:np.full(len(d),prior))
    run=RUNS/'position_lr_42';run.mkdir(exist_ok=True)
    if not (run/'COMPLETE').exists():
        t=time.perf_counter();m=LogisticRegression(C=2.,solver='liblinear',max_iter=1000,random_state=42).fit(positions(train),y);elapsed=time.perf_counter()-t
        joblib.dump(m,run/'classifier.joblib');f=lambda d:m.decision_function(positions(d))
        finalize('position_lr',42,{k:f(v) for k,v in evals.items()},evals,run,{'family':'ablation','features':'position, position squared, position cubed','C':2},elapsed,[run/'classifier.joblib'],f)
    for feature in ['unigram','word12','wordchar']:
        names={'unigram':['unigram_lr'],'word12':['word12_lr'],'wordchar':['tfidf_lr','tfidf_svm','text_position_lr','matched120k_lr']}[feature]
        if all((RUNS/f'{n}_42/COMPLETE').exists() for n in names):continue
        print('FITTING FEATURES',feature,flush=True); t=time.perf_counter()
        from sklearn.pipeline import FeatureUnion
        word=TfidfVectorizer(lowercase=True,strip_accents='unicode',analyzer='word',ngram_range=(1,1) if feature=='unigram' else (1,2),min_df=2,max_df=.98,max_features=70000,sublinear_tf=True)
        vec=word if feature!='wordchar' else FeatureUnion([('word',word),('char',TfidfVectorizer(lowercase=True,strip_accents='unicode',analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=40000,sublinear_tf=True))],n_jobs=1)
        X=vec.fit_transform(train.text); featuretime=time.perf_counter()-t
        artifacts=ROOT/'shared_artifacts';artifacts.mkdir(exist_ok=True)
        vecpath=artifacts/f'{feature}_vectorizer.joblib';joblib.dump(vec,vecpath,compress=3)
        XE={k:vec.transform(d.text) for k,d in evals.items()}
        for name in names:
            run=RUNS/f'{name}_42';run.mkdir(exist_ok=True)
            if (run/'COMPLETE').exists():continue
            t=time.perf_counter();v=vec;vpath=vecpath
            if name=='matched120k_lr':
                v=copy.deepcopy(vec);XX=v.fit_transform(train.iloc[subidx].text);yy=y[subidx]
                vpath=run/'vectorizer.joblib';joblib.dump(v,vpath,compress=3)
                XXE={k:v.transform(d.text) for k,d in evals.items()};ft=0.
            elif name=='text_position_lr':
                XX=sparse.hstack([X,sparse.csr_matrix(positions(train))],format='csr'); yy=y
                XXE={k:sparse.hstack([x,sparse.csr_matrix(positions(evals[k]))],format='csr') for k,x in XE.items()};ft=featuretime
            else:XX=X;yy=y;XXE=XE;ft=featuretime
            m=LinearSVC(C=1.,class_weight='balanced',max_iter=5000,random_state=42) if name=='tfidf_svm' else LogisticRegression(C=2.,solver='liblinear',max_iter=1000,random_state=42)
            m.fit(XX,yy);elapsed=ft+time.perf_counter()-t
            joblib.dump(m,run/'classifier.joblib',compress=3)
            def f(d):
                x=v.transform(d.text)
                if name=='text_position_lr':x=sparse.hstack([x,sparse.csr_matrix(positions(d))],format='csr')
                return m.decision_function(x)
            scores={k:m.decision_function(x) for k,x in XXE.items()}
            config={'family':'classical' if name in ['tfidf_lr','tfidf_svm'] else 'ablation','features':feature,'training_n':len(yy),'feature_fit_seconds_shared':featuretime,'C':1. if name=='tfidf_svm' else 2.,'class_weight':'balanced' if name=='tfidf_svm' else None,'vectorizer':str(vpath.relative_to(ROOT)),'solver':'LinearSVC default dual auto' if name=='tfidf_svm' else 'liblinear'}
            finalize(name,42,scores,evals,run,config,elapsed,[vpath,run/'classifier.joblib'],f)
        del X,XE,XX,XXE,vec;gc.collect()

def neural():
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader,TensorDataset
    torch.set_num_threads(THREADS);torch.set_num_interop_threads(1)
    ds=load();idx=pd.read_csv(DATA/'neural_training_rows.csv').train_row.to_numpy();train=ds['train'].iloc[idx]
    tok=NeuralTokenizer(max_vocab_size=40000,min_freq=2).fit(train.text)
    art=ROOT/'shared_artifacts';art.mkdir(exist_ok=True);joblib.dump(tok,art/'neural_tokenizer.joblib')
    arr={k:tok.transform(d.text,64) for k,d in ds.items() if k!='train'};arr['train']=tok.transform(train.text,64)
    tensors={k:torch.tensor(x,dtype=torch.long) for k,x in arr.items()}
    y=torch.tensor(train.label.to_numpy(),dtype=torch.float32)
    class Net(nn.Module):
        def __init__(self,kind,dropout):
            super().__init__();self.kind=kind;self.embedding=nn.Embedding(tok.vocab_size,64,padding_idx=0);self.dropout=nn.Dropout(dropout)
            if kind=='ffnn':self.encoder=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,64),nn.ReLU())
            else:self.rnn=nn.LSTM(64,64,batch_first=True,bidirectional=(kind=='bilstm'))
            self.classifier=nn.Linear(128 if kind=='bilstm' else 64,1)
        def forward(self,x):
            mask=x!=0;lengths=mask.sum(1).clamp_min(1);emb=self.dropout(self.embedding(x))
            if self.kind=='ffnn':z=self.encoder((emb*mask.unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1))
            else:
                packed=pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False)
                _,(h,_)=self.rnn(packed);z=torch.cat([h[-2],h[-1]],1) if self.kind=='bilstm' else h[-1];z=self.dropout(z)
            return self.classifier(z).squeeze(1)
    def predict(model,x):
        model.eval();out=[]
        with torch.inference_mode():
            for xb in x.split(512):out.append(model(xb).numpy())
        return np.concatenate(out).astype(float)
    for name,kind,dropout in [('ffnn','ffnn',.2),('lstm','lstm',0.),('bilstm','bilstm',0.),('bilstm_dropout','bilstm',.4)]:
        for seed in SEEDS:
            requested=os.environ.get('REVISION_RUN_FILTER')
            if requested and f'{name}_{seed}' not in requested.split(','):continue
            run=RUNS/f'{name}_{seed}';run.mkdir(exist_ok=True)
            if (run/'COMPLETE').exists():continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
            model=Net(kind,dropout);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
            pos_weight=torch.tensor([(len(y)-y.sum().item())/y.sum().item()]);lossfn=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loader=DataLoader(TensorDataset(tensors['train'],y),batch_size=512,shuffle=True,num_workers=0)
            best=-np.inf;state=None;stale=0;logs=[];t=time.perf_counter()
            for epoch in range(1,6):
                te=time.perf_counter();model.train();losses=[]
                for xb,yb in loader:
                    optimizer.zero_grad();loss=lossfn(model(xb),yb);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(loss.item())
                score=roc_auc_score(ds['selection'].label,predict(model,tensors['selection']))
                logs.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_auc':score,'epoch_seconds':time.perf_counter()-te})
                pd.DataFrame(logs).to_csv(run/'epochs.csv',index=False)
                print(name,seed,logs[-1],flush=True)
                if score>best:best=score;state=copy.deepcopy(model.state_dict());bestepoch=epoch;stale=0
                else:stale+=1
                if stale>=2:break
            model.load_state_dict(state);elapsed=time.perf_counter()-t;torch.save(state,run/'model.pt')
            evals={k:v for k,v in ds.items() if k!='train'};scores={k:predict(model,tensors[k]) for k in evals}
            config={'family':'neural','kind':kind,'dropout':dropout,'vocab_size':tok.vocab_size,'embedding_dim':64,'hidden_dim':64,'layers':2 if kind=='ffnn' else 1,'layer_count_scope':'feed-forward hidden layers' if kind=='ffnn' else 'recurrent layers','max_length':64,'batch_size':512,'epochs_max':5,'epochs_completed':epoch,'selected_epoch':bestepoch,'early_stopping_patience':2,'optimizer':'Adam','learning_rate':.001,'weight_decay':0.,'gradient_clip_norm':5.,'positive_weight':pos_weight.item(),'training_n':len(train),'initialization':'PyTorch defaults; manual_seed set before model construction','padding':'masked mean pooling; PAD positions excluded' if kind=='ffnn' else 'packed real token lengths; trailing PAD states excluded','torch':torch.__version__,'device':'cpu','deterministic_algorithms':True}
            f=lambda d:predict(model,torch.tensor(tok.transform(d.text,64),dtype=torch.long))
            finalize(name,seed,scores,evals,run,config,elapsed,[run/'model.pt',art/'neural_tokenizer.joblib'],f)
            del model,optimizer,state;gc.collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','classical','neural']);args=parser.parse_args()
    with threadpool_limits(limits=THREADS):globals()[args.stage]()
