"""Load a trained revision model and reproduce scores from a sentence CSV.
Usage: python predict_saved_model.py ROOT RUN INPUT.csv OUTPUT.csv
ROOT contains runs/ and shared_artifacts/. Neural and classical files are trusted
author-generated archives; do not unpickle unrelated downloaded artifacts.
"""
from pathlib import Path
from evidence_io import read_csv
import json,sys
import numpy as np,pandas as pd,joblib
from scipy import sparse
from scipy.special import expit
sys.path.insert(0,str(Path(__file__).resolve().parent/'legacy_src'))
ARTIFACT_CACHE={}

def predict(root,runname,data):
    run=root/'runs'/runname;key=str(run.resolve())
    if key not in ARTIFACT_CACHE:
        ARTIFACT_CACHE[key]={'config':json.loads((run/'config.json').read_text()),'calibrator':json.loads((run/'calibrator.json').read_text())}
    cache=ARTIFACT_CACHE[key];cfg=cache['config'];cal=cache['calibrator']
    if cfg['family']=='neural':
        import torch
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence
        torch.set_num_threads(4)
        if 'tokenizer' not in cache:cache['tokenizer']=joblib.load(root/'shared_artifacts/neural_tokenizer.joblib')
        tok=cache['tokenizer']
        class Net(nn.Module):
            def __init__(self):
                super().__init__();self.kind=cfg['kind'];self.embedding=nn.Embedding(cfg['vocab_size'],64,padding_idx=0);self.dropout=nn.Dropout(cfg['dropout'])
                if self.kind=='ffnn':self.encoder=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(cfg['dropout']),nn.Linear(64,64),nn.ReLU())
                else:self.rnn=nn.LSTM(64,64,batch_first=True,bidirectional=self.kind=='bilstm')
                self.classifier=nn.Linear(128 if self.kind=='bilstm' else 64,1)
            def forward(self,x):
                mask=x!=0;lengths=mask.sum(1).clamp_min(1);emb=self.dropout(self.embedding(x))
                if self.kind=='ffnn':z=self.encoder((emb*mask.unsqueeze(-1)).sum(1)/lengths.unsqueeze(-1))
                else:
                    _,(h,_)=self.rnn(pack_padded_sequence(emb,lengths.cpu(),batch_first=True,enforce_sorted=False));z=torch.cat([h[-2],h[-1]],1) if self.kind=='bilstm' else h[-1];z=self.dropout(z)
                return self.classifier(z).squeeze(1)
        if 'model' not in cache:
            model=Net();model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True));model.eval();cache['model']=model
        model=cache['model'];x=torch.tensor(tok.transform(data.text,64),dtype=torch.long)
        with torch.inference_mode():raw=np.concatenate([model(batch).numpy() for batch in x.split(512)]).astype(float)
    elif cfg['family']=='baseline':
        if 'prior' not in cache:cache['prior']=json.loads((run/'prior.json').read_text())['prior']
        raw=np.full(len(data),cache['prior'])
    else:
        if 'model' not in cache:cache['model']=joblib.load(run/'classifier.joblib')
        model=cache['model']
        if runname.startswith('position_lr'):x=np.column_stack([data.position,data.position**2,data.position**3])
        else:
            if 'vectorizer' not in cache:cache['vectorizer']=joblib.load(root/Path(cfg['vectorizer'].replace('\\','/')))
            vectorizer=cache['vectorizer'];x=vectorizer.transform(data.text)
            if runname.startswith('text_position'):x=sparse.hstack([x,np.column_stack([data.position,data.position**2,data.position**3])],format='csr')
        raw=model.decision_function(x)
    if cal['kind']=='temperature':prob=expit(raw/cal['temperature'])
    elif cal['kind']=='sigmoid':prob=expit(raw*cal['coefficient']+cal['intercept'])
    else:prob=raw
    return raw,prob,(prob>=cfg['threshold']).astype(int)

if __name__=='__main__':
    root=Path(sys.argv[1]);data=read_csv(sys.argv[3],dtype={'abstract_id':str});raw,prob,pred=predict(root,sys.argv[2],data)
    data['raw_score']=raw;data['probability']=prob;data['prediction']=pred;data.to_csv(sys.argv[4],index=False)
