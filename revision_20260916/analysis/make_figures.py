"""All figures are deterministic plots of revised data, not generated illustrations."""
from pathlib import Path
from evidence_io import read_csv
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];T=ROOT/'tables';F=ROOT/'figures';F.mkdir(exist_ok=True)
selected=json.loads((T/'selected_model.json').read_text())['model']
labels={'majority':'Prior baseline','tfidf_lr':'TF-IDF logistic regression','tfidf_svm':'TF-IDF linear SVM','ffnn':'Average-embedding network','lstm':'LSTM','bilstm':'BiLSTM','bilstm_dropout':'BiLSTM with dropout','position_lr':'Position only','unigram_lr':'Word unigrams','word12_lr':'Word unigrams + bigrams','text_position_lr':'Word + character + position','matched120k_lr':'Word + character (120k)'}
C=['#0072B2','#D55E00'];S=['Internal test','Cross-corpus test']
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.6,'savefig.facecolor':'white','pdf.fonttype':42})
ci=read_csv(T/'cluster_intervals.csv');metrics=read_csv(T/'all_metrics.csv');font_checks={}
def export(fig,num):
    fig.canvas.draw()
    text_artists=[a for a in fig.findobj(match=matplotlib.text.Text) if a.get_visible() and a.get_text().strip()]
    sizes=[a.get_fontsize() for a in text_artists]
    families=sorted(set(f for a in text_artists for f in a.get_fontfamily()))
    assert min(sizes)>=8 and max(sizes)<=12,(num,sizes)
    assert families==['Arial'],(num,families)
    font_checks[num]={'min_font_pt':min(sizes),'max_font_pt':max(sizes),'font_families':families}
    fig.savefig(F/f'Fig{num}.pdf',bbox_inches='tight',pad_inches=.10)
    fig.savefig(F/f'Fig{num}.png',dpi=300,bbox_inches='tight',pad_inches=.10)
    # RGB TIFF with lossless compression, no alpha channel, 600 dpi.
    import io
    b=io.BytesIO();fig.savefig(b,format='png',dpi=600,bbox_inches='tight',pad_inches=.10);b.seek(0)
    Image.open(b).convert('RGB').save(F/f'Fig{num}.tif',compression='tiff_lzw',dpi=(600,600))
    plt.close(fig)
def aucrow(model,split):return ci[(ci.run==model+'_42')&(ci.split==split)&(ci.metric=='roc_auc')].iloc[0]

d=read_csv(T/'dataset_counts.csv').set_index('split').loc[['train','selection','calibration','threshold','test','external']]
fig,ax=plt.subplots(figsize=(7.0,3.2));x=np.arange(len(d));ax.bar(x,d.negative,color='#999999',label='Other');ax.bar(x,d.positive,bottom=d.negative,color=C[0],label='Findings/conclusion')
ax.set_xticks(x,['Training','Model selection','Calibration','Threshold','Internal test','Cross-corpus']);ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Sentences');ax.ticklabel_format(style='plain',axis='y');ax.legend(loc='upper right')
for i,(_,r) in enumerate(d.iterrows()):ax.text(i,r.sentences+3500,f'{int(r.abstracts):,} abstracts',ha='center',fontsize=8)
ax.set_ylim(0,d.sentences.max()*1.18);fig.tight_layout();export(fig,1)

models=['majority','tfidf_lr','tfidf_svm','ffnn','lstm','bilstm','bilstm_dropout']
fig,ax=plt.subplots(figsize=(7.0,3.8));y=np.arange(len(models))
for j,split in enumerate(['test','external']):
    rr=[aucrow(m,split) for m in models];v=np.array([r.estimate for r in rr]);lo=np.array([r.ci_low for r in rr]);hi=np.array([r.ci_high for r in rr]);ax.errorbar(v,y+(j-.5)*.18,xerr=[v-lo,hi-v],fmt=['o','s'][j],color=C[j],capsize=2,label=S[j])
ax.set_yticks(y,[labels[m] for m in models]);ax.invert_yaxis();ax.set_xlabel('ROC AUC with 95% abstract-bootstrap interval');ax.set_xlim(.48,1.);ax.legend(loc='lower left');ax.grid(axis='x',alpha=.2);fig.tight_layout();export(fig,2)

pred={split:read_csv(ROOT/'runs'/f'{selected}_42'/f'predictions_{split}.csv') for split in ['test','external']}
fig,axs=plt.subplots(1,2,figsize=(7.0,3.1))
for j,(split,d) in enumerate(pred.items()):
    fpr,tpr,_=roc_curve(d.label,d.probability);prec,rec,_=precision_recall_curve(d.label,d.probability)
    axs[0].plot(fpr,tpr,color=C[j],linestyle=['-','--'][j],label=S[j]);axs[1].plot(rec,prec,color=C[j],linestyle=['-','--'][j],label=S[j])
axs[0].plot([0,1],[0,1],color='0.6',lw=.8,ls=':');axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='A  ROC curves');axs[1].set(xlabel='Recall',ylabel='Precision',title='B  Precision–recall curves')
for ax in axs:ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.legend(loc='lower left');ax.grid(alpha=.15)
fig.tight_layout();export(fig,3)

curves=read_csv(T/'calibration_curve_intervals.csv');fig,axs=plt.subplots(1,2,figsize=(7.0,3.2))
for i,split in enumerate(['test','external']):
    ax=axs[i]
    for j,kind in enumerate(['pre','post']):
        d=curves[(curves.split==split)&(curves.probability==kind)];ax.errorbar(d.mean_probability,d.observed_fraction,yerr=[np.maximum(0,d.observed_fraction-d.ci_low),np.maximum(0,d.ci_high-d.observed_fraction)],fmt=['o-','s--'][j],ms=3,color=C[j],capsize=2,label=['Before calibration','After calibration'][j])
    ax.plot([0,1],[0,1],':',color='.5',lw=.8);ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=('A  ' if i==0 else 'B  ')+S[i],xlim=(0,1),ylim=(0,1));ax.legend(loc='upper left');ax.grid(alpha=.15)
fig.tight_layout();export(fig,4)

ab=['position_lr','unigram_lr','word12_lr','tfidf_lr','text_position_lr','matched120k_lr'];fig,ax=plt.subplots(figsize=(7.0,3.7));y=np.arange(len(ab))
for j,split in enumerate(['test','external']):
    rr=[aucrow(m,split) for m in ab];v=np.array([r.estimate for r in rr]);lo=np.array([r.ci_low for r in rr]);hi=np.array([r.ci_high for r in rr]);ax.errorbar(v,y+(j-.5)*.18,xerr=[v-lo,hi-v],fmt=['o','s'][j],color=C[j],capsize=2,label=S[j])
ax.set_yticks(y,[labels[m] for m in ab]);ax.invert_yaxis();ax.set_xlabel('ROC AUC with 95% abstract-bootstrap interval');ax.legend(loc='lower left');ax.grid(axis='x',alpha=.2);fig.tight_layout();export(fig,5)

fig,axs=plt.subplots(1,2,figsize=(6.7,3.1))
for j,(split,d) in enumerate(pred.items()):
    cm=confusion_matrix(d.label,d.prediction,labels=[0,1]);ax=axs[j];ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.flat)*1.1)
    for (a,b),n in np.ndenumerate(cm):ax.text(b,a,f'{n:,}',ha='center',va='center',fontsize=12,color='white' if n>cm.max()*.65 else 'black')
    ax.set_xticks([0,1],['Other','Findings/conclusion'],fontsize=8);ax.set_yticks([0,1],['Other','Findings/conclusion'],fontsize=8);ax.set(xlabel='Predicted class',ylabel='Source-derived class',title=('A  ' if j==0 else 'B  ')+S[j]);ax.spines[['top','right']].set_visible(True)
fig.tight_layout();export(fig,6)

sub=read_csv(T/'length_subgroups.csv');cut=json.loads((T/'statistics_manifest.json').read_text())['length_cutoffs_training_words'];fig,ax=plt.subplots(figsize=(6.7,3.1))
for j,split in enumerate(['test','external']):
    d=sub[(sub.split==split)&(sub.metric=='roc_auc')];ax.errorbar(d.quartile+(j-.5)*.06,d.estimate,yerr=[d.estimate-d.ci_low,d.ci_high-d.estimate],fmt=['o-','s--'][j],color=C[j],capsize=3,label=S[j])
ax.set_xticks([1,2,3,4],[f'≤{cut[0]:g}',f'>{cut[0]:g}–{cut[1]:g}',f'>{cut[1]:g}–{cut[2]:g}',f'>{cut[2]:g}']);ax.set(xlabel='Words per sentence (training-derived cutoffs)',ylabel='ROC AUC');ax.legend(loc='lower right');ax.grid(alpha=.2);fig.tight_layout();export(fig,7)

check=[]
for p in F.glob('*.tif'):
    im=Image.open(p);dpi=tuple(float(x) for x in im.info.get('dpi'));check.append({'file':p.name,'width_px':im.width,'height_px':im.height,'dpi':dpi,'mode':im.mode,'bytes':p.stat().st_size,'width_inches':im.width/dpi[0],'height_inches':im.height/dpi[1],**font_checks[int(p.stem.removeprefix('Fig'))]})
(F/'figure_validation.json').write_text(json.dumps(check,indent=2))
print('Generated and measured seven RGB TIFF figures, with PDF and PNG working copies.')
