"""Run all planned experiments and write tables, plots, predictions and metadata."""
from pathlib import Path
import json, time, platform, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.impute import SimpleImputer
from data import load_arff, vector_groups, outer_split, inner_splits
from rules import scores as rule_scores
from train import tune_logistic, tune_tabm, oof_scores
from models import logistic_model, TabMClassifier, tabpfn_model
from evaluate import threshold_at_fpr, metrics, paired_group_bootstrap

ROOT=Path(__file__).parent; OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SEEDS=[11,22,33]; FPR_TARGETS=[.01,.05]

def model_fit(kind, X, y, seed, config):
    n=max(1,int(.15*len(X)))
    if kind=='logistic': return logistic_model(config).fit(X,y)
    if kind=='tabm': return TabMClassifier(**config,seed=seed).fit(X.iloc[n:],y[n:],X.iloc[:n],y[:n])
    return tabpfn_model(seed).fit(X,y)

def main():
    warnings.filterwarnings('ignore'); X,y=load_arff(ROOT/'Training Dataset.arff'); groups=vector_groups(X)
    tri,tei=outer_split(X,y,groups); Xtr,Xte=X.iloc[tri].reset_index(drop=True),X.iloc[tei].reset_index(drop=True); ytr,yte=y[tri],y[tei]; gtr,gte=groups[tri],groups[tei]
    splits=inner_splits(Xtr,ytr,gtr); assert all(not set(gtr[a])&set(gtr[b]) for a,b in splits)
    json.dump({'n_rows':len(X),'n_features':X.shape[1],'positive_phishing':int(y.sum()),'duplicates':int(len(X)-len(np.unique(groups))), 'train_rows':len(Xtr),'test_rows':len(Xte),'seed_outer':2026},open(OUT/'data_audit.json','w'),indent=2)
    rows=[]; predictions={}; fitted={}
    for seed in SEEDS:
        configs={'logistic':tune_logistic(Xtr,ytr,splits), 'tabm':tune_tabm(Xtr,ytr,splits,seed), 'tabpfn':None}
        for kind in ['logistic','tabm','tabpfn','rule']:
            start=time.perf_counter()
            if kind=='rule': oof=rule_scores(Xtr); test=rule_scores(Xte); config='fixed risk score'
            else:
                oof=oof_scores(kind,Xtr,ytr,splits,seed,configs[kind]); model=model_fit(kind,Xtr,ytr,seed,configs[kind]); test=model.predict_proba(Xte)[:,1]; fitted[(seed,kind)]=model; config=configs[kind]
            elapsed=time.perf_counter()-start; predictions[(seed,kind)]=test
            for target in FPR_TARGETS:
                threshold=threshold_at_fpr(ytr,oof,target); m=metrics(yte,test,threshold)
                rows.append({'seed':seed,'method':kind,'fpr_target':target,'threshold':threshold,'fit_and_eval_seconds':elapsed,'config':str(config),**m})
            print(f'Baigta: seed={seed}, method={kind}, {elapsed:.1f} s', flush=True)
    result=pd.DataFrame(rows); result.to_csv(OUT/'results_by_seed.csv',index=False)
    summary=result.groupby(['method','fpr_target'],as_index=False).agg({'recall':['mean','std'],'precision':['mean','std'],'fpr':['mean','std'],'ap':['mean','std'],'false_alerts_per_1000':['mean','std'],'fit_and_eval_seconds':'mean'}); summary.columns=['_'.join(c).strip('_') for c in summary.columns]; summary.to_csv(OUT/'results_summary.csv',index=False)
    # Main hypothesis: paired group bootstrap per seed at 1% FPR.
    ci=[]
    for seed in SEEDS:
        ra=result.query("seed==@seed and method=='tabm' and fpr_target==0.01").iloc[0]; rb=result.query("seed==@seed and method=='logistic' and fpr_target==0.01").iloc[0]
        low,high=paired_group_bootstrap(yte,predictions[(seed,'tabm')],ra.threshold,predictions[(seed,'logistic')],rb.threshold,gte,np.random.default_rng(seed),n=500)
        ci.append({'seed':seed,'tabm_minus_logistic_recall':ra.recall-rb.recall,'ci95_low':low,'ci95_high':high,'both_fpr_le_1pct':bool(ra.fpr<=.01 and rb.fpr<=.01)})
    ci=pd.DataFrame(ci); ci.to_csv(OUT/'hypothesis_bootstrap.csv',index=False)
    # Figures: direct method comparison, PR curves and confusion matrices (seed 11, 1% target).
    sns.set_theme(style='whitegrid'); s=result.query('fpr_target==0.01').copy(); s['method']=s.method.replace({'tabm':'TabM','tabpfn':'TabPFN','logistic':'Logistinė regresija','rule':'Rizikos taisyklė'})
    fig,ax=plt.subplots(figsize=(10,5)); sns.barplot(s,x='method',y='recall',errorbar='sd',ax=ax,palette='deep'); ax.set(xlabel='',ylabel='Recall (FPR tikslas 1 %)',ylim=(0,1)); fig.tight_layout(); fig.savefig(OUT/'recall_comparison.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6));
    for kind,label in [('tabm','TabM'),('tabpfn','TabPFN'),('logistic','Logistinė regresija'),('rule','Rizikos taisyklė')]:
        p,r,_=precision_recall_curve(yte,predictions[(11,kind)]); ax.plot(r,p,label=f'{label} (AP={metrics(yte,predictions[(11,kind)],0.5)["ap"]:.3f})')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(OUT/'pr_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.5));
    for ax,(kind,label) in zip(axes,[('tabm','TabM'),('tabpfn','TabPFN'),('logistic','Logistinė regresija'),('rule','Rizikos taisyklė')]):
        rr=result.query("seed==11 and method==@kind and fpr_target==0.01").iloc[0]; cm=np.array([[rr.tn,rr.fp],[rr.fn,rr.tp]])
        sns.heatmap(cm,annot=True,fmt='g',cbar=False,cmap='Blues',ax=ax);ax.set(title=label,xlabel='Prognozė (0,1)',ylabel='Tikroji (0,1)')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=180);plt.close(fig)
    # Error examples and model-specific permutation importances.
    errors=[]
    for kind in ['tabm','tabpfn','logistic','rule']:
        rr=result.query("seed==11 and method==@kind and fpr_target==0.01").iloc[0]; pred=predictions[(11,kind)]>=rr.threshold
        for typ,ix in [('FP',np.where((yte==0)&pred)[0][:5]),('FN',np.where((yte==1)&~pred)[0][:5])]:
            for i in ix: errors.append({'method':kind,'error':typ,'test_row':int(i),'score':predictions[(11,kind)][i],**Xte.iloc[i].to_dict()})
    pd.DataFrame(errors).to_csv(OUT/'error_examples.csv',index=False)
    importances=[]
    for kind in ['logistic']:
        model=fitted[(11,kind)]
        pi=permutation_importance(model,Xte,yte,n_repeats=5,random_state=2026,scoring='average_precision')
        for n,v in zip(X.columns,pi.importances_mean): importances.append({'method':kind,'feature':n,'importance_ap_drop':v})
    pd.DataFrame(importances).to_csv(OUT/'permutation_importance.csv',index=False)
    pd.DataFrame(importances).pivot(index='feature',columns='method',values='importance_ap_drop').sort_values('logistic').tail(15).plot.barh(figsize=(9,7));plt.xlabel('AP sumažėjimas po permutacijos');plt.tight_layout();plt.savefig(OUT/'permutation_importance.png',dpi=180);plt.close()
    json.dump({'python':platform.python_version(),'platform':platform.platform(),'seeds':SEEDS,'outer_seed':2026,'notes':'TabPFN requires its official pretrained weights on first run.'},open(OUT/'reproducibility.json','w'),indent=2)
    print('Completed. Outputs:',OUT)
if __name__=='__main__': main()
