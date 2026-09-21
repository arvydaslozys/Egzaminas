"""Ablation and missing-data robustness tests, isolated from main experiment."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_arff, vector_groups, outer_split, inner_splits
from models import logistic_model, TabMClassifier, tabpfn_model
from rules import scores as rule_scores
from evaluate import threshold_at_fpr, metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).parent
URL=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol','Prefix_Suffix']
SEED=11

def mask_test(A, rate):
    rng=np.random.default_rng(2026+round(rate*100)); B=A.copy(); mask=rng.random(B.shape)<rate
    # Training-only modal imputation is applied consistently before model inference.
    for j,c in enumerate(B.columns): B.loc[mask[:,j],c]=A[c].mode().iloc[0]
    return B

def fit_predict(kind, A, y, fit, cal, test, test_X):
    if kind=='rule': return rule_scores(A.iloc[cal]), rule_scores(test_X), 'fiksuota taisyklė'
    if kind=='logistic':
        # Column count is scenario-dependent (30 or the 5-feature URL subset).
        m=Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore')),('model',LogisticRegression(C=1,max_iter=3000,solver='liblinear'))]).fit(A.iloc[fit],y[fit])
        return m.predict_proba(A.iloc[cal])[:,1],m.predict_proba(test_X)[:,1],'C=1'
    if kind=='tabm':
        early=fit[:max(1,len(fit)//6)]; train=fit[max(1,len(fit)//6):]
        m=TabMClassifier(width=128,depth=2,lr=1e-3,weight_decay=1e-4,dropout=.1,seed=SEED,epochs=30,patience=6).fit(A.iloc[train],y[train],A.iloc[early],y[early])
        return m.predict_proba(A.iloc[cal])[:,1],m.predict_proba(test_X)[:,1],'TabM 128x2'
    m=tabpfn_model(SEED).fit(A.iloc[fit],y[fit])
    return m.predict_proba(A.iloc[cal])[:,1],m.predict_proba(test_X)[:,1],'TabPFN default'

def main():
    # TabPFN uses the working directory to resolve its locally cached checkpoint.
    os.chdir(ROOT)
    X,y=load_arff(ROOT/'Training Dataset.arff'); g=vector_groups(X); tri,tei=outer_split(X,y,g)
    A=X.iloc[tri].reset_index(drop=True); T=X.iloc[tei].reset_index(drop=True); yt=y[tri]; ye=y[tei]; gt=g[tri]
    fit,cal=inner_splits(A,yt,gt)[0]
    scenarios=[('abliacija visi 30 požymių',list(A.columns),0),('abliacija URL pogrupis',URL,0),('trūksta 10 procentų',list(A.columns),.10),('trūksta 20 procentų',list(A.columns),.20)]
    rows=[]
    for name,cols,rate in scenarios:
        Z=A[cols]; Q=T[cols] if rate==0 else mask_test(T[cols],rate)
        for kind in ['logistic','tabm','tabpfn','rule']:
            if kind=='rule' and not set(URL).issubset(cols): continue
            cal_p,test_p,config=fit_predict(kind,Z,yt,fit,cal,T,Q)
            threshold=threshold_at_fpr(yt[cal],cal_p,.01)
            rows.append({'scenario':name,'method':kind,'missing_rate':rate,'threshold':threshold,'config':config,**metrics(ye,test_p,threshold)})
            print(f'Baigta: {name}; {kind}',flush=True)
    r=pd.DataFrame(rows); r.to_csv(OUT/'abliacija_ir_atsparumas.csv',index=False)
    sns.set_theme(style='whitegrid'); names={'logistic':'Logistinė regresija','tabm':'TabM','tabpfn':'TabPFN','rule':'Rizikos taisyklė'};r['Metodas']=r.method.map(names)
    for metric,label,file in [('recall','Recall, FPR tikslas 1 %','abliacija_recall.png'),('ap','AP','abliacija_ap.png')]:
        fig,ax=plt.subplots(figsize=(10,5));sns.barplot(r,x='scenario',y=metric,hue='Metodas',ax=ax);ax.set(xlabel='',ylabel=label);ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(OUT/file,dpi=180);plt.close(fig)
if __name__=='__main__': main()
