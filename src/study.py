"""Reproducible retrospective screening experiment. Not clinical software."""
from pathlib import Path
import json, hashlib, urllib.request, platform, importlib.metadata, joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['age','male','bmi','waist','current_smoker','vigorous_recreation']
SEED=20261015

def fetch_data():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for year,suffix in [(2015,'I'),(2017,'J')]:
        for module in ['DEMO','BMX','BPX','GHB','DIQ','BPQ','SMQ','PAQ']:
            name=f'{module}_{suffix}.xpt';p=raw/name
            url=f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}'
            if not p.exists():
                with urllib.request.urlopen(url,timeout=60) as response: content=response.read()
                if not content.startswith(b'HEADER RECORD'):raise ValueError('Not an XPT: '+name)
                p.write_bytes(content)
            manifest.append(dict(file=name,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    (ROOT/'data/sources.json').write_text(json.dumps(manifest,indent=2))

def cohort(suffix):
    d=pd.read_sas(ROOT/f'data/raw/DEMO_{suffix}.xpt')
    for module in ['BMX','BPX','GHB','DIQ','BPQ','SMQ','PAQ']:
        t=pd.read_sas(ROOT/f'data/raw/{module}_{suffix}.xpt')
        assert t.SEQN.is_unique
        d=d.merge(t,on='SEQN',how='left',validate='one_to_one')
    flow={'all':len(d)}
    d=d[d.RIDAGEYR>=20].copy();flow['age_20_plus']=len(d)
    d=d[d.RIDEXPRG!=1].copy();flow['no_reported_pregnancy']=len(d)
    d=d[d.DIQ010.isin([2,3]) & (d.BPQ020==2) & (d.BPQ050A!=1) & (d.DIQ050!=1) & (d.DIQ070!=1)].copy()
    flow['no_known_diabetes_or_hypertension']=len(d)
    sys=d[[f'BPXSY{i}' for i in [1,2,3]]].to_numpy()
    dia=d[[f'BPXDI{i}' for i in [1,2,3]]].to_numpy()
    # Zero diastolic values are permitted by the NHANES codebook.
    valid=np.isfinite(sys)&np.isfinite(dia)&(sys>dia)&(dia>=0)
    d['bp_pairs']=valid.sum(axis=1)
    d['sbp']=np.divide(np.where(valid,sys,0).sum(axis=1),valid.sum(axis=1),out=np.full(len(d),np.nan),where=valid.sum(axis=1)>0)
    d['dbp']=np.divide(np.where(valid,dia,0).sum(axis=1),valid.sum(axis=1),out=np.full(len(d),np.nan),where=valid.sum(axis=1)>0)
    d=d[(d.bp_pairs>=2)&d.LBXGH.notna()].copy()
    flow['complete_screening_endpoints']=len(d)
    d['diabetes']=(d.LBXGH>=6.5).astype(int)
    d['hypertension']=((d.sbp>=130)|(d.dbp>=80)).astype(int)
    d['hypertension_140']=((d.sbp>=140)|(d.dbp>=90)).astype(int)
    d['age']=d.RIDAGEYR;d['male']=d.RIAGENDR.map({1:1,2:0})
    d['bmi']=d.BMXBMI;d['waist']=d.BMXWAIST
    d['current_smoker']=np.where(d.SMQ020==2,0,np.where(d.SMQ040.isin([1,2]),1,np.where(d.SMQ040==3,0,np.nan)))
    d['vigorous_recreation']=d.PAQ650.map({1:1,2:0})
    return d.reset_index(drop=True),flow

def metrics(y,p,threshold,weight=None):
    y=np.asarray(y);p=np.asarray(p);a=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,a,labels=[0,1],sample_weight=weight).ravel()
    def div(a,b):return float(a/b) if b else None
    return dict(auroc=float(roc_auc_score(y,p,sample_weight=weight)),ap=float(average_precision_score(y,p,sample_weight=weight)),
        brier=float(brier_score_loss(y,p,sample_weight=weight)),sensitivity=div(tp,tp+fn),specificity=div(tn,tn+fp),
        ppv=div(tp,tp+fp),npv=div(tn,tn+fn),referral=float(np.average(a,weights=weight)),threshold=float(threshold),
        tp=float(tp),fp=float(fp),tn=float(tn),fn=float(fn))

def threshold80(y,p):
    # Highest observed positive score retaining at least ceil(0.8*n_positive) positives.
    positive=np.sort(np.asarray(p)[np.asarray(y)==1])[::-1]
    return float(positive[int(np.ceil(.8*len(positive)))-1])

def bootstrap(y,p,threshold,n=1000):
    rng=np.random.default_rng(SEED);y=np.asarray(y);p=np.asarray(p);v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if np.unique(y[i]).size<2:continue
        m=metrics(y[i],p[i],threshold);v.append([m[k] for k in ['auroc','ap','sensitivity','specificity','ppv']])
    return {k:np.quantile(np.asarray(v)[:,j],[.025,.975]).tolist() for j,k in enumerate(['auroc','ap','sensitivity','specificity','ppv'])}

def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)

def run():
    fetch_data();dev,flow1=cohort('I');test,flow2=cohort('J')
    for name in ['results','figures']:(ROOT/name).mkdir(exist_ok=True)
    strata=dev.diabetes*2+dev.hypertension
    train,other=train_test_split(np.arange(len(dev)),test_size=.4,random_state=SEED,stratify=strata)
    cal,val=train_test_split(other,test_size=.5,random_state=SEED,stratify=strata.iloc[other])
    assert not set(dev.SEQN)&set(test.SEQN)
    summary={'seed':SEED,'flow':{'2015_2016':flow1,'2017_2018':flow2},'split':{},'models':{},'subgroups':[],
        'feature_missingness':{},'descriptive':{},'environment':{'python':platform.python_version(),'platform':platform.platform()}}
    for name,d in [('development',dev),('test',test)]:
        summary['feature_missingness'][name]={k:int(d[k].isna().sum()) for k in FEATURES}
        summary['descriptive'][name]={'n':len(d),'age_mean':float(d.age.mean()),'male_n':int(d.male.sum()),'bmi_mean':float(d.bmi.mean()),
            'diabetes_n':int(d.diabetes.sum()),'hypertension_n':int(d.hypertension.sum()),'both_n':int((d.diabetes&d.hypertension).sum())}
    for name,idx in [('training',train),('calibration',cal),('validation',val)]:
        summary['split'][name]={'n':len(idx),'diabetes_n':int(dev.iloc[idx].diabetes.sum()),'hypertension_n':int(dev.iloc[idx].hypertension.sum())}
    for name in ['numpy','pandas','scikit-learn','scipy','cryptography','pqcrypto','matplotlib']:
        summary['environment'][name]=importlib.metadata.version(name)
    preds={};selected={};rows=[];artifact_manifest={}
    for task in ['diabetes','hypertension']:
        fitted={};task_result={}
        for model_name,features in [('LR_core',FEATURES[:4]),('LR_full',FEATURES),('HGB_full',FEATURES)]:
            if model_name.startswith('LR'):
                model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=SEED))
            else:
                model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=1.,random_state=SEED))
            model.fit(dev.iloc[train][features],dev.iloc[train][task])
            raw_cal=model.predict_proba(dev.iloc[cal][features])[:,1]
            calibrator=LogisticRegression(C=1e6,max_iter=2000).fit(logit(raw_cal),dev.iloc[cal][task])
            def predict(d,model=model,c=calibrator,features=features):return c.predict_proba(logit(model.predict_proba(d[features])[:,1]))[:,1]
            pv=predict(dev.iloc[val]);threshold=threshold80(dev.iloc[val][task],pv)
            pt=predict(test);m=metrics(test[task],pt,threshold)
            task_result[model_name]={'validation':metrics(dev.iloc[val][task],pv,threshold),'test':m,'weighted_test':metrics(test[task],pt,threshold,test.WTMEC2YR)}
            preds[(task,model_name)]=pt;fitted[model_name]=(predict,model,calibrator,features)
            rows.append({'task':task,'model':model_name,**m})
        # Selection uses only the development validation partition.
        best=max(task_result,key=lambda k:task_result[k]['validation']['auroc'])
        selected[task]=best;pt=preds[(task,best)];threshold=task_result[best]['test']['threshold']
        summary['models'][task]={'selected':best,'candidates':task_result,'ci':bootstrap(test[task],pt,threshold)}
        predict,model,c,features=fitted[best]
        artifact_dir=ROOT/'artifacts';artifact_dir.mkdir(exist_ok=True)
        artifact_path=artifact_dir/f'{task}.joblib'
        joblib.dump({'model':model,'calibrator':c,'features':features,'threshold':threshold,'task':task},artifact_path)
        artifact_manifest[task]={'path':str(artifact_path.relative_to(ROOT)),'sha256':hashlib.sha256(artifact_path.read_bytes()).hexdigest(),'features':features,'threshold':threshold,'selected':best}
        rng=np.random.default_rng(SEED)
        stressed=test.copy();mask=rng.random((len(test),len(features)))<.2
        for j,f in enumerate(features):stressed[f]=stressed[f].astype(float).mask(mask[:,j])
        summary['models'][task]['missing20_test']=metrics(test[task],predict(stressed),threshold)
        # Importance is descriptive, evaluated on holdout; no subsequent tuning.
        importance={}
        for f in features:
            vals=[]
            for _ in range(20):
                perm=test.copy();perm[f]=rng.permutation(perm[f].to_numpy())
                vals.append(roc_auc_score(test[task],pt)-roc_auc_score(test[task],predict(perm)))
            importance[f]={'mean_auroc_drop':float(np.mean(vals)),'sd':float(np.std(vals))}
        summary['models'][task]['permutation_importance']=importance
        for group,mask in [('female',test.male==0),('male',test.male==1),('age_under_50',test.age<50),('age_50_plus',test.age>=50)]+[(f'RIDRETH3_{int(v)}',test.RIDRETH3==v) for v in sorted(test.RIDRETH3.unique())]:
            yy=test.loc[mask,task]
            if yy.nunique()<2:continue
            result={'task':task,'group':group,'n':int(mask.sum()),'positive_n':int(yy.sum()),**metrics(yy,pt[mask],threshold)}
            result['sensitivity_ci']=bootstrap(yy,pt[mask],threshold,n=300)['sensitivity']
            summary['subgroups'].append(result)
        # Sensitivity to a different BP endpoint, evaluated without retuning.
        if task=='hypertension':summary['models'][task]['bp140_test']=metrics(test.hypertension_140,pt,threshold)
    ad=preds[('diabetes',selected['diabetes'])]>=summary['models']['diabetes']['candidates'][selected['diabetes']]['test']['threshold']
    ah=preds[('hypertension',selected['hypertension'])]>=summary['models']['hypertension']['candidates'][selected['hypertension']]['test']['threshold']
    yy=(test.diabetes|test.hypertension).to_numpy().astype(bool);aa=ad|ah
    summary['joint_referral']={'n':len(test),'referred_n':int(aa.sum()),'positive_either_n':int(yy.sum()),'tp':int((aa&yy).sum()),'fp':int((aa&~yy).sum()),'fn':int((~aa&yy).sum()),'tn':int((~aa&~yy).sum()),'both_positive_n':int((test.diabetes&test.hypertension).sum()),'both_captured_n':int((aa&(test.diabetes&test.hypertension).astype(bool)).sum())}
    pd.DataFrame(rows).to_csv(ROOT/'results/model_metrics.csv',index=False)
    pd.DataFrame(summary['subgroups']).to_csv(ROOT/'results/subgroup_metrics.csv',index=False)
    splits=[]
    for name,idx in [('training',train),('calibration',cal),('validation',val)]:
        splits.extend([{'SEQN':int(v),'cycle':'2015_2016','partition':name} for v in dev.iloc[idx].SEQN])
    splits.extend([{'SEQN':int(v),'cycle':'2017_2018','partition':'test'} for v in test.SEQN])
    pd.DataFrame(splits).to_csv(ROOT/'results/split_manifest.csv',index=False)
    out=test[['SEQN','diabetes','hypertension','hypertension_140','WTMEC2YR']].copy()
    for (task,model),p in preds.items():out[f'{task}_{model}']=p
    out.to_csv(ROOT/'results/test_predictions.csv',index=False)
    (ROOT/'results/study.json').write_text(json.dumps(summary,indent=2))
    (ROOT/'artifacts/manifest.json').write_text(json.dumps(artifact_manifest,indent=2))
    plots(test,preds,summary)
    print(json.dumps({'cohort':summary['descriptive'],'split':summary['split'],'selected':{k:{'model':v['selected'],'test':v['candidates'][v['selected']]['test'],'ci':v['ci']} for k,v in summary['models'].items()}},indent=2))

def plots(test,preds,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    from sklearn.metrics import roc_curve,precision_recall_curve
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for col,task in enumerate(['diabetes','hypertension']):
        for model,color in zip(['LR_core','LR_full','HGB_full'],['#6f7f8f','#176b8e','#ab4936']):
            p=preds[(task,model)];fpr,tpr,_=roc_curve(test[task],p)
            axs[0,col].plot(fpr,tpr,label=f'{model} AUC {roc_auc_score(test[task],p):.3f}',color=color)
            precision,recall,_=precision_recall_curve(test[task],p)
            axs[1,col].plot(recall,precision,color=color,label=model)
        axs[0,col].plot([0,1],[0,1],'--',color='gray',lw=.8)
        axs[1,col].axhline(test[task].mean(),ls='--',color='gray',lw=.8,label='Sample prevalence')
        axs[0,col].set(title=task.capitalize()+' screening endpoint',xlabel='False positive rate',ylabel='Sensitivity')
        axs[1,col].set(xlabel='Recall',ylabel='Precision')
        for ax in axs[:,col]:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.suptitle('Temporal holdout performance  NHANES 2017–2018',fontsize=13)
    fig.tight_layout();fig.savefig(ROOT/'figures/02_discrimination.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axs,['diabetes','hypertension']):
        model=summary['models'][task]['selected'];p=preds[(task,model)]
        observed,expected=calibration_curve(test[task],p,n_bins=8,strategy='quantile')
        ax.plot(expected,observed,'o-',color='#176b8e',label=model);ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(title=task.capitalize(),xlabel='Mean predicted probability',ylabel='Observed fraction positive',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.suptitle('Calibration on the temporal holdout  eight equal-count bins')
    fig.tight_layout();fig.savefig(ROOT/'figures/03_calibration.png',dpi=190);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axs,['diabetes','hypertension']):
        p=preds[(task,summary['models'][task]['selected'])];y=test[task].to_numpy();prev=y.mean();grid=np.linspace(.01,.5,100)
        benefit=[]
        for t in grid:
            a=p>=t;benefit.append((a&y.astype(bool)).mean()-(a&~y.astype(bool)).mean()*t/(1-t))
        ax.plot(grid,benefit,label='Model',color='#176b8e');ax.plot(grid,prev-(1-prev)*grid/(1-grid),ls='--',label='Refer all',color='#ab4936');ax.axhline(0,color='gray',label='Refer none')
        ax.set(title=task.capitalize(),xlabel='Assumed referral threshold probability',ylabel='Net benefit per person',ylim=(-.05,max(.05,prev*1.1)));ax.legend(fontsize=8)
    fig.suptitle('Exploratory decision curves  thresholds are hypothetical utilities')
    fig.tight_layout();fig.savefig(ROOT/'figures/04_decision_curves.png',dpi=190);plt.close(fig)

if __name__=='__main__':run()
