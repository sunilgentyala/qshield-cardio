"""Research CLI. Accepts nonidentifying feature JSON and emits screening scores."""
import argparse, json, hashlib
from pathlib import Path
import joblib
import pandas as pd
from study import logit
ROOT=Path(__file__).resolve().parents[1]

def predict(record, trusted_manifest_sha256):
    path=ROOT/'artifacts/manifest.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=trusted_manifest_sha256:raise ValueError('Manifest integrity failed')
    manifest=json.loads(path.read_text());out={'screening_only':True,'tasks':{}}
    # Trust anchor must be provisioned independently. Hash checks do not authenticate
    # an attacker-controlled manifest. Joblib is safe only for trusted artifacts.
    for task,meta in manifest.items():
        p=ROOT/meta['path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Artifact integrity failed')
        artifact=joblib.load(p)
        df=pd.DataFrame([{k:record.get(k,float('nan')) for k in artifact['features']}])
        raw=artifact['model'].predict_proba(df)[:,1]
        prob=float(artifact['calibrator'].predict_proba(logit(raw))[0,1])
        out['tasks'][task]={'probability':prob,'screen_positive_priority':prob>=artifact['threshold']}
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--trusted-manifest-sha256',required=True);a=p.parse_args()
    print(json.dumps(predict(json.loads(Path(a.record).read_text()),a.trusted_manifest_sha256),indent=2))
