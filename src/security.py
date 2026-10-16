"""Application-envelope research harness, not an interoperable TLS protocol."""
from pathlib import Path
import json, os, time, hashlib, platform, importlib.metadata
from dataclasses import dataclass, field
import numpy as np
from pqcrypto.kem import ml_kem_768 as kem
from pqcrypto.sign import ml_dsa_44 as dsa
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
ROOT=Path(__file__).resolve().parents[1]
PROFILE='X25519+ML-KEM-768/HKDF-SHA256/AES-256-GCM/ML-DSA-44'
MANIFEST=ROOT/'artifacts/manifest.json'
MODEL_DIGEST=hashlib.sha256(MANIFEST.read_bytes()).hexdigest() if MANIFEST.exists() else 'model-sha256-test'

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def flip(b):return bytes([b[0]^1])+b[1:]
def derive(pq,classic,transcript):
    return HKDF(algorithm=hashes.SHA256(),length=32,salt=hashlib.sha256(transcript).digest(),info=b'QShield-Cardio-envelope-v1').derive(pq+classic)

@dataclass
class Receiver:
    pqpk:bytes=field(init=False);pqsk:bytes=field(init=False)
    xsk:object=field(default_factory=X25519PrivateKey.generate)
    spk:bytes=b''
    expected_model:str=MODEL_DIGEST
    seen:set=field(default_factory=set)
    def __post_init__(self):self.pqpk,self.pqsk=kem.keygen()
    @property
    def xpk(self):return self.xsk.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    def receive(self,env,now=1000,role='clinician'):
        if role not in {'clinician','research_reader'}:raise ValueError('unauthorized role')
        signed={k:v for k,v in env.items() if k!='signature'}
        dsa.verify(self.spk,canonical(signed),bytes.fromhex(env['signature']))
        meta=env['metadata']
        if meta['profile']!=PROFILE:raise ValueError('profile mismatch')
        if meta['model']!=self.expected_model:raise ValueError('model mismatch')
        if meta['issued']>now or now-meta['issued']>300:raise ValueError('stale or future')
        if meta['event_id'] in self.seen:raise ValueError('replay')
        pq=kem.decaps(self.pqsk,bytes.fromhex(env['kem_ct']))
        classic=self.xsk.exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(env['ephemeral_xpk'])))
        transcript=canonical({'kem_ct':env['kem_ct'],'ephemeral_xpk':env['ephemeral_xpk'],'metadata':meta})
        key=derive(pq,classic,transcript)
        plain=AESGCM(key).decrypt(bytes.fromhex(env['nonce']),bytes.fromhex(env['ciphertext']),canonical(meta))
        obj=json.loads(plain)
        if set(obj)!={'screening_only','p_diabetes','p_hypertension'} or obj['screening_only'] is not True:raise ValueError('schema')
        if not all(isinstance(obj[k],(float,int)) and not isinstance(obj[k],bool) and 0<=obj[k]<=1 for k in ['p_diabetes','p_hypertension']):raise ValueError('probability')
        self.seen.add(meta['event_id'])
        return obj

def seal(receiver,ssk,event_id,issued=1000,profile=PROFILE,model=None,obj=None):
    ct,pq=kem.encaps(receiver.pqpk);xsk=X25519PrivateKey.generate()
    xpk=xsk.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    classic=xsk.exchange(X25519PublicKey.from_public_bytes(receiver.xpk))
    meta={'profile':profile,'model':model or receiver.expected_model,'event_id':event_id,'issued':issued,'endpoint_version':'screening-1'}
    transcript=canonical({'kem_ct':ct.hex(),'ephemeral_xpk':xpk.hex(),'metadata':meta})
    key=derive(pq,classic,transcript);nonce=os.urandom(12)
    if obj is None:obj={'screening_only':True,'p_diabetes':.07,'p_hypertension':.31}
    ciphertext=AESGCM(key).encrypt(nonce,canonical(obj),canonical(meta))
    env={'metadata':meta,'kem_ct':ct.hex(),'ephemeral_xpk':xpk.hex(),'nonce':nonce.hex(),'ciphertext':ciphertext.hex()}
    env['signature']=dsa.sign(ssk,canonical(env)).hex()
    return env

def resign(env,ssk):
    env['signature']=dsa.sign(ssk,canonical({k:v for k,v in env.items() if k!='signature'})).hex();return env

def measure(fn,n=500):
    for _ in range(20):fn()
    samples=[]
    for _ in range(n):
        start=time.perf_counter_ns();fn();samples.append((time.perf_counter_ns()-start)/1e6)
    return {'n':n,'median_ms':float(np.median(samples)),'p95_ms':float(np.quantile(samples,.95)),'samples_ms':samples}

def run():
    r=Receiver();spk,ssk=dsa.keygen();r.spk=spk
    kpk,ksk=kem.keygen();ct,ss=kem.encaps(kpk)
    sig=dsa.sign(ssk,b'research-model-manifest')
    bench={
        'ML-KEM-768 keygen':measure(kem.keygen),
        'ML-KEM-768 encaps':measure(lambda:kem.encaps(kpk)),
        'ML-KEM-768 decaps':measure(lambda:kem.decaps(ksk,ct)),
        'ML-DSA-44 keygen':measure(dsa.keygen),
        'ML-DSA-44 sign':measure(lambda:dsa.sign(ssk,b'research-model-manifest')),
        'ML-DSA-44 verify':measure(lambda:dsa.verify(spk,b'research-model-manifest',sig)),
        'Envelope seal':measure(lambda:seal(r,ssk,os.urandom(12).hex()))}
    # Fresh event identifiers avoid replay rejection during receive timing.
    events=[seal(r,ssk,f'benchmark-{i}') for i in range(520)];iterator=iter(events)
    bench['Envelope verify decrypt']=measure(lambda:r.receive(next(iterator)))
    cases={}
    for case in ['valid','tampered_signature','tampered_ciphertext','kem_corruption_resigned','wrong_model','downgrade','expired','future','unauthorized_role','invalid_probability','replay','wrong_signing_key']:
        success=0
        for i in range(100):
            kwargs={};role='clinician'
            if case=='wrong_model':kwargs['model']='unapproved-model'
            if case=='downgrade':kwargs['profile']='classical-only'
            if case=='expired':kwargs['issued']=600
            if case=='future':kwargs['issued']=1001
            if case=='invalid_probability':kwargs['obj']={'screening_only':True,'p_diabetes':1.2,'p_hypertension':.3}
            env=seal(r,ssk,f'{case}-{i}',**kwargs)
            if case=='tampered_signature':env['signature']=flip(bytes.fromhex(env['signature'])).hex()
            if case=='tampered_ciphertext':env['ciphertext']=flip(bytes.fromhex(env['ciphertext'])).hex()
            if case=='kem_corruption_resigned':env['kem_ct']=flip(bytes.fromhex(env['kem_ct'])).hex();resign(env,ssk)
            if case=='wrong_signing_key':_,other_sk=dsa.keygen();resign(env,other_sk)
            if case=='unauthorized_role':role='guest'
            if case=='replay':r.receive(env)
            accepted=True
            try:r.receive(env,role=role)
            except Exception:accepted=False
            success+=int(accepted==(case=='valid'))
        cases[case]={'trials':100,'expected_behavior_observed':success,'expected':'accept' if case=='valid' else 'reject'}
    env=seal(r,ssk,'size')
    result={'profile':PROFILE,'model_manifest_digest':MODEL_DIGEST,'benchmarks':bench,'functional_cases':cases,
        'sizes_bytes':{'kem_public_key':len(kpk),'kem_ciphertext':len(ct),'signature_public_key':len(spk),'signature':len(sig),'json_envelope':len(canonical(env))},
        'limitations':'Single-process application harness. No certified module, TLS integration, PKI, persistent anti-replay storage, penetration test or side-channel evaluation.',
        'environment':{'platform':platform.platform(),'processor':platform.processor(),'pqcrypto':importlib.metadata.version('pqcrypto'),'cryptography':importlib.metadata.version('cryptography')}}
    (ROOT/'results/security.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));names=list(bench);med=[bench[k]['median_ms'] for k in names];p95=[bench[k]['p95_ms'] for k in names]
    ax.barh(names,med,color='#176b8e');ax.plot(p95,range(len(names)),'o',color='#ab4936',label='95th percentile');ax.set_xscale('log');ax.set_xlabel('Milliseconds per operation  logarithmic scale');ax.set_title('Local cryptography benchmark  500 measured repetitions');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/05_security_benchmark.png',dpi=190)
    print(json.dumps({'functional_cases':cases,'benchmarks':{k:{x:v[x] for x in ['n','median_ms','p95_ms']} for k,v in bench.items()},'sizes_bytes':result['sizes_bytes']},indent=2))

if __name__=='__main__':run()
