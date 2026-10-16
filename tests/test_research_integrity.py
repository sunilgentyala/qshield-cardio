import unittest,sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from study import cohort,FEATURES,threshold80,ROOT
from security import Receiver,seal,PROFILE
from pqcrypto.sign import ml_dsa_44 as dsa
from infer import predict

class ResearchIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.dev,_=cohort('I');cls.test,_=cohort('J')
    def test_participant_separation(self):self.assertFalse(set(self.dev.SEQN)&set(self.test.SEQN))
    def test_endpoint_values(self):
        for df in [self.dev,self.test]:
            self.assertTrue((df.diabetes==(df.LBXGH>=6.5)).all())
            self.assertTrue((df.hypertension==((df.sbp>=130)|(df.dbp>=80))).all())
            self.assertTrue((df.bp_pairs>=2).all())
    def test_no_diagnostic_predictors(self):
        self.assertEqual(set(FEATURES),{'age','male','bmi','waist','current_smoker','vigorous_recreation'})
    def test_exclusion_rules(self):
        for df in [self.dev,self.test]:
            self.assertTrue(df.DIQ010.isin([2,3]).all());self.assertTrue((df.BPQ020==2).all())
            self.assertFalse((df.DIQ050==1).any());self.assertFalse((df.DIQ070==1).any())
            self.assertFalse((df.BPQ050A==1).any());self.assertFalse((df.RIDEXPRG==1).any())
    def test_frozen_threshold_logic(self):
        y=np.array([1]*5+[0]*3);p=np.array([.9,.8,.7,.6,.5,.4,.3,.2])
        self.assertAlmostEqual(threshold80(y,p),.6)
    def test_source_digests(self):
        for x in json.loads((ROOT/'data/sources.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/'data/raw'/x['file']).read_bytes()).hexdigest(),x['sha256'])
    def test_envelope_roundtrip_and_replay(self):
        receiver=Receiver();pk,sk=dsa.keygen();receiver.spk=pk;e=seal(receiver,sk,'unit-test')
        self.assertTrue(receiver.receive(e)['screening_only'])
        with self.assertRaises(ValueError):receiver.receive(e)
    def test_ciphertext_mutation(self):
        receiver=Receiver();pk,sk=dsa.keygen();receiver.spk=pk;e=seal(receiver,sk,'unit-mutation');e['ciphertext']='00'+e['ciphertext'][2:]
        with self.assertRaises(Exception):receiver.receive(e)
    def test_artifact_trust_anchor_and_inference(self):
        manifest=ROOT/'artifacts/manifest.json';digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        result=predict({'age':45,'male':0,'bmi':28,'waist':94,'current_smoker':0,'vigorous_recreation':1},digest)
        self.assertTrue(result['screening_only'])
        for value in result['tasks'].values():self.assertTrue(0<=value['probability']<=1)
        with self.assertRaises(ValueError):predict({},'0'*64)
    def test_recorded_security_trials(self):
        s=json.loads((ROOT/'results/security.json').read_text())
        for c in s['functional_cases'].values():self.assertEqual(c['trials'],c['expected_behavior_observed'])
    def test_split_manifest_partition(self):
        import pandas as pd
        m=pd.read_csv(ROOT/'results/split_manifest.csv')
        self.assertTrue(m.SEQN.is_unique)
        self.assertEqual(set(m[m.partition!='test'].SEQN),set(self.dev.SEQN))
        self.assertEqual(set(m[m.partition=='test'].SEQN),set(self.test.SEQN))

if __name__=='__main__':unittest.main(verbosity=2)
