"""Cost of rejecting invalid envelopes, by fault type.

Supports the receiver check order (role, signature, profile, model, time, replay, then KEM and AEAD):
cheap or attacker-independent checks first, so invalid messages cost little. Same scope limits as security.py.
"""
import json, os, platform
from pqcrypto.sign import ml_dsa_44 as dsa
from security import Receiver, seal, resign, flip, measure, ROOT


def main():
    r = Receiver()
    spk, ssk = dsa.keygen()
    r.spk = spk
    _, other_sk = dsa.keygen()
    out = {}

    def run(name, make, role='clinician', n=300):
        items = [make(i) for i in range(n + 20)]
        it = iter(items)

        def call():
            try:
                r.receive(next(it), role=role)
            except Exception:
                pass
        m = measure(call, n=n)
        out[name] = {'median_ms': m['median_ms'], 'p95_ms': m['p95_ms'], 'n': n}

    def tag(i): return f'rc-{os.urandom(6).hex()}-{i}'
    run('valid message accepted', lambda i: seal(r, ssk, tag(i)))
    run('unauthorized role', lambda i: seal(r, ssk, tag(i)), role='guest')
    run('bad signature', lambda i: (lambda e: (e.__setitem__('signature', flip(bytes.fromhex(e['signature'])).hex()), e)[1])(seal(r, ssk, tag(i))))
    run('wrong signing key', lambda i: resign(seal(r, ssk, tag(i)), other_sk))
    run('wrong model digest (signed)', lambda i: seal(r, ssk, tag(i), model='unapproved-model'))
    run('expired (signed)', lambda i: seal(r, ssk, tag(i), issued=600))
    run('corrupted KEM ciphertext (re-signed)', lambda i: (lambda e: resign((e.__setitem__('kem_ct', flip(bytes.fromhex(e['kem_ct'])).hex()), e)[1], ssk))(seal(r, ssk, tag(i))))
    run('corrupted AEAD ciphertext (re-signed)', lambda i: (lambda e: resign((e.__setitem__('ciphertext', flip(bytes.fromhex(e['ciphertext'])).hex()), e)[1], ssk))(seal(r, ssk, tag(i))))
    out['environment'] = {'platform': platform.platform(), 'python': platform.python_version()}
    (ROOT / 'results/security_rejection.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'environment'}, indent=1))


if __name__ == '__main__':
    main()
