"""Additional envelope experiments: byte-level mutation, payload scaling, classical baseline cost.

Same scope limits as security.py: single-process harness, not a protocol, not a certified module.
"""
import json, os, time, platform
import numpy as np
from pathlib import Path
from pqcrypto.sign import ml_dsa_44 as dsa
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from security import Receiver, seal, canonical, measure, ROOT

NONCE = bytes(12)
HEX_FIELDS = ['kem_ct', 'ephemeral_xpk', 'nonce', 'ciphertext', 'signature']


def mutate_hex(env, field, rng):
    b = bytearray(bytes.fromhex(env[field]))
    i = int(rng.integers(0, len(b)))
    b[i] ^= 1 << int(rng.integers(0, 8))
    env[field] = bytes(b).hex()
    return i


def main():
    rng = np.random.default_rng(20261016)
    r = Receiver()
    spk, ssk = dsa.keygen()
    r.spk = spk
    out = {}

    # F1: single-bit mutation of every hex-encoded field, receiver without re-signing, 400 trials per field
    fuzz = {}
    for f in HEX_FIELDS:
        rejected = 0
        for i in range(400):
            env = seal(r, ssk, f'fz-{f}-{i}')
            mutate_hex(env, f, rng)
            try:
                r.receive(env)
            except Exception:
                rejected += 1
        fuzz[f] = {'trials': 400, 'rejected': rejected}
    # metadata mutation: change one metadata value
    rejected = 0
    for i in range(400):
        env = seal(r, ssk, f'fz-meta-{i}')
        k = ['profile', 'model', 'event_id', 'issued', 'endpoint_version'][i % 5]
        env['metadata'][k] = (env['metadata'][k] + 1) if k == 'issued' else env['metadata'][k] + 'x'
        try:
            r.receive(env)
        except Exception:
            rejected += 1
    fuzz['metadata'] = {'trials': 400, 'rejected': rejected}

    # F2: authenticated-layer fuzz: mutate then re-sign with the legitimate key, so only the AEAD and KEM layers can reject
    fuzz_resigned = {}
    for f in ['kem_ct', 'ephemeral_xpk', 'nonce', 'ciphertext']:
        rejected = 0
        for i in range(400):
            env = seal(r, ssk, f'fr-{f}-{i}')
            mutate_hex(env, f, rng)
            env['signature'] = dsa.sign(ssk, canonical({k: v for k, v in env.items() if k != 'signature'})).hex()
            try:
                r.receive(env)
            except Exception:
                rejected += 1
        fuzz_resigned[f] = {'trials': 400, 'rejected': rejected}
    out['mutation_fuzz'] = fuzz
    out['mutation_fuzz_resigned'] = fuzz_resigned

    # F3: payload scaling for seal and open (hybrid envelope)
    scaling = []
    for size in [64, 1024, 16384, 262144]:
        obj = {'screening_only': True, 'p_diabetes': .07, 'p_hypertension': .31}
        # timing only: encrypt payload with the derived-style key to show AEAD scaling is negligible next to signing
        key = AESGCM.generate_key(256)
        data = os.urandom(size)
        nonce = os.urandom(12)
        m = measure(lambda: AESGCM(key).encrypt(nonce, data, b'meta'), n=300)
        scaling.append({'payload_bytes': size, 'aes_gcm_median_ms': m['median_ms'], 'aes_gcm_p95_ms': m['p95_ms']})
    out['aead_scaling'] = scaling

    # F4: classical-only baseline (X25519 + Ed25519 + AES-GCM) vs the hybrid post-quantum envelope
    xr = X25519PrivateKey.generate()
    xr_pub = xr.public_key()
    ed = Ed25519PrivateKey.generate()
    ed_pub = ed.public_key()
    payload = canonical({'screening_only': True, 'p_diabetes': .07, 'p_hypertension': .31})

    def classical_seal():
        e = X25519PrivateKey.generate()
        epub = e.public_key()
        shared = e.exchange(xr_pub)
        ct = AESGCM(shared).encrypt(NONCE, payload, b'meta')
        return epub, ct, ed.sign(ct)

    def classical_open(item):
        epub, ct, sg = item
        ed_pub.verify(sg, ct)
        return AESGCM(xr.exchange(epub)).decrypt(NONCE, ct, b'meta')

    item = classical_seal()
    assert classical_open(item) == payload
    cs = measure(classical_seal, n=500)
    cv = measure(lambda: classical_open(item), n=500)
    ct, sg = item[1], item[2]
    r2 = Receiver()
    r2.spk = spk
    hs = measure(lambda: seal(r2, ssk, os.urandom(12).hex()), n=500)
    events = [seal(r2, ssk, f'cmp-{i}') for i in range(520)]
    it = iter(events)
    ho = measure(lambda: r2.receive(next(it)), n=500)
    env = seal(r2, ssk, 'sz')
    out['classical_vs_hybrid'] = {
        'classical_seal_median_ms': cs['median_ms'], 'classical_seal_p95_ms': cs['p95_ms'],
        'classical_open_median_ms': cv['median_ms'], 'classical_open_p95_ms': cv['p95_ms'],
        'hybrid_seal_median_ms': hs['median_ms'], 'hybrid_seal_p95_ms': hs['p95_ms'],
        'hybrid_open_median_ms': ho['median_ms'], 'hybrid_open_p95_ms': ho['p95_ms'],
        'classical_bytes': len(ct) + len(sg) + 32, 'hybrid_bytes': len(canonical(env)),
        'note': 'Classical baseline: ephemeral X25519, AES-256-GCM with a fixed test nonce, Ed25519 signature; no metadata, replay or role checks, so it is a lower bound on classical envelope cost.'}
    out['environment'] = {'platform': platform.platform(), 'python': platform.python_version()}
    (ROOT / 'results/security_extended.json').write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(9, 3.6))
    names = list(fuzz) + [f'{k} (re-signed)' for k in fuzz_resigned]
    vals = [fuzz[k]['rejected'] / fuzz[k]['trials'] for k in fuzz] + [fuzz_resigned[k]['rejected'] / fuzz_resigned[k]['trials'] for k in fuzz_resigned]
    ax.barh(names, vals, color='#176b8e')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Fraction of mutated envelopes rejected (400 trials each)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/10_mutation_fuzz.png', dpi=190)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
    main()
