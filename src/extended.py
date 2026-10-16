"""Extended analyses. All models and thresholds are fixed before any test-cycle outcome is read.

Adds design-aware uncertainty, paired comparisons, rule-based baselines, repeated-split
robustness, a reverse-direction cycle check, threshold-target sensitivity, calibration
slope and intercept, age-specific thresholds and net-benefit tables.
Not clinical software.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import cohort, metrics, threshold80, logit, FEATURES, SEED, ROOT

TASKS = ['diabetes', 'hypertension']
NB = 1000


def make_model(name):
    if name.startswith('LR'):
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             LogisticRegression(C=1., max_iter=2000, random_state=SEED))
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                         HistGradientBoostingClassifier(max_iter=100, learning_rate=.05, max_leaf_nodes=7,
                                                        min_samples_leaf=30, l2_regularization=1., random_state=SEED))


FEATSETS = {'LR_core': FEATURES[:4], 'LR_full': FEATURES, 'HGB_full': FEATURES, 'LR_age': ['age']}


def fit_candidates(dev, task, seed, names=('LR_core', 'LR_full', 'HGB_full')):
    """Replicates the primary protocol: 60/20/20 stratified split, sigmoid calibration, validation threshold."""
    strata = dev.diabetes * 2 + dev.hypertension
    train, other = train_test_split(np.arange(len(dev)), test_size=.4, random_state=seed, stratify=strata)
    cal, val = train_test_split(other, test_size=.5, random_state=seed, stratify=strata.iloc[other])
    out = {}
    for name in names:
        f = FEATSETS[name]
        m = make_model(name).fit(dev.iloc[train][f], dev.iloc[train][task])
        c = LogisticRegression(C=1e6, max_iter=2000).fit(logit(m.predict_proba(dev.iloc[cal][f])[:, 1]), dev.iloc[cal][task])

        def predict(d, m=m, c=c, f=f):
            return c.predict_proba(logit(m.predict_proba(d[f])[:, 1]))[:, 1]
        pv = predict(dev.iloc[val])
        out[name] = dict(predict=predict, pv=pv, yv=dev.iloc[val][task].to_numpy(), auroc_val=roc_auc_score(dev.iloc[val][task], pv))
    return out, val


def pick(cands):
    return max(cands, key=lambda k: cands[k]['auroc_val'] if k != 'LR_age' else -1)


def rao_wu(test, rng):
    """Rao-Wu rescaled bootstrap multipliers: m_h = n_h - 1 PSUs drawn with replacement per stratum."""
    mult = np.zeros(len(test))
    for _, g in test.groupby('SDMVSTRA'):
        psus = g.SDMVPSU.unique()
        n = len(psus)
        draws = rng.choice(psus, size=n - 1, replace=True)
        for p in psus:
            r = int((draws == p).sum())
            mult[g.index[g.SDMVPSU == p]] = n / (n - 1) * r
    return mult


def q(v):
    return np.quantile(np.asarray(v), [.025, .975]).tolist()


def calib_slope_intercept(y, p):
    z = logit(p)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(z, y)
    off = LogisticRegression(C=1e6, max_iter=2000, fit_intercept=True)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, bins=8):
    order = np.argsort(p)
    parts = np.array_split(order, bins)
    return float(sum(len(ix) / len(y) * abs(np.mean(y[ix]) - np.mean(p[ix])) for ix in parts))


def run():
    dev, _ = cohort('I')
    test, _ = cohort('J')
    test = test.reset_index(drop=True)
    res = {'seed': SEED, 'replicates': NB}
    primary, vals, thr = {}, {}, {}
    for task in TASKS:
        cands, val = fit_candidates(dev, task, SEED, names=('LR_core', 'LR_full', 'HGB_full', 'LR_age'))
        sel = pick(cands)
        primary[task] = cands
        thr[task] = {k: threshold80(v['yv'], v['pv']) for k, v in cands.items()}
        vals[task] = sel
    P = {t: {k: v['predict'](test) for k, v in primary[t].items()} for t in TASKS}
    y = {t: test[t].to_numpy() for t in TASKS}

    # E1 design-aware uncertainty (Rao-Wu PSU bootstrap), selected models, frozen thresholds
    rng = np.random.default_rng(SEED)
    reps = {t: {'cluster_unweighted': [], 'design_weighted': []} for t in TASKS}
    for _ in range(NB):
        mult = rao_wu(test, rng)
        for t in TASKS:
            p = P[t][vals[t]]
            keep = mult > 0
            for key, w in [('cluster_unweighted', mult), ('design_weighted', mult * test.WTMEC2YR.to_numpy())]:
                if y[t][keep].min() == y[t][keep].max():
                    continue
                m = metrics(y[t][keep], p[keep], thr[t][vals[t]], weight=w[keep])
                reps[t][key].append([m['auroc'], m['sensitivity'], m['specificity'], m['ppv']])
    res['design_aware'] = {}
    for t in TASKS:
        p = P[t][vals[t]]
        point_w = metrics(y[t], p, thr[t][vals[t]], weight=test.WTMEC2YR)
        point_u = metrics(y[t], p, thr[t][vals[t]])
        out = {'selected': vals[t], 'strata': int(test.SDMVSTRA.nunique()), 'psu_per_stratum': 2}
        for key, point in [('cluster_unweighted', point_u), ('design_weighted', point_w)]:
            a = np.array(reps[t][key])
            out[key] = {'n_rep': len(a)}
            for j, k in enumerate(['auroc', 'sensitivity', 'specificity', 'ppv']):
                out[key][k] = {'point': point[k], 'ci': q(a[:, j]), 'se': float(a[:, j].std(ddof=1))}
        res['design_aware'][t] = out

    # E2 paired bootstrap of AUROC differences
    rng = np.random.default_rng(SEED + 1)
    res['paired_auroc'] = {}
    comps = {'diabetes': [('HGB_full', 'LR_core'), ('LR_full', 'LR_core'), ('LR_core', 'LR_age'), ('HGB_full', 'LR_age')],
             'hypertension': [('LR_core', 'LR_full'), ('LR_core', 'HGB_full'), ('LR_core', 'LR_age'), ('LR_full', 'LR_age')]}
    for t in TASKS:
        res['paired_auroc'][t] = []
        for a, b in comps[t]:
            d = []
            for _ in range(NB):
                i = rng.integers(0, len(test), len(test))
                if y[t][i].min() == y[t][i].max():
                    continue
                d.append(roc_auc_score(y[t][i], P[t][a][i]) - roc_auc_score(y[t][i], P[t][b][i]))
            res['paired_auroc'][t].append({'a': a, 'b': b, 'delta': float(roc_auc_score(y[t], P[t][a]) - roc_auc_score(y[t], P[t][b])),
                                           'ci': q(d), 'excludes_zero': bool(q(d)[0] > 0 or q(d)[1] < 0)})

    # E3 rule baselines specified in advance from common screening practice; no tuning
    rules = {
        'diabetes': [('Age 35 or older and BMI 25 or higher', (test.age >= 35) & (test.bmi >= 25)),
                     ('Age 45 or older', test.age >= 45), ('BMI 30 or higher', test.bmi >= 30)],
        'hypertension': [('Age 40 or older', test.age >= 40), ('Age 50 or older', test.age >= 50),
                         ('BMI 30 or higher', test.bmi >= 30)]}
    res['rule_baselines'] = {}
    for t in TASKS:
        res['rule_baselines'][t] = []
        pm = P[t][vals[t]]
        for name, a in rules[t]:
            a = a.fillna(False).to_numpy()
            tp, fp, fn = int((a & (y[t] == 1)).sum()), int((a & (y[t] == 0)).sum()), int((~a & (y[t] == 1)).sum())
            frac = float(a.mean())
            k = int(round(frac * len(pm)))
            topk = np.zeros(len(pm), bool)
            topk[np.argsort(-pm)[:k]] = True
            res['rule_baselines'][t].append({'rule': name, 'referral': frac, 'sensitivity': tp / (tp + fn), 'ppv': tp / (tp + fp) if tp + fp else None,
                                             'model_sensitivity_at_same_referral': float((topk & (y[t] == 1)).sum() / y[t].sum()),
                                             'model_ppv_at_same_referral': float((topk & (y[t] == 1)).sum() / max(k, 1))})
        a_age = P[t]['LR_age']
        m = metrics(y[t], a_age, thr[t]['LR_age'])
        res['rule_baselines'][t].append({'rule': 'Age-only logistic model, validation 80% threshold', 'referral': m['referral'],
                                         'sensitivity': m['sensitivity'], 'ppv': m['ppv'], 'auroc': m['auroc']})

    # E4 sensitivity-target sweep (thresholds chosen on validation only)
    res['target_sweep'] = {}
    for t in TASKS:
        c = primary[t][vals[t]]
        res['target_sweep'][t] = []
        pos = np.sort(c['pv'][c['yv'] == 1])[::-1]
        for target in [.7, .8, .9, .95]:
            th = float(pos[int(np.ceil(target * len(pos))) - 1])
            m = metrics(y[t], P[t][vals[t]], th)
            res['target_sweep'][t].append({'target': target, 'threshold': th, 'val_positives': int(len(pos)), **{k: m[k] for k in ['sensitivity', 'specificity', 'ppv', 'referral']}})

    # E5 repeated-split robustness of the whole pipeline
    rows = []
    for seed in range(101, 131):
        for t in TASKS:
            cands, _ = fit_candidates(dev, t, seed)
            sel = pick(cands)
            th = threshold80(cands[sel]['yv'], cands[sel]['pv'])
            p = cands[sel]['predict'](test)
            m = metrics(y[t], p, th)
            rows.append({'seed': seed, 'task': t, 'selected': sel, 'threshold': th, **{k: m[k] for k in ['auroc', 'sensitivity', 'specificity', 'ppv', 'referral']}})
    rep = pd.DataFrame(rows)
    rep.to_csv(ROOT / 'results/repeated_splits.csv', index=False)
    res['repeated_splits'] = {}
    for t in TASKS:
        d = rep[rep.task == t]
        res['repeated_splits'][t] = {'n_seeds': int(len(d)), 'selected_counts': d.selected.value_counts().to_dict(),
                                     **{k: {'mean': float(d[k].mean()), 'sd': float(d[k].std()), 'min': float(d[k].min()), 'max': float(d[k].max()),
                                            'q025': float(d[k].quantile(.025)), 'q975': float(d[k].quantile(.975))} for k in ['auroc', 'sensitivity', 'ppv', 'referral', 'threshold']}}

    # E8 reverse direction: develop on 2017-2018, evaluate on 2015-2016 with the identical protocol
    res['reverse_cycle'] = {}
    for t in TASKS:
        cands, _ = fit_candidates(test, t, SEED, names=('LR_core', 'LR_full', 'HGB_full'))
        sel = pick(cands)
        th = threshold80(cands[sel]['yv'], cands[sel]['pv'])
        p = cands[sel]['predict'](dev)
        m = metrics(dev[t], p, th)
        res['reverse_cycle'][t] = {'selected': sel, 'dev_positives': int(test[t].sum()), 'eval_n': len(dev), 'eval_positives': int(dev[t].sum()),
                                   **{k: m[k] for k in ['auroc', 'ap', 'sensitivity', 'specificity', 'ppv', 'referral', 'threshold']}}

    # E6 calibration slope, intercept and ECE of the selected models
    rng = np.random.default_rng(SEED + 2)
    res['calibration'] = {}
    for t in TASKS:
        p = P[t][vals[t]]
        s, i0 = calib_slope_intercept(y[t], p)
        bs, bi = [], []
        for _ in range(NB):
            ix = rng.integers(0, len(p), len(p))
            a, b = calib_slope_intercept(y[t][ix], p[ix])
            bs.append(a)
            bi.append(b)
        res['calibration'][t] = {'slope': s, 'slope_ci': q(bs), 'intercept': i0, 'intercept_ci': q(bi), 'ece8': ece(y[t], p),
                                 'mean_predicted': float(p.mean()), 'observed': float(y[t].mean())}

    # E7 age-specific thresholds for the BP model, chosen on validation only
    c = primary['hypertension'][vals['hypertension']]
    strata = dev.diabetes * 2 + dev.hypertension
    tr, oth = train_test_split(np.arange(len(dev)), test_size=.4, random_state=SEED, stratify=strata)
    ca, va = train_test_split(oth, test_size=.5, random_state=SEED, stratify=strata.iloc[oth])
    vdf = dev.iloc[va]
    young_v = (vdf.age < 50).to_numpy()
    th_y = threshold80(c['yv'][young_v], c['pv'][young_v])
    th_o = threshold80(c['yv'][~young_v], c['pv'][~young_v])
    pt = P['hypertension'][vals['hypertension']]
    yt = (test.age < 50).to_numpy()
    th_vec = np.where(yt, th_y, th_o)
    ref = pt >= th_vec
    yy = y['hypertension']
    out = {'threshold_under50': th_y, 'threshold_50plus': th_o, 'val_positives_under50': int(c['yv'][young_v].sum()), 'val_positives_50plus': int(c['yv'][~young_v].sum())}
    for nm, mask in [('overall', np.ones(len(yy), bool)), ('under50', yt), ('50plus', ~yt)]:
        tp = int((ref & mask & (yy == 1)).sum())
        fn = int((~ref & mask & (yy == 1)).sum())
        fp = int((ref & mask & (yy == 0)).sum())
        tn = int((~ref & mask & (yy == 0)).sum())
        out[nm] = {'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp), 'ppv': tp / (tp + fp), 'referral': float((ref & mask).sum() / mask.sum())}
    res['age_specific_threshold_bp'] = out

    # E9 net benefit at stated thresholds
    res['net_benefit'] = {}
    for t, grid in [('diabetes', [.01, .02, .03, .05]), ('hypertension', [.2, .3, .4, .5])]:
        p, yy = P[t][vals[t]], y[t].astype(bool)
        prev = yy.mean()
        rows = []
        for th in grid:
            a = p >= th
            rows.append({'threshold': th, 'model': float((a & yy).mean() - (a & ~yy).mean() * th / (1 - th)), 'refer_all': float(prev - (1 - prev) * th / (1 - th)), 'refer_none': 0.0})
        res['net_benefit'][t] = rows

    # E10 alternative endpoints without retraining
    alt = {}
    d57 = (test.LBXGH >= 5.7).astype(int).to_numpy()
    m = metrics(d57, P['diabetes'][vals['diabetes']], thr['diabetes'][vals['diabetes']])
    alt['hba1c_5p7'] = {k: m[k] for k in ['auroc', 'ap', 'sensitivity', 'specificity', 'ppv']} | {'positives': int(d57.sum())}
    sbp_only = (test.sbp >= 130).astype(int).to_numpy()
    m = metrics(sbp_only, P['hypertension'][vals['hypertension']], thr['hypertension'][vals['hypertension']])
    alt['sbp_130'] = {k: m[k] for k in ['auroc', 'ap', 'sensitivity', 'specificity', 'ppv']} | {'positives': int(sbp_only.sum())}
    res['alt_endpoints'] = alt

    # E11 planning arithmetic
    plan = []
    for prev in [.028, .05, .10]:
        for sens, hw in [(.8, .05), (.8, .075), (.9, .05)]:
            npos = int(np.ceil(1.96 ** 2 * sens * (1 - sens) / hw ** 2))
            plan.append({'prevalence': prev, 'sensitivity': sens, 'half_width': hw, 'positives': npos, 'recruits_10pct_nonevaluable': int(np.ceil(npos / prev / .9))})
    res['planning'] = plan
    res['selected_models'] = vals
    (ROOT / 'results/extended.json').write_text(json.dumps(res, indent=2))
    figures(res, rep, test, P, y, vals)
    print(json.dumps({k: res[k] for k in ['paired_auroc', 'rule_baselines', 'reverse_cycle', 'repeated_splits']}, indent=1)[:6000])


def figures(res, rep, test, P, y, vals):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axs, TASKS):
        d = rep[rep.task == t]
        ax.hist(d.auroc, bins=10, color='#176b8e', alpha=.85)
        ax.axvline(res['design_aware'][t]['cluster_unweighted']['auroc']['point'], color='#ab4936', ls='--', label='Primary split')
        ax.set(title=('HbA1c' if t == 'diabetes' else 'Blood pressure') + ' endpoint', xlabel='Temporal AUROC across 30 development splits', ylabel='Splits')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/07_repeated_splits.png', dpi=190)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axs, TASKS):
        da = res['design_aware'][t]
        labs = ['Participant bootstrap\n(primary)', 'PSU bootstrap\n(unweighted)', 'Design-weighted\n(Rao-Wu)']
        base = None
        pts, lo, hi = [], [], []
        for key in ['cluster_unweighted', 'design_weighted']:
            e = da[key]['sensitivity']
            pts.append(e['point'])
            lo.append(e['ci'][0])
            hi.append(e['ci'][1])
        # primary conditional interval is stored in study.json
        S = json.loads((ROOT / 'results/study.json').read_text())
        c = S['models'][t]['ci']['sensitivity']
        pts = [S['models'][t]['candidates'][S['models'][t]['selected']]['test']['sensitivity']] + pts
        lo = [c[0]] + lo
        hi = [c[1]] + hi
        pts, lo, hi = map(np.array, (pts, lo, hi))
        ax.errorbar(pts, range(3), xerr=[pts - lo, hi - pts], fmt='o', color='#176b8e', capsize=4)
        ax.set_yticks(range(3), labs)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Sensitivity at frozen threshold')
        ax.set_title('HbA1c' if t == 'diabetes' else 'Blood pressure')
        ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/08_design_aware_intervals.png', dpi=190)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t in zip(axs, TASKS):
        rows = res['target_sweep'][t]
        ax.plot([r['referral'] for r in rows], [r['sensitivity'] for r in rows], 'o-', color='#176b8e', label='Selected model')
        for r in rows:
            ax.annotate(f"{int(r['target'] * 100)}%", (r['referral'], r['sensitivity']), textcoords='offset points', xytext=(5, -10), fontsize=8)
        for r in res['rule_baselines'][t]:
            ax.plot(r['referral'], r['sensitivity'], 's', color='#ab4936')
        ax.plot([], [], 's', color='#ab4936', label='Rule or age-only baseline')
        ax.set(title=('HbA1c' if t == 'diabetes' else 'Blood pressure') + ' endpoint', xlabel='Fraction referred', ylabel='Test sensitivity', xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/09_operating_points.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    run()
