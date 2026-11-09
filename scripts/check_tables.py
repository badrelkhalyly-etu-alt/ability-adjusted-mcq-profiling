"""Check that every cell of the manuscript LaTeX tables equals its source CSV in results/ (prints mismatches)."""
from pathlib import Path
import pandas as pd, os, json
os.chdir(Path(__file__).resolve().parents[1])
BS = chr(92)
def rows(tex):
    s = open(tex, encoding='utf8').read()
    body = s.split(BS + 'midrule')[1].split(BS + 'bottomrule')[0]
    out = []
    for r in body.split(BS + BS):
        if not r.strip():
            continue
        cells = [c.strip().replace(BS + '_', '_').replace('{,}', '').replace(BS + '%', '') for c in r.split('&')]
        out.append(cells)
    return out
T = 'results/tables/'
tot = 0
def report(name, n, bad):
    global tot
    tot += n
    print(name, 'cells', n, 'bad', bad)

# e10
df = pd.read_csv(T + 'math_formA_E10_test_scores.csv')
R = rows('manuscript/tables/tab_e10.tex'); bad = 0; n = 0
for r in R:
    row = df[df.model == r[0]].iloc[0]
    for c, v in zip(['logloss', 'brier', 'ece'], r[1:]):
        n += 1
        if '%.4f' % row[c] != v: bad += 1; print('e10', r[0], c, v, '%.4f' % row[c])
print('e10 rows', len(R), len(df)); report('e10', n, bad)

p = pd.read_csv(T + 'math_formA_E10_paired.csv'); p['key'] = p.better + '|' + p.than
R = rows('manuscript/tables/tab_e10_paired.tex'); bad = 0; n = 0
for r in R:
    row = p[p.key == r[0] + '|' + r[1]].iloc[0]
    for c, v in zip(['gain_nats_per_item', 'ci_lo', 'ci_hi'], r[2:]):
        n += 1
        if '%.5f' % row[c] != v: bad += 1; print('paired', r[:2], c, v, '%.5f' % row[c])
print('paired rows', len(R), len(p)); report('paired', n, bad)

ms = pd.read_csv(T + 'math_formA_model_selection_train.csv')
R = rows('manuscript/tables/tab_model_selection.tex'); bad = 0; n = 0
for r in R:
    row = ms[ms.model == r[0]].iloc[0]
    exp = ['%.1f' % row.loglik, '%d' % row.n_params, '%.1f' % row.AIC, '%.1f' % row.BIC, str(row.converged)]
    for a, b in zip(r[1:], exp):
        n += 1
        if a != b: bad += 1; print('ms', r[0], a, b)
print('ms rows', len(R), len(ms)); report('ms', n, bad)

e6 = pd.read_csv(T + 'math_formA_E6_positive_summary.csv')
R = rows('manuscript/tables/tab_e6.tex'); bad = 0; n = 0
for r, (i, row) in zip(R, e6.iterrows()):
    exp = ['%.2f' % row.mag, '%.2f' % row.prop, '%d' % row.n, str(row.dur), '%.2f' % row.power, '%.2f' % row.K2, '%.2f' % row.ari,
           '%.2f' % row.rmse_bdev, '%.2f' % row.accuracy, '--' if pd.isna(row.ari_dur) else '%.2f' % row.ari_dur]
    for a, b in zip(r, exp):
        n += 1
        if a != b: bad += 1; print('e6', a, b)
print('e6 rows', len(R), len(e6)); report('e6', n, bad)

cf = pd.read_csv(T + 'crossform_bic.csv', index_col=0)
R = rows('manuscript/tables/tab_crossform.tex'); bad = 0; n = 0
mn = cf.drop('n_train').min()
for r in R:
    m = r[0]
    for f, v in zip(['Form A', 'Form B', 'Form C'], r[2:]):
        n += 1
        e = '%.0f' % (cf.loc[m, f] - mn[f]); v = v.replace(BS + 'textbf{', '').replace('}', '')
        if e != v: bad += 1; print('cf', m, f, v, e)
print('cf n_train', cf.loc['n_train'].tolist()); report('crossform', n, bad)
# crossform json vs csv
cj = json.load(open('results/metrics/crossform_bic.json'))
for f in cj:
    for m, v in cj[f].items():
        if abs(cf.loc[m, f] - v) > 1e-6: print('cf json/csv mismatch', f, m)

# data
dd = pd.read_csv(T + 'data_descriptives.csv').set_index('form')
R = rows('manuscript/tables/tab_data.tex'); bad = 0; n = 0
spec = [('n', '%d'), ('items', '%d'), ('polytomous_items', '%d'), ('countries', '%d'), ('schools', '%d'), ('booklets', '%d'), ('oecd_share', '%.2f'),
        ('mean_pcorrect', '%.3f'), ('score_mean', '%.2f'), ('pct_missing_resp', '%.2f'), ('pct_any_not_reached', '%.1f'), ('time', None),
        ('no_time', '%d'), ('zero_time', '%d')]
for r, (c, fmt) in zip(R, spec):
    for f, v in zip('ABC', r[1:]):
        n += 1
        row = dd.loc[f]
        e = ('%.1f (%.1f--%.1f)' % (row.time_median, row.time_q1, row.time_q3)) if fmt is None else fmt % row[c]
        if e != v: bad += 1; print('data', c, f, v, e)
report('data', n, bad)

# contingency tex vs csv/json
cj = json.load(open('results/metrics/math_formA_contingency.json'))
s = open('manuscript/tables/math_formA_contingency.tex', encoding='utf8').read()
blocks = s.split(BS + 'begin{table}')[1:]
names = ["profile_x_M1_3", "profile_x_M1_2", "M1_3_x_scoreband", "profile_x_oecd", "profile_x_booklet", "M1_3_x_oecd"]
for t, b in zip(names, blocks):
    c = pd.read_csv(T + f'math_formA_ct_{t}.csv', index_col=0); rp = pd.read_csv(T + f'math_formA_ct_{t}_rowpct.csv', index_col=0)
    rc = (c.div(c.sum(axis=1), axis=0) * 100)
    bad = 0; n = 0
    if not (abs(rc.values - rp.values) < 0.051).all(): print('rowpct mismatch', t)
    st = cj[t]['stats']
    cap = b.split('caption{')[1].split('}' + BS + 'label')[0] if False else b
    exp_cap = '$' + BS + 'chi^2(%d)=%.1f$, Cram' % (st['dof'], st['chi2'])
    if exp_cap not in b: print('caption chi2 mismatch', t)
    if ("$V=%.2f$, $n=%d$" % (st['cramers_v'], st['n'])) not in b: print('caption V/n mismatch', t, st)
    n += 4
    body = b.split(BS + 'midrule')[1].split(BS + 'bottomrule')[0]
    trs = [x for x in body.split(BS + BS) if x.strip()]
    for tr, (idx, crow) in zip(trs, c.iterrows()):
        cells = [x.strip() for x in tr.split('&')]
        if cells[0].replace(BS + '_', '_') != str(idx): print('label', t, cells[0], idx)
        for v, cnt, pc in zip(cells[1:], crow.values, rp.loc[idx].values):
            n += 1
            e = '%d (%.1f%s%%)' % (cnt, pc, BS)
            if v != e: bad += 1; print('ct', t, idx, v, e)
    # json counts
    for col, d in cj[t]['counts'].items():
        for rk, val in d.items():
            if c.loc[type(c.index[0])(rk) if not isinstance(c.index[0], str) else rk, col] != val: print('ct json mismatch', t, rk, col)
    report('ct ' + t, n, bad)
# robustness summary
rs = pd.read_csv(T + 'robustness_summary.csv', index_col=0)
keys = ['resp_reference', 'resp_nr_incorrect', 'resp_weighted_nominal', 'resp_weighted_ess',
        'dur_M3_2_vs_M2_linear', 'dur_M3_2_vs_M2_cubic', 'dur_M3_3_vs_M2_cubic', 'dur_M2_cubic_vs_linear',
        'dev_corr_nr_incorrect', 'dev_corr_weighted', 'min_class_prop_weighted', 'kish_ess']
R = [r for r in rows('manuscript/tables/tab_robustness.tex') if len(r) == 4]; bad = 0; n = 0
for r, k in zip(R, keys):
    for v, f in zip(r[1:], ['math_formA', 'math_formB', 'math_formC']):
        x = rs.loc[k, f]
        e = ('%.2f' % x) if k.startswith(('dev_', 'min_')) else ('%.0f' % x)
        n += 1
        if v.replace('$-$', '-') != e: bad += 1; print('robustness', k, f, v, e)
print('robustness rows', len(R), len(keys)); report('robustness', n, bad)
print('TOTAL table cells', tot)
