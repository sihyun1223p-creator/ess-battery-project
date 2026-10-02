# ===== [셀 0] 준비 : 저장한 pkl 불러오기 (cells_all이 없으면 자동 로드) =====
import os, pickle, logging
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.cm as cm
logging.getLogger().setLevel(logging.CRITICAL)

def _show(name):
    """그래프를 results/ 에 저장한 뒤 화면에 표시"""
    _dir = globals().get('RESULT_DIR', 'results')
    os.makedirs(_dir, exist_ok=True)
    plt.savefig(os.path.join(_dir, name + '.png'), dpi=130, bbox_inches='tight')
    plt.show()


if 'cells_all' not in globals():
    cells_all = {}
    for tag in ['b1', 'b2', 'b3']:
        with open(os.path.join(DATA_DIR, f'{tag}_cells.pkl'), 'rb') as f:
            cells_all.update(pickle.load(f))
    print(f'pkl 로드 완료 : 총 {len(cells_all)}개 셀')

# ===== [셀 A] 셀 단위 피처 테이블 만들기 =====
import re
from scipy.stats import skew, kurtosis

EOL_AH = 0.88   # 논문 EOL 기준 : 공칭 1.1Ah의 80%

def parse_policy(p):
    """'5.4C(50%)-3.6C' -> C1=5.4, SOC 전환=50%, C2=3.6, 0~80% 평균 C-rate"""
    m = re.match(r'\s*([\d.]+)C\((\d+)%\)-([\d.]+)C', p)
    if not m:
        return np.nan, np.nan, np.nan, np.nan
    c1, s, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    s_ = min(s, 0.8)
    t = s_ / c1 + max(0.8 - s_, 0) / c2          # 0→80% 충전 소요시간(h)
    return c1, s * 100, c2, 0.8 / t

def find_knee(q):
    """두 직선 piecewise fit으로 SSE 최소가 되는 분기점(=knee) 사이클"""
    q = pd.Series(q).rolling(15, center=True, min_periods=1).median().values
    x = np.arange(len(q))
    best, knee = np.inf, np.nan
    for k in range(50, len(q) - 20, 5):
        sse = 0
        for xs, ys in [(x[:k], q[:k]), (x[k:], q[k:])]:
            coef = np.polyfit(xs, ys, 1)
            sse += ((np.polyval(coef, xs) - ys) ** 2).sum()
        if sse < best:
            best, knee = sse, k + 1
    return knee

rows = []
for cid, c in cells_all.items():
    s = c['summary']
    qd = s['QDischarge']
    n = len(qd)
    c1, soc, c2, c_avg = parse_policy(c['policy'])
    row = dict(cell=cid, batch=c['batch'], policy=c['policy'],
               cycle_life=c['cycle_life'], n_cycles=n, last_QD=qd[-1],
               C1=c1, SOC_switch=soc, C2=c2, C_avg=c_avg)
    # ΔQ(V) = Q100 - Q10
    q10 = c['Qdlin'][10] if len(c['Qdlin']) > 10 else None
    q100 = c['Qdlin'][100] if len(c['Qdlin']) > 100 else None
    if q10 is not None and q100 is not None and n > 100:
        dq = q100 - q10
        row.update(dQ_logvar=np.log10(np.var(dq)), dQ_min=np.log10(abs(dq.min()) + 1e-9),
                   dQ_mean=np.log10(abs(dq.mean()) + 1e-9), dQ_skew=skew(dq), dQ_kurt=kurtosis(dq))
        q = qd[1:100]  # cycle 2~100
        row.update(QD_c2=qd[1], QD_max_minus_c2=qd[1:100].max() - qd[1],
                   QD_slope_2_100=np.polyfit(np.arange(2, 101), qd[1:100], 1)[0],
                   QD_c100_minus_c2=qd[99] - qd[1],
                   IR_c2=s['IR'][1], IR_diff=s['IR'][99] - s['IR'][1],
                   Tmax_mean=s['Tmax'][1:100].mean(), Tavg_mean=s['Tavg'][1:100].mean(),
                   chargetime_mean=s['chargetime'][1:6].mean())
    if not np.isnan(c['cycle_life']) and n > 120:
        row['knee'] = find_knee(qd[1:int(min(c['cycle_life'], n - 1))])
    rows.append(row)

feat = pd.DataFrame(rows)
feat['log_life'] = np.log10(feat['cycle_life'])
print(feat.groupby('batch').agg(cells=('cell', 'count'),
                                nan_life=('cycle_life', lambda x: x.isna().sum()),
                                life_mean=('cycle_life', 'mean'),
                                life_min=('cycle_life', 'min'),
                                life_max=('cycle_life', 'max')).round(0))
print('\n[cycle_life NaN 셀] - 왜 NaN인지 확인')
print(feat[feat['cycle_life'].isna()][['cell', 'policy', 'n_cycles', 'last_QD']]
      .assign(reached_EOL=lambda d: d['last_QD'] < EOL_AH))

# ===== [셀 B] Q1. Cycle Life 분포 (배치 비교) =====
valid = feat.dropna(subset=['cycle_life']).copy()
BCOL = {'b1': 'tab:blue', 'b2': 'tab:orange', 'b3': 'tab:green'}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
bins = np.linspace(valid['cycle_life'].min(), valid['cycle_life'].max(), 25)
for b, g in valid.groupby('batch'):
    axes[0].hist(g['cycle_life'], bins=bins, alpha=0.5, color=BCOL[b], label=f'{b} (n={len(g)})')
axes[0].axvline(500, ls='--', c='gray'); axes[0].axvline(1000, ls='--', c='gray')
axes[0].set(title='Cycle Life Histogram by Batch', xlabel='Cycle Life', ylabel='num of cells')
axes[0].legend()
valid.boxplot(column='cycle_life', by='batch', ax=axes[1])
axes[1].set(title='Cycle Life Box Plot by Batch', xlabel='', ylabel='Cycle Life')
plt.suptitle(''); plt.tight_layout(); _show('eda_q1_life_dist')

valid['life_group'] = pd.cut(valid['cycle_life'], [0, 500, 1000, np.inf],
                             labels=['short(<500)', 'mid', 'long(>1000)'])
print(pd.crosstab(valid['batch'], valid['life_group'], normalize='index').round(3) * 100, '\n')
print(valid.groupby('batch')['cycle_life'].describe().round(0))

# IQR 기준 이상치 셀
out = []
for b, g in valid.groupby('batch'):
    q1, q3 = g['cycle_life'].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    out.append(g[(g['cycle_life'] < lo) | (g['cycle_life'] > hi)])
print('\n[IQR 이상치 셀]')
print(pd.concat(out)[['cell', 'batch', 'policy', 'cycle_life', 'C_avg', 'QD_c2', 'Tmax_mean']])

# ===== [셀 C] Q2. 열화 곡선 + knee point =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
norm = plt.Normalize(valid['log_life'].min(), valid['log_life'].max())
for ax, b in zip(axes, ['b1', 'b2', 'b3']):
    for cid in valid.loc[valid['batch'] == b, 'cell']:
        qd = pd.Series(cells_all[cid]['summary']['QDischarge'][1:]).rolling(5, min_periods=1).median()
        ax.plot(np.arange(2, len(qd) + 2), qd, lw=0.8,
                color=cm.coolwarm_r(norm(np.log10(cells_all[cid]['cycle_life']))))
    ax.axhline(EOL_AH, c='k', ls='--', lw=1, label='EOL 0.88Ah')
    ax.set(title=f'{b} QD Curve', xlabel='Cycle', ylim=(0.85, 1.12))
axes[0].set_ylabel('QD (Ah)'); axes[0].legend()
plt.tight_layout(); _show('eda_q2_qd_curves')

# 정규화 : x = cycle / cycle_life (0~1) -> 열화 '모양'이 같은지 비교
fig, ax = plt.subplots(figsize=(10, 5))
for cid, r in valid.set_index('cell').iterrows():
    qd = pd.Series(cells_all[cid]['summary']['QDischarge'][1:int(r.cycle_life)]).rolling(5, min_periods=1).median()
    ax.plot(np.linspace(0, 1, len(qd)), qd / qd.iloc[:5].mean(), color=BCOL[r.batch], lw=0.5, alpha=0.6)
ax.set(title='Normalized QD (cycle / cycle_life)', xlabel='life fraction', ylabel='QD / QD_init', ylim=(0.75, 1.03))
plt.tight_layout(); _show('eda_q2_normalized')

kv = valid.dropna(subset=['knee'])
kv = kv.assign(knee_ratio=kv['knee'] / kv['cycle_life'])
print(kv.groupby('batch')['knee_ratio'].describe().round(2))
print(f"knee vs cycle_life Pearson r = {kv['knee'].corr(kv['cycle_life']):.3f}")

# ===== [셀 D] Q3. ΔQ(V) = Q100 - Q10 =====
dv = valid.dropna(subset=['dQ_logvar'])
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, b in zip(axes, ['b1', 'b2', 'b3']):
    for cid in dv.loc[dv['batch'] == b, 'cell']:
        c = cells_all[cid]
        ax.plot(c['Vdlin'], c['Qdlin'][100] - c['Qdlin'][10], lw=0.8,
                color=cm.coolwarm_r(norm(np.log10(c['cycle_life']))))
    ax.set(title=f'{b} ΔQ(V) = Q100 - Q10', xlabel='Voltage (V)')
axes[0].set_ylabel('ΔQ (Ah)')
sm = plt.cm.ScalarMappable(cmap='coolwarm_r', norm=norm)
fig.colorbar(sm, ax=axes, label='log10(cycle_life)')
_show('eda_q3_dq_curves')

fig, ax = plt.subplots(figsize=(8, 6))
for b, g in dv.groupby('batch'):
    r = g['dQ_logvar'].corr(g['log_life'])
    ax.scatter(g['dQ_logvar'], g['cycle_life'], color=BCOL[b], label=f'{b} (r={r:.2f})', alpha=0.8)
ax.set_yscale('log')
ax.set(title=f"log10 Var(ΔQ) vs Cycle Life  (all r={dv['dQ_logvar'].corr(dv['log_life']):.2f})",
       xlabel='log10 Var(ΔQ)', ylabel='Cycle Life (log scale)')
ax.legend(); plt.tight_layout(); _show('eda_q3_dq_scatter')

dv = dv.assign(group=np.where(dv['cycle_life'] > 1000, 'long', np.where(dv['cycle_life'] < 500, 'short', 'mid')))
print(dv.groupby('group')[['dQ_logvar', 'dQ_min', 'dQ_mean', 'dQ_skew', 'dQ_kurt']].mean().round(3))

# ===== [셀 E] Q4. 충전 조건(C-rate) vs 수명 =====
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for b, g in valid.groupby('batch'):
    axes[0].scatter(g['C_avg'], g['cycle_life'], color=BCOL[b], label=b, alpha=0.8)
    axes[1].scatter(g['C1'], g['cycle_life'], color=BCOL[b], label=b, alpha=0.8)
axes[0].set(title=f"avg C-rate (0~80%) vs Cycle Life  r={valid['C_avg'].corr(valid['cycle_life']):.2f}",
            xlabel='avg C-rate (0~80% SOC)', ylabel='Cycle Life')
axes[1].set(title=f"1st step C-rate vs Cycle Life  r={valid['C1'].corr(valid['cycle_life']):.2f}",
            xlabel='C1', ylabel='Cycle Life')
for ax in axes: ax.legend()
plt.tight_layout(); _show('eda_q4_crate')

pol = valid.groupby(['batch', 'policy'])['cycle_life'].agg(['mean', 'std', 'count'])
print('[정책 내 편차가 큰 정책 Top 10] - 같은 조건인데 수명이 다른 경우')
print(pol[pol['count'] >= 2].sort_values('std', ascending=False).head(10).round(0))
print('\n[파싱 안 된 정책]', valid.loc[valid['C_avg'].isna(), 'policy'].unique())

# ===== [셀 F] Q5. 상관관계 / 다중공선성 =====
FEATS = ['dQ_logvar', 'dQ_min', 'dQ_mean', 'dQ_skew', 'dQ_kurt',
         'QD_c2', 'QD_max_minus_c2', 'QD_slope_2_100', 'QD_c100_minus_c2',
         'IR_c2', 'IR_diff', 'Tmax_mean', 'Tavg_mean', 'chargetime_mean', 'C_avg']
fv = valid.dropna(subset=['dQ_logvar'])
corr_tab = pd.DataFrame({b: g[FEATS].corrwith(g['log_life']) for b, g in fv.groupby('batch')})
corr_tab['all'] = fv[FEATS].corrwith(fv['log_life'])
corr_tab = corr_tab.reindex(corr_tab['all'].abs().sort_values(ascending=False).index)
print('[log10(cycle_life)와의 Pearson 상관 : 배치별]')
print(corr_tab.round(2))

cm_ = fv[FEATS].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm_, cmap='RdBu_r', vmin=-1, vmax=1); plt.colorbar(im)
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=60, ha='right')
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        ax.text(j, i, f'{cm_.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(cm_.iloc[i, j]) > 0.6 else 'black')
ax.set_title('Feature Correlation (multicollinearity check)')
plt.tight_layout(); _show('eda_q5_corr_heatmap')

pairs = cm_.where(np.triu(np.ones(cm_.shape, dtype=bool), 1)).stack()
print('[|r| > 0.8 피처 쌍]')
print(pairs[pairs.abs() > 0.8].sort_values(key=abs, ascending=False).round(2))
