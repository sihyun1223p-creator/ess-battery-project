"""추가 실험 3, 4 (공식 성능 표와 별개)
3) Valid 분할 안정성 : Hold-out 셀을 다르게 뽑아도 같은 모델이 선택되는가?
4) 조기 예측 민감도 : 100 사이클보다 적은 사이클만으로도 예측이 되는가?
실행 :  python src/robustness.py   또는 노트북에서  from robustness import run_robustness; rb = run_robustness()
"""
import os, sys, warnings, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from config import DATA_DIR, RESULT_DIR, SEED
from preprocess import load_cells, clean_cells, first_valid_idx
from features import build_feature_table, FEATURE_SETS
from train import get_candidates, mape

warnings.filterwarnings('ignore')
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
SHORT_LIFE = 550


def split_stability(df, n_rep=30):
    """실험 3 : Batch 1의 Train/Valid 분할(정책 단위)을 n_rep번 바꿔 가며 모델 선택을 반복"""
    b1, b2 = [df[df.batch == b].reset_index(drop=True) for b in ('b1', 'b2')]
    rows = []
    for seed in range(n_rep):
        tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(b1, groups=b1['policy']))
        tr, va = b1.iloc[tr_i], b1.iloc[va_i]
        gkf = GroupKFold(n_splits=min(5, tr['policy'].nunique()))
        for name, fset, model, grid in get_candidates(verbose=False):
            cols = FEATURE_SETS[fset]
            if grid:
                model = GridSearchCV(model, grid, cv=gkf, scoring='neg_mean_absolute_error').fit(
                    tr[cols], tr['y'], groups=tr['policy']).best_estimator_
            model.fit(tr[cols], tr['y'])
            rows.append(dict(seed=seed, model=name, n_valid=len(va),
                             valid=mape(va['y'], model.predict(va[cols])), test=mape(b2['y'], model.predict(b2[cols]))))
        print(f'\r  분할 {seed + 1}/{n_rep}', end='')
    print()
    long = pd.DataFrame(rows)
    win = long.loc[long.groupby('seed')['valid'].idxmin()]            # 분할마다 Valid가 가장 낮은 모델
    g = long.groupby('model', sort=False)
    table = pd.DataFrame({
        '모델': list(g.groups),
        'Valid 1위 횟수': [int((win['model'] == m).sum()) for m in g.groups],
        'Valid MAPE 평균(%)': g['valid'].mean().round(2).values,
        'Valid MAPE 표준편차': g['valid'].std().round(2).values,
        'Valid MAPE 최소~최대': [f'{a:.1f} ~ {b:.1f}' for a, b in zip(g['valid'].min(), g['valid'].max())],
        'Test MAPE 평균(%)': g['test'].mean().round(2).values,
        'Test MAPE 표준편차': g['test'].std().round(2).values,
    })
    summary = {'반복 횟수': n_rep, 'Valid 셀 수 범위': f"{long['n_valid'].min()} ~ {long['n_valid'].max()}",
               '선택된 모델의 Test MAPE 평균(%)': round(float(win['test'].mean()), 2)}
    return table, long, summary


def early_prediction(cells, df, ns=(20, 30, 40, 50, 60, 80, 100)):
    """실험 4 : ΔQ(V) = Q_n − Q_10 에서 n을 줄여 가며 같은 Linear 모델을 학습·평가 (분할은 train.py와 동일)"""
    rows = []
    b1_idx = df.index[df.batch == 'b1']
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(b1_idx, groups=df.loc[b1_idx, 'policy']))
    tr_cells, va_cells = set(df.loc[b1_idx[tr_i], 'cell']), set(df.loc[b1_idx[va_i], 'cell'])
    for n in ns:
        x = {}
        for cid in df['cell']:
            c = cells[cid]; off = first_valid_idx(c)
            q10, qn = c['Qdlin'][off + 9], c['Qdlin'][off + n - 1]
            if q10 is not None and qn is not None:
                x[cid] = np.log10(np.var(qn - q10))
        d = df[['cell', 'batch', 'cycle_life', 'y']].assign(x=df['cell'].map(x)).dropna(subset=['x'])
        tr, va, b2 = d[d.cell.isin(tr_cells)], d[d.cell.isin(va_cells)], d[d.batch == 'b2']
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[['x']], tr['y'])
        p2 = m.predict(b2[['x']])
        short = b2['cycle_life'] < SHORT_LIFE
        rows.append({'사용 사이클 n': n, 'Batch 1 상관 r': round(d[d.batch == 'b1'][['x', 'y']].corr().iloc[0, 1], 3),
                     'Valid MAPE(%)': round(mape(va['y'], m.predict(va[['x']])), 2),
                     'Test MAPE(%)': round(mape(b2['y'], p2), 2),
                     f'Batch 2 단수명(<{SHORT_LIFE}) 판별 AUC': round(roc_auc_score(short, -p2), 3) if 0 < short.sum() < len(b2) else np.nan})
    return pd.DataFrame(rows)


def _md(df):
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    return lines + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.values]


def run_robustness(data_dir=DATA_DIR, out_dir=RESULT_DIR, n_rep=30, show=True):
    os.makedirs(out_dir, exist_ok=True)
    cells, _ = clean_cells(load_cells(data_dir))
    df = build_feature_table(cells)

    print(f'[실험 3] Valid 분할 안정성 ({n_rep}회 반복, 2~3분 소요)')
    table, long, summary = split_stability(df, n_rep)
    print(table.to_string(index=False)); print(summary)
    print('\n[실험 4] 조기 예측 민감도 (Linear, 피처 = log Var(Q_n − Q_10))')
    early = early_prediction(cells, df)
    print(early.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = list(table['모델'])
    axes[0].boxplot([long[long.model == m]['valid'].values for m in names])
    axes[0].set_xticks(range(1, len(names) + 1))
    axes[0].set_xticklabels([m.replace(' (', '\n(') for m in names])   # matplotlib 버전에 관계없이 동작
    axes[0].set(ylabel='Valid MAPE (%)', title=f'분할 {n_rep}회 반복 시 Valid MAPE 분포')
    n = early['사용 사이클 n']
    axes[1].plot(n, early['Valid MAPE(%)'], marker='o', label='Valid (Batch 1)')
    axes[1].plot(n, early['Test MAPE(%)'], marker='s', label='Test (Batch 2)')
    axes[1].set(xlabel='사용한 사이클 수 n  (ΔQ = Q_n − Q_10)', ylabel='MAPE (%)', title='조기 예측 민감도', ylim=(0, None))
    axes[1].set_xticks(list(n)); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'robustness.png'), dpi=150)

    md = [f'실험 3. Valid 분할 안정성 ({n_rep}회 반복)', ''] + _md(table)
    md += [''] + [f'- {k} : {v}' for k, v in summary.items()]
    md += ['', '실험 4. 조기 예측 민감도', ''] + _md(early)
    open(os.path.join(out_dir, 'robustness.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    if show:
        plt.show()
    print(f'\n결과 저장 → {out_dir}/robustness.md, robustness.png')
    return dict(stability=table, summary=summary, early=early)


if __name__ == '__main__':
    run_robustness(show=False)
