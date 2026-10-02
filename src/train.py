"""모델 학습 및 평가 : Batch 1 학습 → Batch 2 테스트
실행 :  python src/train.py   또는 노트북에서  from train import run; out = run()
"""
import os, sys, warnings, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from config import DATA_DIR, RESULT_DIR, SEED, TARGET_MAPE
from preprocess import load_cells, clean_cells
from features import build_feature_table, FEATURE_SETS

warnings.filterwarnings('ignore')
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def mape(y_log, p_log):
    """log10 예측을 사이클 단위로 되돌린 뒤 MAPE(%)"""
    y, p = 10 ** np.asarray(y_log), 10 ** np.asarray(p_log)
    return float(np.mean(np.abs(p - y) / y) * 100)


def get_candidates(verbose=True):
    en = lambda: make_pipeline(StandardScaler(), ElasticNet(max_iter=100000))
    grid = {'elasticnet__alpha': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
    cands = [
        ('Linear (variance)', 'variance', make_pipeline(StandardScaler(), LinearRegression()), None),
        ('ElasticNet (main)', 'main', en(), grid),
        ('ElasticNet (full)', 'full', en(), grid),
        ('RandomForest (main)', 'main', RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED), None),
    ]
    try:
        from lightgbm import LGBMRegressor
        cands.append(('LightGBM (main)', 'main', LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=7,
                                                              min_child_samples=5, random_state=SEED, verbose=-1), None))
    except Exception as e:      # 미설치(ImportError) 또는 맥에서 libomp 없음(OSError) → LightGBM만 건너뜀
        if verbose: print(f'(LightGBM 생략 : {type(e).__name__})')
    return cands


def run(data_dir=DATA_DIR, out_dir=RESULT_DIR, show=True):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 로드 → 정제 → 피처
    cells, report = clean_cells(load_cells(data_dir))
    report.to_csv(os.path.join(out_dir, 'cleaning_report.csv'), index=False, encoding='utf-8-sig')
    print('[정제] 배치별 제외 사유 (빈칸 = 사용)')
    print(pd.crosstab(report['batch'], report['excluded'].replace('', 'kept')).to_string(), '\n')
    df = build_feature_table(cells)
    b1, b2 = [df[df.batch == b].reset_index(drop=True) for b in ('b1', 'b2')]

    # 2. 분할 : Batch 1 → Train / Valid(Hold-out). 같은 충전 정책의 셀은 한쪽에만 들어가도록 정책 단위로 분리(누수 방지)
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(b1, groups=b1['policy']))
    tr, va = b1.iloc[tr_i].reset_index(drop=True), b1.iloc[va_i].reset_index(drop=True)
    gkf = GroupKFold(n_splits=min(5, tr['policy'].nunique()))
    print(f'[분할] Train {len(tr)} / Valid {len(va)} (Batch 1) · Test {len(b2)} (Batch 2)\n')

    # 3. 학습 / 평가
    perf, preds, models = {}, {}, {}
    for name, fset, model, grid in get_candidates():
        cols = FEATURE_SETS[fset]
        if grid:   # 하이퍼파라미터는 Train 내부 CV로만 선택 (Valid/Test 미사용)
            model = GridSearchCV(model, grid, cv=gkf, scoring='neg_mean_absolute_error').fit(
                tr[cols], tr['y'], groups=tr['policy']).best_estimator_
        cv = [mape(tr['y'].iloc[v], clone(model).fit(tr[cols].iloc[t], tr['y'].iloc[t]).predict(tr[cols].iloc[v]))
              for t, v in gkf.split(tr, groups=tr['policy'])]
        model.fit(tr[cols], tr['y'])
        m_train, m_valid = float(np.mean(cv)), mape(va['y'], model.predict(va[cols]))
        p2 = model.predict(b2[cols])
        m_test = mape(b2['y'], p2)
        perf[name] = {
            'Train (Batch 1 CV)': m_train, 'Valid (Batch 1 Hold-out)': m_valid, 'Test (Batch 2)': m_test,
            # Gap 부호 : (+) = 뒤쪽 성능이 더 나쁨 (MAPE는 오차 지표이므로 '뒤 - 앞'으로 계산)
            'Gap (Train-Valid)': m_valid - m_train, 'Gap (Valid-Test)': m_test - m_valid,
            'Gap (Target-Test)': m_test - TARGET_MAPE,
        }
        preds[name], models[name] = p2, (model, cols)

    res = pd.DataFrame(perf).round(2)
    best = res.loc['Valid (Batch 1 Hold-out)'].idxmin()          # 최종 모델은 Valid 기준으로 선택
    res.to_csv(os.path.join(out_dir, 'model_performance.csv'), encoding='utf-8-sig')
    note = {'Gap (Train-Valid)': '(+) : 과적합 의심', 'Gap (Valid-Test)': '(+) : 배치간 일반화 저하 의심',
            'Gap (Target-Test)': f'Target : 원논문 {TARGET_MAPE}%'}
    md = [f'최종 모델 : {best}', '', '| 구분 | MAPE (%) | 비고 |', '|---|---|---|']
    md += [f"| {i} | {v:+.2f} | {note[i]} |" if i in note else f'| {i} | {v:.2f} | |' for i, v in res[best].items()]
    md += ['', '| 모델 | ' + ' | '.join(res.index) + ' |', '|---|' + '---|' * len(res.index)]
    md += [f'| {m} | ' + ' | '.join(f'{v:.2f}' for v in res[m]) + ' |' for m in res.columns]
    open(os.path.join(out_dir, 'performance.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print(f'[성능] MAPE(%) · Gap(%p, (+) = 뒤쪽이 더 나쁨) · Target = 원논문 {TARGET_MAPE}%')
    print(res.to_string(), f'\n\n[최종 모델 (Valid 기준)] {best}\n')

    # 4. 오류 분석 (최종 모델, Batch 2)
    p2 = preds[best]
    lo, hi = b1['cycle_life'].min(), b1['cycle_life'].max()
    err = b2[['cell', 'policy', 'cycle_life', 'dQ_logvar']].assign(pred=(10 ** p2).round(0))
    err['APE(%)'] = ((err['pred'] - err['cycle_life']).abs() / err['cycle_life'] * 100).round(1)
    err['signed(%)'] = ((err['pred'] - err['cycle_life']) / err['cycle_life'] * 100).round(1)
    err['range'] = np.where(err['cycle_life'] < lo, 'below_train', np.where(err['cycle_life'] > hi, 'above_train', 'in_train'))
    err['dQ_in_train_range'] = err['dQ_logvar'].between(b1['dQ_logvar'].min(), b1['dQ_logvar'].max())
    err.sort_values('APE(%)', ascending=False).to_csv(os.path.join(out_dir, 'test_errors.csv'), index=False, encoding='utf-8-sig')
    print('[오류 분석] Batch 2에서 가장 크게 틀린 셀 Top 8')
    print(err.sort_values('APE(%)', ascending=False).head(8).to_string(index=False))
    print(f'\n[오류 분석] 학습 수명 범위({lo:.0f}~{hi:.0f}) 안/밖 MAPE')
    print(err.groupby('range')['APE(%)'].agg(['mean', 'count']).round(2).to_string())
    print('\n[오류 분석] ΔQ 피처가 학습 범위 안/밖일 때 MAPE')
    print(err.groupby('dQ_in_train_range')['APE(%)'].agg(['mean', 'count']).round(2).to_string(), '\n')

    # 5. 그래프
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(b2['cycle_life'], 10 ** p2, c='tab:orange', label=f"Batch 2 (MAPE {res.loc['Test (Batch 2)', best]:.1f}%)")
    lim = [300, max(b1['cycle_life'].max(), b2['cycle_life'].max()) * 1.1]
    ax.plot(lim, lim, 'k--', lw=1); ax.axvspan(lo, hi, color='tab:blue', alpha=.08, label='Batch 1 train range')
    ax.set(xlim=lim, ylim=lim, xlabel='Observed cycle life', ylabel='Predicted cycle life', title=f'Predicted vs Observed : {best}')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'pred_vs_obs.png'), dpi=150)

    rows = ['Train (Batch 1 CV)', 'Valid (Batch 1 Hold-out)', 'Test (Batch 2)']
    ax = res.loc[rows].T.plot.bar(figsize=(10, 4.5), rot=15)
    ax.axhline(TARGET_MAPE, c='r', ls='--', lw=1, label=f'Paper {TARGET_MAPE}%'); ax.set_ylabel('MAPE (%)'); ax.legend()
    ax.figure.tight_layout(); ax.figure.savefig(os.path.join(out_dir, 'mape_compare.png'), dpi=150)

    m, cols = models['ElasticNet (main)']
    coef = pd.Series(m[-1].coef_, index=cols).sort_values()
    plt.figure()
    ax = coef.plot.barh(figsize=(6.5, 3.8), color=['tab:red' if v < 0 else 'tab:blue' for v in coef])
    ax.set(title='ElasticNet (main) standardized coefficients', xlabel='effect on log10(cycle_life)')
    ax.figure.tight_layout(); ax.figure.savefig(os.path.join(out_dir, 'coef.png'), dpi=150)
    print('[ElasticNet (main) 계수]'); print(coef.round(4).to_string())
    if show:
        plt.show()
    print(f'\n결과 저장 → {out_dir}')
    return dict(performance=res, best=best, errors=err, cleaning=report, features=df)


if __name__ == '__main__':
    run(show=False)
