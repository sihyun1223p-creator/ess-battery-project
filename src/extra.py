"""추가 실험 (공식 성능 표와 별개) : 최종 모델의 Batch 2 예측값을 그대로 사용한다.
1) 순위 예측 성능 : 절대 수명은 틀려도 '어느 셀이 먼저 열화되는가'의 순서는 맞는가?
2) 배치 보정 실험 : Batch 2 셀 k개의 실제 수명을 알면 오차가 얼마나 줄어드는가?
실행 :  python src/extra.py   또는 노트북에서  from extra import run_extra; ex = run_extra(out)
"""
import os, sys, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from config import RESULT_DIR, SEED

if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
SHORT_LIFE = 550          # 원논문의 장/단수명 분류 기준


def _mape(y, p):
    return float(np.mean(np.abs(p - y) / y) * 100)


def ranking_experiment(err):
    """실험 1 : 예측 수명 순위와 실제 수명 순위의 일치도"""
    is_new = err['policy'].str.contains('newstructure')
    rows = []
    for name, g in [('Batch 2 전체', err), ('newstructure 제외', err[~is_new]), ('newstructure만', err[is_new])]:
        if len(g) < 4:
            continue
        y, p = g['cycle_life'].values, g['pred'].values
        k = max(1, round(len(g) * 0.25))                       # 수명이 짧은 하위 25%
        hit = len(set(np.argsort(y)[:k]) & set(np.argsort(p)[:k]))
        short = y < SHORT_LIFE
        auc = roc_auc_score(short, -p) if 0 < short.sum() < len(g) else np.nan
        rows.append({'대상': name, '셀 수': len(g), 'Spearman 순위상관': round(spearmanr(p, y)[0], 3),
                     f'하위 25% 선별 적중': f'{hit}/{k}', f'단수명(<{SHORT_LIFE}) 판별 AUC': round(auc, 3),
                     'MAPE(%)': round(_mape(y, p), 2)})
    return pd.DataFrame(rows)


def calibration_experiment(err, ks=(1, 2, 3, 5, 10), n_rep=1000):
    """실험 2 : Batch 2에서 무작위 k셀의 실제 수명으로 절편(log 스케일 평균 오차)만 보정 → 나머지 셀에서 평가.
    비교 기준 '평균만 사용'은 모델 없이 그 k셀의 평균(기하평균) 수명을 나머지 전부의 예측값으로 쓴 경우."""
    rng = np.random.default_rng(SEED)
    y, p = err['cycle_life'].values, err['pred'].values
    r = np.log10(y) - np.log10(p)                              # log 스케일 오차
    n, rows = len(y), []
    for k in ks:
        if k >= n - 5:
            continue
        cal, base = [], []
        for _ in range(n_rep):
            idx = rng.choice(n, k, replace=False)
            rest = np.setdiff1d(np.arange(n), idx)
            cal.append(_mape(y[rest], p[rest] * 10 ** r[idx].mean()))
            base.append(_mape(y[rest], np.full(len(rest), 10 ** np.log10(y[idx]).mean())))
        rows.append({'보정에 쓴 셀 수 k': k, '보정 후 MAPE 평균(%)': round(np.mean(cal), 2),
                     '5~95% 범위': f'{np.percentile(cal, 5):.1f} ~ {np.percentile(cal, 95):.1f}',
                     '평균만 사용 MAPE(%)': round(np.mean(base), 2),
                     '_lo': np.percentile(cal, 5), '_hi': np.percentile(cal, 95)})
    ref = {'보정 없음': round(_mape(y, p), 2), '전체 셀로 보정(참고용 상한)': round(_mape(y, p * 10 ** r.mean()), 2)}
    return pd.DataFrame(rows), ref


def _md(df):
    cols = [c for c in df.columns if not c.startswith('_')]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    return lines + ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df[cols].values]


def run_extra(out=None, out_dir=RESULT_DIR, show=True):
    if out is None:
        from train import run
        out = run(show=False)
    err = out['errors'].reset_index(drop=True)
    rank = ranking_experiment(err)
    cal, ref = calibration_experiment(err)

    print(f"\n===== 추가 실험 (최종 모델 : {out['best']}, Batch 2 {len(err)}셀) =====")
    print('\n[실험 1] 순위 예측 성능'); print(rank.to_string(index=False))
    print('\n[실험 2] 배치 보정 (무작위 k셀 선택을 1,000회 반복, 나머지 셀에서 평가)')
    print(cal.drop(columns=['_lo', '_hi']).to_string(index=False)); print(ref)

    # 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ya, pa = err['cycle_life'].rank(), err['pred'].rank()
    is_new = err['policy'].str.contains('newstructure')
    axes[0].scatter(ya[~is_new], pa[~is_new], c='tab:orange', label='Batch 2')
    if is_new.any():
        axes[0].scatter(ya[is_new], pa[is_new], c='tab:purple', marker='s', label='Batch 2 (newstructure)')
    axes[0].plot([1, len(err)], [1, len(err)], 'k--', lw=1)
    axes[0].set(xlabel='실제 수명 순위 (1 = 가장 짧음)', ylabel='예측 수명 순위',
                title=f"순위 일치도 (Spearman {rank.iloc[0]['Spearman 순위상관']:.2f})")
    axes[0].legend()
    k = cal['보정에 쓴 셀 수 k']
    axes[1].errorbar(k, cal['보정 후 MAPE 평균(%)'], yerr=[cal['보정 후 MAPE 평균(%)'] - cal['_lo'], cal['_hi'] - cal['보정 후 MAPE 평균(%)']],
                     marker='o', capsize=4, label='모델 + 절편 보정')
    axes[1].plot(k, cal['평균만 사용 MAPE(%)'], marker='s', ls=':', c='gray', label='평균만 사용 (모델 없음)')
    axes[1].axhline(ref['보정 없음'], c='tab:red', ls='--', lw=1, label=f"보정 없음 {ref['보정 없음']}%")
    axes[1].axhline(9.1, c='k', ls=':', lw=1, label='원논문 9.1%')
    axes[1].set(xlabel='보정에 사용한 Batch 2 셀 수 k', ylabel='나머지 셀 MAPE (%)', title='배치 보정 효과', ylim=(0, None))
    axes[1].set_xticks(list(k)); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'extra_experiments.png'), dpi=150)

    md = [f"최종 모델 : {out['best']} / Batch 2 {len(err)}셀", '', '실험 1. 순위 예측 성능', ''] + _md(rank)
    md += ['', '실험 2. 배치 보정 (무작위 k셀, 1,000회 반복)', ''] + _md(cal)
    md += ['', f"- 보정 없음 : {ref['보정 없음']}% / 전체 셀로 보정(참고용 상한) : {ref['전체 셀로 보정(참고용 상한)']}%"]
    open(os.path.join(out_dir, 'extra_experiments.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    if show:
        plt.show()
    print(f'\n결과 저장 → {out_dir}/extra_experiments.md, extra_experiments.png')
    return dict(ranking=rank, calibration=cal.drop(columns=['_lo', '_hi']), reference=ref)


if __name__ == '__main__':
    run_extra(show=False)
