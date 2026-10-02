"""피처 엔지니어링 : 초기 100 사이클만 사용 (모든 index는 '첫 유효 사이클' 기준)"""
import re
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from preprocess import first_valid_idx

# DAY 1 EDA 결론을 반영한 피처 세트
FEATURE_SETS = {
    # 3개 배치 모두에서 log 수명과 일관된 강한 상관 (r = -0.79 ~ -0.90)
    'variance': ['dQ_logvar'],
    # ΔQ 통계 + 초기 용량/저항 (배치 간 부호가 뒤집히지 않는 피처)
    'main': ['dQ_logvar', 'dQ_min', 'dQ_skew', 'dQ_kurt',
             'QD_c2', 'QD_max_minus_c2', 'QD_slope_2_100', 'IR_c2'],
    # 비교용 : 배치마다 상관 부호가 뒤집힌 피처(충전시간, C-rate, 온도)까지 포함
    'full': ['dQ_logvar', 'dQ_min', 'dQ_mean', 'dQ_skew', 'dQ_kurt',
             'QD_c2', 'QD_max_minus_c2', 'QD_slope_2_100', 'IR_c2', 'IR_diff',
             'Tavg_mean', 'chargetime_mean', 'C_avg'],
}


def parse_avg_crate(policy):
    """'5.4C(50%)-3.6C' → 0~80% SOC 구간 평균 C-rate"""
    m = re.match(r'\s*([\d.]+)C\((\d+)%\)-([\d.]+)C', policy)
    if not m:
        return np.nan
    c1, s, c2 = float(m.group(1)), min(float(m.group(2)) / 100, 0.8), float(m.group(3))
    return 0.8 / (s / c1 + max(0.8 - s, 0) / c2)


def make_features(cell):
    off = first_valid_idx(cell)                   # 첫 유효 사이클 = cycle 1
    s = cell['summary']
    cyc = lambda n: off + n - 1                   # cycle 번호 → index
    q10, q100 = cell['Qdlin'][cyc(10)], cell['Qdlin'][cyc(100)]
    if q10 is None or q100 is None:
        return None
    dq = q100 - q10                               # ΔQ(V) = Q100 - Q10
    qd = pd.Series(s['QDischarge'][cyc(2):cyc(100) + 1]).rolling(5, center=True, min_periods=1).median().values
    ir = s['IR'][cyc(2):cyc(100) + 1]
    return dict(
        dQ_logvar=np.log10(np.var(dq)),
        dQ_min=np.log10(abs(dq.min()) + 1e-9),
        dQ_mean=np.log10(abs(dq.mean()) + 1e-9),
        dQ_skew=skew(dq), dQ_kurt=kurtosis(dq),
        QD_c2=qd[0], QD_max_minus_c2=qd.max() - qd[0],
        QD_slope_2_100=np.polyfit(np.arange(2, 101), qd, 1)[0],
        IR_c2=ir[0], IR_diff=ir[-1] - ir[0],
        Tavg_mean=s['Tavg'][cyc(2):cyc(100) + 1].mean(),
        chargetime_mean=s['chargetime'][cyc(2):cyc(6) + 1].mean(),
        C_avg=parse_avg_crate(cell['policy']),
    )


def build_feature_table(cells):
    rows = []
    for cid, c in cells.items():
        f = make_features(c)
        if f is not None:
            rows.append(dict(cell=cid, batch=c['batch'], policy=c['policy'], cycle_life=c['cycle_life'], **f))
    df = pd.DataFrame(rows)
    df['y'] = np.log10(df['cycle_life'])          # target
    return df
