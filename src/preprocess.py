"""데이터 로드 및 정제
1) .mat → 셀 단위 dict(pkl) 변환 (summary 전체 + Qdlin 초기 0~100 사이클)
2) 라벨을 신뢰할 수 없는 셀 제외
"""
import os, gc, pickle, logging
import numpy as np, pandas as pd
from config import DATA_DIR, EOL_AH

BATCH_FILES = {
    'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}
SUM_KEYS = ['QDischarge', 'QCharge', 'IR', 'Tmax', 'Tavg', 'Tmin', 'chargetime']


def convert_mat_to_pkl(data_dir=DATA_DIR, max_cycle=101):
    """원본 .mat(2~3GB)을 배치별로 읽어 필요한 부분만 pkl로 저장 (최초 1회만 실행)"""
    import mat73
    logging.getLogger().setLevel(logging.CRITICAL)
    for tag, fname in BATCH_FILES.items():
        b = mat73.loadmat(os.path.join(data_dir, fname))['batch']
        cells = {}
        for i in range(len(b['cycle_life'])):
            cyc, summ = b['cycles'][i], b['summary'][i]
            try:
                life = float(np.asarray(b['cycle_life'][i]).squeeze())
            except Exception:
                life = np.nan
            cells[f'{tag}c{i}'] = {
                'batch': tag, 'cycle_life': life, 'policy': str(b['policy_readable'][i]),
                'summary': {k: np.asarray(summ[k]).ravel() for k in SUM_KEYS if k in summ},
                'Qdlin': [None if q is None else np.asarray(q, dtype=np.float32) for q in cyc['Qdlin'][:max_cycle]],
                'Vdlin': np.asarray(b['Vdlin'][i]),
            }
        with open(os.path.join(data_dir, f'{tag}_cells.pkl'), 'wb') as f:
            pickle.dump(cells, f)
        print(f'[{tag}] {len(cells)} cells saved')
        del b, cells; gc.collect()


def load_cells(data_dir=DATA_DIR):
    cells = {}
    for tag in BATCH_FILES:
        with open(os.path.join(data_dir, f'{tag}_cells.pkl'), 'rb') as f:
            cells.update(pickle.load(f))
    return cells


def first_valid_idx(cell):
    """첫 번째 유효 사이클의 index.
    배치에 따라 index 0이 빈 값(QD=0, Qdlin=None)인 경우가 있어, 같은 index를 그대로 비교하면
    사이클 번호가 어긋난다. 모든 셀을 '첫 유효 사이클' 기준으로 맞추기 위해 사용한다."""
    qd, ql = cell['summary']['QDischarge'], cell['Qdlin']
    for i in range(min(len(qd), len(ql))):
        if qd[i] > 0.5 and ql[i] is not None:
            return i
    return None


def clean_cells(cells, min_cycles=100):
    """라벨을 신뢰할 수 없는 셀 제외. (kept dict, 셀별 판정 DataFrame) 반환

    제외 기준
    - no_label      : cycle_life가 NaN
    - too_short     : 유효 사이클이 100개 미만 (초기 100 사이클 피처 계산 불가)
    - not_reach_EOL : 라벨은 있으나 방전용량이 EOL(0.88Ah) 부근까지 떨어진 적이 없음
                      → 실험이 중간에 끝난 셀로, cycle_life가 실제 수명보다 짧게 기록됨(중도절단)
    """
    kept, rows = {}, []
    for cid, c in cells.items():
        qd = c['summary']['QDischarge']
        off = first_valid_idx(c)
        n_valid = 0 if off is None else len(qd) - off
        qmin = np.nan if off is None else pd.Series(qd[off:]).rolling(3, min_periods=1).median().min()
        if np.isnan(c['cycle_life']):
            reason = 'no_label'
        elif off is None or n_valid < min_cycles or len(c['Qdlin']) < off + min_cycles:
            reason = 'too_short'
        elif qmin > EOL_AH + 0.02:
            reason = 'not_reach_EOL'
        else:
            reason = ''
            kept[cid] = c
        rows.append(dict(cell=cid, batch=c['batch'], policy=c['policy'], cycle_life=c['cycle_life'],
                         first_valid_idx=off, n_cycles=n_valid, min_QD=qmin, excluded=reason))
    return kept, pd.DataFrame(rows)
