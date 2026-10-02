최종 모델 : Linear (variance)

| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch 1 CV) | 9.16 | |
| Valid (Batch 1 Hold-out) | 8.09 | |
| Test (Batch 2) | 29.88 | |
| Gap (Train-Valid) | -1.07 | (+) : 과적합 의심 |
| Gap (Valid-Test) | +21.79 | (+) : 배치간 일반화 저하 의심 |
| Gap (Target-Test) | +20.78 | Target : 원논문 9.1% |

| 모델 | Train (Batch 1 CV) | Valid (Batch 1 Hold-out) | Test (Batch 2) | Gap (Train-Valid) | Gap (Valid-Test) | Gap (Target-Test) |
|---|---|---|---|---|---|---|
| Linear (variance) | 9.16 | 8.09 | 29.88 | -1.07 | 21.79 | 20.78 |
| ElasticNet (main) | 5.77 | 19.03 | 30.32 | 13.26 | 11.29 | 21.22 |
| ElasticNet (full) | 6.78 | 18.31 | 34.55 | 11.53 | 16.24 | 25.45 |
| RandomForest (main) | 9.53 | 12.82 | 33.91 | 3.29 | 21.08 | 24.81 |
| LightGBM (main) | 9.60 | 9.39 | 33.52 | -0.20 | 24.13 | 24.42 |
