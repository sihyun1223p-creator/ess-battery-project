# data

용량 문제로 원본 데이터는 저장소에 포함하지 않습니다.

1. Kaggle "MIT-Stanford Battery Dataset"에서 아래 파일을 내려받습니다.
   - `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (Batch 1, 학습)
   - `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (Batch 2, 테스트)
   - `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (Batch 3, 추가 검증)
2. 내려받은 `.mat` 파일을 이 `data/` 폴더에 둡니다. (다른 위치라면 환경변수 `ESS_DATA_DIR`로 지정)
3. 최초 1회 변환합니다 (배치당 수 분 소요, `b1/b2/b3_cells.pkl` 생성).

```bash
python -c "import sys; sys.path.insert(0,'src'); from preprocess import convert_mat_to_pkl; convert_mat_to_pkl()"
```
