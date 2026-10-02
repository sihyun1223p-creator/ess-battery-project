"""경로 설정 : DATA_DIR 한 곳만 본인 환경에 맞게 수정하면 됩니다."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# b1/b2/b3_cells.pkl 이 들어 있는 폴더. 기본값은 프로젝트의 data/ (환경변수 ESS_DATA_DIR로 변경 가능)
DATA_DIR = os.environ.get('ESS_DATA_DIR', os.path.join(ROOT, 'data'))
RESULT_DIR = os.path.join(ROOT, 'results')
SEED = 42
EOL_AH = 0.88          # 공칭 1.1Ah의 80%
TARGET_MAPE = 9.1      # 원논문 Regression 성능 (%)
