import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from nltk.corpus import stopwords
import re
import nltk

nltk.download('stopwords')

# 텍스트 정규화 함수 정의
def preprocess_text(text):
  text = re.sub(r'[^\w\s]', '', text)
  stop_words = set(stopwords.words('korean'))
  text = ' '.join([word for word in text.split() if word not in stop_words])
  return text

# 현재 파일의 디렉토리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(__file__)

# 엑셀 파일 경로 설정
file_path = os.path.join(current_dir, "data/분석 데이터_암호해제본.xlsx")

# 엑셀 파일 읽기
df = pd.read_excel(file_path, engine='openpyxl')

# 제외할 카테고리 리스트
exclude_categories = ["AS", "상담", "서비스지점", "채권/집금"]

# 카테고리 필터링: 제외할 카테고리를 제거한 새로운 데이터프레임 생성
df_filtered = df[~df['상위접수유형 텍스트'].isin(exclude_categories)]

# 필요한 열 선택
texts = df_filtered['접수내역_LONG TEXT']
labels = df_filtered['상위접수유형 텍스트']

# 텍스트 데이터를 훈련 및 테스트 세트로 분리 (80% 훈련, 20% 테스트)
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 레이블을 숫자로 변환 (xgb 관련 오류 수정 위함)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Sentence Transformer 모델 불러오기
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트를 임베딩으로 변환 (숫자 벡터로 변환)
x_train_embeddings = model.encode(x_train.tolist(), show_progress_bar=True)
x_test_embeddings = model.encode(x_test.tolist(), show_progress_bar=True)

# SMOTE 적용
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_embeddings, y_train_encoded)

# 클래스 1의 개수를 계산
class_1_count = sum(y_train_resampled == 1)

# class_1_count가 0인 경우를 처리
if class_1_count == 0:
  scale_pos_weight = 1  # 클래스 불균형이 없을 때
else:
  scale_pos_weight = len(y_train_resampled) / class_1_count

# 2. XGBoost 모델 사용
xgb_classifier = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)

param_grid_xgb = {
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.3],
    'xgbclassifier__max_depth': [3, 6, 9],
    'xgbclassifier__n_estimators': [100, 200, 300],
    # 'xbgclassifier__scale_pos_weight': [scale_pos_weight]
}

# XGBoost 파이프라인 생성
xgb_pipeline = make_pipeline(StandardScaler(), xgb_classifier)

# 그리드 서치로 XGBoost 최적의 하이퍼파라미터 찾기
grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(x_train_resampled, y_train_resampled)

# 최적의 파라미터 출력
print("XGBoost 최적의 하이퍼파라미터: ", grid_search_xgb.best_params_)

# XGBoost 모델로 테스트 데이터 예측
y_pred_xgb = grid_search_xgb.best_estimator_.predict(x_test_embeddings)

# 성능 평가
print("XGBoost 성능 평가: ")
print(classification_report(y_test_encoded, y_pred_xgb))