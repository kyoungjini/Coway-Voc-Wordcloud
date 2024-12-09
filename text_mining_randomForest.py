import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# 현재 파일의 디렉토리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(__file__)

# 텍스트 정규화 함수 정의
def preprocess_text(text):
  text = re.sub(r'[^\w\s]', '', text)
  stop_words = set(stopwords.words('korean'))
  text = ' '.join([word for word in text.split() if word not in stop_words])
  return text

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

# Sentence Transformer 모델 불러오기
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트를 임베딩으로 변환 (숫자 벡터로 변환)
x_train_embeddings = model.encode(x_train.tolist(), show_progress_bar=True)
x_test_embeddings = model.encode(x_test.tolist(), show_progress_bar=True)

# SMOTE 적용
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_embeddings, y_train)

# 로지스틱 회귀의 하이퍼파라미터 튜닝
param_grid = {
  'logisticregression__C': [0.01, 0.1, 1, 10, 100],
  'logisticregression__penalty': ['l1', 'l2'],
  'logisticregression__solver': ['liblinear', 'saga']
}

# 로지스틱 회귀 분류기 사용 + 클래스 가중치 설정
logistic_pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))

# 그리드 서치로 최적의 하이퍼파라미터 찾기
grid_search = GridSearchCV(logistic_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_resampled, y_train_resampled)

# 최적의 파라미터 출력
print("최적의 하이퍼파라미터: ", grid_search.best_params_)

# 최적의 모델로 테스트 데이터 예측
y_pred_logistic = grid_search.best_estimator_.predict(x_test_embeddings)

# 성능 평가
print("Logistic Regression 성능 평가: ")
print(classification_report(y_test, y_pred_logistic))

# Random Forest 적용
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_classifier.fit(x_train_resampled, y_train_resampled)

# Random Forest로 예측
y_pred_rf = rf_classifier.predict(x_test_embeddings)

# 성능 평가
print("Random Forest 성능 평가: ")
print(classification_report(y_test, y_pred_rf))