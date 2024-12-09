import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

# 현재 파일의 디렉토리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(__file__)

# 엑셀 파일 경로 설정
file_path = os.path.join(current_dir, "data/분석 데이터_암호해제본.xlsx")

# 엑셀 파일 읽기
df = pd.read_excel(file_path, engine='openpyxl')

# 제외할 카테고리 리스트
exclude_categories = ["AS", "상담", "서비스지점", "채권/집금", "반환/탈퇴", "결제/수납", "홈케어서비스"]

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

# SVM 모델 사용 (최적의 하이퍼파라미터 설정)
svm_pipeline = make_pipeline(StandardScaler(), SVC(C=10, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42))

# 모델 학습
svm_pipeline.fit(x_train_resampled, y_train_resampled)

# SVM 모델로 테스트 데이터 예측
y_pred_svm = svm_pipeline.predict(x_test_embeddings)

# 성능 평가
print("SVM 성능 평가: ")
print(classification_report(y_test, y_pred_svm, zero_division=0))
