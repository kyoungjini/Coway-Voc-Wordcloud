import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np

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

# Sentence Transformer 모델 불러오기
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트를 임베딩으로 변환 (숫자 벡터로 변환)
x_train_embeddings = model.encode(x_train.tolist(), show_progress_bar=True)
x_test_embeddings = model.encode(x_test.tolist(), show_progress_bar=True)

# SMOTE 적용
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_embeddings, y_train)

# 로지스틱 회귀 분류기 사용 + 클래스 가중치 설정
classifier = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, class_weight='balanced', max_iter=1000))

# # 훈련 데이터로 분류기 학습
# classifier.fit(x_train_embeddings, y_train)

# 훈련 데이터로 분류기 학습
classifier.fit(x_train_resampled, y_train_resampled)

# 테스트 데이터로 예측
y_pred = classifier.predict(x_test_embeddings)

# 분류 성능 평가 및 중복 레이블 확인
print(classification_report(y_test, y_pred))


### LIME 예측 ###

# LIME을 이용한 예측 해석 추가
explainer = LimeTextExplainer(class_names=np.unique(labels))

# 특정 테스트 예제에 대해 LIME을 사용해 해석하기
idx = 0  # 첫 번째 테스트 데이터 샘플 해석 (필요에 따라 변경 가능)
x_sample = x_test.iloc[idx]

def predict_proba(texts):
    # 텍스트를 SentenceTransformer를 통해 임베딩
    embeddings = model.encode(texts, show_progress_bar=False)
    # 로지스틱 회귀 분류기를 사용해 확률 예측
    return classifier.predict_proba(embeddings)

# LIME을 이용해 해당 샘플의 예측 해석
exp = explainer.explain_instance(x_sample, predict_proba, num_features=10)
explanation_list = exp.as_list()

# 해석 결과 출력
print(f"LIME 설명 결과 (테스트 샘플 {idx}):")
for feature, weight in explanation_list:
    print(f"{feature}: {weight}")