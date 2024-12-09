import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import numpy as np
import matplotlib.font_manager as fm
from PIL import Image

# 현재 파일의 디렉토리를 기준으로 상대 경로 설정
current_dir = os.path.dirname(__file__)

# 한글 폰트 경로 설정
font_path = os.path.join(current_dir, "Fonts/NanumGothic.ttf")

# 마스크 이미지 로드
mask_image_path = os.path.join(current_dir, "data/wordcloud_water.png")
mask = np.array(Image.open(mask_image_path))

# 한국어 불용어 목록
korean_stopwords = [
  "은", "는", "이", "가", "의", "를", "을",
  "에", "으로", "도", "과", "수", "다"
  "그리고", "하지만", "때문에", "인한", "인해", "왜",
  "또", "또는", "및", "후", "고", "나고",
  "안내", "안내드렸으나", "안내했으나", "드렸으나",
  "확인", "확인됨", "확인후",
  "부탁드립니다", "확인부탁드립니다", "고객", "고객님", "고객은", "고객이",
  "요청하십니다", "연락",
  "함", "하심", "하십니다", "하시며", "하고", "했으나", "하였으나",
  "본인", "자녀", "배우자", "부모님",
  "미수긍", "정수기", "요청", "요청으로", "항의", "불만",
  "해당", "매", "코웨이", "코웨이에서", "아직", "받았으나",
  "주장", "클레임", "물", "이내", "안", "ㅇ", "다", "너무", "다른",
  "점검을", "a", "갔다고", "받은"
]

# 엑셀 파일 경로 설정
file_path = os.path.join(current_dir, "data/분석 데이터_암호해제본.xlsx")

# 엑셀 파일 읽기
df = pd.read_excel(file_path, engine='openpyxl')

# 제외할 카테고리 리스트 (데이터 수 적음)
exclude_categories = ["AS", "상담", "서비스지점", "채권/집금"]

# 카테고리 필터링: 제외할 카테고리를 제거한 새로운 데이터프레임 생성
df_filtered = df[~df['상위접수유형 텍스트'].isin(exclude_categories)]

# 필요한 열 선택
texts = df_filtered['접수내역_LONG TEXT'].fillna("")
categories = df_filtered['상위접수유형 텍스트']

# TF-IDF 기반 키워드 추출
tfidf = TfidfVectorizer(stop_words=korean_stopwords, token_pattern=r'\b[^\d\W]+\b' )
tfidf_matrix = tfidf.fit_transform(texts)
feature_names = tfidf.get_feature_names_out()

# 카테고리별 워드 클라우드 생성
category_keywords = {}

# print(set(categories))

for category in set(categories):
  category_texts = [texts.iloc[i] for i in range(len(texts)) if categories.iloc[i] == category]

  if len(category_texts) > 0:
    # TF-IDF 계산
    category_tfidf_matrix = tfidf.transform(category_texts)

    # 키워드 추출
    keyword_scores = category_tfidf_matrix.mean(axis=0).A1  # 평균 TF-IDF 스코어
    sorted_indices = keyword_scores.argsort()[::-1] # 높은 값부터 정렬
    top_keywords = {feature_names[i]: keyword_scores[i] for i in sorted_indices[:30] if keyword_scores[i] > 0}

    # 워드 클라우드 생성
    if top_keywords:
      wordcloud = WordCloud(
        mask=mask,
        background_color="white",
        colormap='viridis',
        font_path=font_path,
        max_font_size=100,
        random_state=42
      ).generate_from_frequencies(top_keywords)

      # 워드 클라우드 시각화
      plt.figure(figsize=(5, 5))
      plt.imshow(wordcloud, interpolation="bilinear")
      plt.axis("off")
      plt.title(f"Word Cloud for {category} Category", fontproperties=fm.FontProperties(fname=font_path))
      plt.show()

      # 키워드 저장
      category_keywords[category] = top_keywords
