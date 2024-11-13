import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request #인터넷을 이용해 데이터 요청
from konlpy.tag import Okt #한국어 텍스트 처리하기 위한 파이썬 도구
from tqdm import tqdm #진행율 바 표기

#seperate title [1.리뷰파일 다운로드] ==================================
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

#seperate title [2.판다스로 데이터확인] ==================================
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
print(train_data.info()) 
print(test_data.info())
# >>>
# >>> <class 'pandas.core.frame.DataFrame'>
# >>> #   Column    Non-Null Count   Dtype 
# >>> ---  ------    --------------   ----- 
# >>>  0   id        150000 non-null  int64 
# >>>  1   document  149995 non-null  object
# >>>  2   label     150000 non-null  int64 
# >>> dtypes: int64(2), object(1)
# >>> <class 'pandas.core.frame.DataFrame'>
# >>>  #   Column    Non-Null Count  Dtype 
# >>> ---  ------    --------------  ----- 
# >>>  0   id        50000 non-null  int64 
# >>> 1   document  49997 non-null  object
# >>>  2   label     50000 non-null  int64 
# >>> dtypes: int64(2), object(1)
#*개발자 분석내용- 데이터 확인 결과 document필드에서 결측치가 있음을 발견

#seperate title [3.결측데이터 확인 및 제거] ==================================
print("훈련데이터 결측치 갯수는:", train_data["document"].isna().sum())
print("훈련데이터 결측치 갯수는:", test_data["document"].isna().sum())
# >>> 훈련데이터 결측치 갯수는: 5
# >>> 훈련데이터 결측치 갯수는: 3
train_data = train_data.dropna(subset="document")
test_data = test_data.dropna(subset="document")
print("테스트데이터 결측치 갯수는:", train_data["document"].isna().sum())
print("테스트데이터 결측치 갯수는:", test_data["document"].isna().sum())
# >>> 테스트데이터 결측치 갯수는: 0
# >>> 테스트데이터 결측치 갯수는: 0
#*개발자 분석내용- 최초의 훈련데이터에 5개, 테스트데이터에 3개의 결측데이터가 관측되었고 pandas의 dropna로 제거

#seperate title [4.중복데이터 확인 및 제거] ==================================
#중복된 document 검사
print("훈련데이터 총 데이터 갯수:",train_data["document"].count())
print("훈련데이터 중복제외 갯수:",train_data["document"].nunique())
# >>> 훈련데이터 총 데이터 갯수: 149995
# >>> 훈련데이터 중복제외 갯수: 146182
#*개발자 분석내용- 총데이터 수량과 유니크데이터 수량의 차이가 있음으로 중복데이터가 존재하고 있다.
print("테스트데이터 총 데이터 갯수:",test_data["document"].count())
print("테스트데이터 중복제외 갯수:",test_data["document"].nunique())
# >>> 테스트데이터 총 데이터 갯수: 49997
# >>> 테스트데이터 중복제외 갯수: 49157
#*개발자 분석내용- 총데이터 수량과 유니크데이터 수량의 차이가 있음으로 중복데이터가 존재하고 있다. 테스트데이터는 훈련대상 데이터가 아니지만 중복 내용을 제거하기로 함

# ref참조 DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
train_data = train_data.drop_duplicates(subset="document")
test_data = test_data.drop_duplicates(subset="document")
print("훈련데이터 중복된 데이터 수 :", train_data["document"].count()-train_data["document"].nunique())
print("테스트데이터 중복된 데이터 수 :", test_data["document"].count()-test_data["document"].nunique())
# >>> 중복된 데이터 수 : 0
# >>> 중복된 데이터 수 : 0
#*개발자분석내용- 모든 데이터의 중복이 제거 되었음

#seperate title [5.한글과 공백을 제외한 문자제거와 형태소별로 분류] ==================================
print(train_data[:2])
# >>>   id                               document   label
0   9976970      아 더빙... 진짜 짜증나네요 목소리      0
1   3819312     흠...포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
#*개발자분석- ... ? 영문등은 감성분석에 불필요함으로 제거가 필요함
#            정규표현식을 이용한 한글과 공백을 제외한 모든 단어는 제거
train_data["document"]=train_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True)
print(train_data[:2])
# >>>        id                          document  label
# >>> 0   9976970    아 더빙 진짜 짜증나네요 목소리      0
# >>> 1   3819312    흠포스터보고 초딩영화줄오버연기조차 가볍지 않구나      1
test_data["document"]=test_data["document"].replace(r"[^\sㄱ-ㅎㅏ-ㅣ가-힣]","",regex=True)















