# K&C 산업뉴스 모니터링 명세서

해당 프로젝트에서는 Prototype은 news api를 통해서 news url을 수집한뒤에 newspaper3k package에서 뉴스 전문을 크롤링하는 방식을 채택
### 0. 디렉토리 구조
#### a. source
+ newsapi에서 크롤링한 뉴스 기사 (megazine, title, text, url)데이터 포함
+ 크롤링 일자별 폴더안에 Tech와 General, 기업별 sheet정리된 General news Data가 포함

#### b. backup
+ news-api는 사용량의 한정이 있고 인터넷 상황에 따라 끊길 수 있으므로 매체별 backup자료를 생성

#### c. article
+ news-api url을 이용하여 newspaper3k로 뉴스 전문 크롤링 테스트한 데이터 포함

#### d. crawler
+ craler code가 있음

#### e. classifier
 - <p style="font-size:18px"><strong>중요뉴스를 산출하는 코드와 자료가 있음</strong></p>

##### 1) data: merge 데이터와 score 데이터가 있음
##### 2) models: word2vec model 저장
##### 3) packages: stopwords와 전처리 코드 포함
##### 4) result: LDA결과와, 사전데이터, 사전별 점수 사출 데이터 포함


### 1. news url crawler 사용법
+ 필수 패키지 설치 사용
```{python}
pip install -r requirement.txt
```
+ url crawler 사용
```{python}
python crawler.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode tech
python crawler.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode general
python crawler.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode google
```
+ 예시
```{python}
python crawler.py --start_date 2019-06-01 --end_date 2019-06-26 --mode tech
python crawler.py --start_date 2019-06-01 --end_date 2019-06-26 --mode general
python crawler.py --start_date 2019-06-01 --end_date 2019-06-26 --mode google
```
+ start_date & end_date 뒤에는 시작날짜와 를 yyyy-mm-dd
+ 시작날짜와 종료날짜를 설정하지 않을 경우 오늘 날짜의 기사만 가져옴
+ mode 뒤에는 tech(IT 전문 매체) 또는 general(정론지) 또는 구글 뉴스를 설정하면 된다.

### 2. news_cralwer 사용법

+ google crawler는 안정적이지 않음
```{python}
python crawler.py --function article --mode tech
python crawler.py --function article --mode general
python crawler.py --function article --mode google
```

### 3. 데이터 생성기 사용법
+ 매일 매일 실행할때의 auto 기능 해당 날짜와 기간을 정핼 수 있는 manual 기능이 구분되어 있다.

+ manual로 진행할 때는 날짜를 입력해주어야 한다.
```{python}
python .\classifier\make_all_data_set.py --method auto
```
+ manual의 경우 기간의 걸쳐서 데이터를 merge할 수 있다.
```{python}
python .\classifier\make_all_data_set.py --method manual
```

### 3-1. 데이터 사전 구축기
+ 중요뉴스에 대한 업데이트가 있을때만 실행하면 된다.
+ 현재 중요 뉴스에 대한 업데이트는 수동으로 진행해야함

+ headline frequency 사전을 구축할 경우
+ ./classifier/data/ 안에 중요뉴스를 모아 놓은 knc_importance.xlsx 파일이 있어야 한다.
```{python}
python .\classifier\make_dictionary.py --method title
```

+ lda topic modeling keyword를 구축할 경우
+ ./classifier/data/category 안에 월 마다 Topic category가 분류된 knc_importance_{month(mm)}.xlsx가 있어야한다.
+ 현재 중요기사의 Topic을 자동으로 분류하는 기능은 없으나 keyword_score_LDA 행의 각 Category별 점수를 이용하여 가능할 것으로 보암
```{python}
python .\classifier\make_dictionary.py --method lda
```
+ 한 달의 한 번 중요기사를 모아서 ./classifier/data/category/knc_importance_{month(mm)}.xlsx 파일을 만들어주면 된다.
+ 이 경우 갱신하려는 월을 ex) 08과 같이 입력해주면 된다.

### 4. 뉴스 중요 점수 산출기

+ 매일 매일 실행할때의 auto 기능
+ classifier/data/Score 폴더 안에 headline_score.csv, total_keysord_score.csv 파일이 있어야 한다.(make_dictionary 에서 생성)
+ manual로 진행할 때는 날짜를 입력해주어야 한다.

+ headline으로만 점수를 산출할 경우
```{python}
python .\classifier\make_score.py --method title --date 2019-06-28
python .\classifier\make_score.py --method title --date 2019-07-16
python .\classifier\make_score.py --method title --date Today
```

+ lda를 통해서 전문을 포함한 점수를 산출할 경우
```{python}
python .\classifier\make_score.py --method lda --date 2019-06-28
python .\classifier\make_score.py --method lda --date 2019-07-16
python .\classifier\make_score.py --method lda --date Today
```

### 5. 뉴스 classifier 사용


+ 매일매일 학습을 진행할때
```{python}
python .\classifier\news_classifier.py --method auto --train True
```

+ 날짜를 지정해 새로 학습을 시킬 때
+ manual로 진행할 때는 날짜와 모델이름을 입력해주어야 한다.
+ ./classifier/models에 word2vec models파일 3개가 있어야한다.
```{python}
python .\classifier\news_classifier.py --method manual --train True --date 2019-09-25 --model W2V_model_20190925
```


### 6. batch 파일

+ url 크롤러 사용법
```{bash}
call .\batch\01.url_crawler.bat (시작날짜(yyyy-mm-dd)) (시작날짜(yyyy-mm-dd))
```
```{bash}
call .\batch\01.url_crawler.bat 2019-07-22 2019-07-24
```

+ news 전문 크롤러 사용법
```{bash}
call .\batch\02.article_crawler.bat
```

+ 데이터셋 만들기
```{bash}
call .\batch\03.merge.bat [method: auto or manual]
```
```{bash}
call .\batch\03.merge.bat auto
```
+ 사전은 그때그때 중요기사를 업데이트해야 시행하는 기능이므로 따로 windows batch파일을 만들지는 않았다.

+ 점수 산출기
```{bash}
call .\batch\04.score_test.bat [method: title or lda] [date: Today or YYYY-mm-dd]
```
```{bash}
call .\batch\04.score_test.bat title 2019-09-25
```

+ 중요뉴스 추출기
```{bash}
call .\batch\04.score_test.bat [method: manual or auto] [date: Today or YYYY-mm-dd] [model : model_name]
```
```{bash}
call .\batch\04.score_test.bat title True 2019-09-25 W2V_news_20190819
```
