# news api & news crawler 사용법

해당 프로젝트에서는 news api를 통해서 news url을 수집한뒤에 newspaper3k package에서 뉴스 전문을 크롤링하는 방식을 채택하고 있다.

### 1. news url crawler 사용법
+ 필수 패키지 설치 사용 
```{python}
pip install -r requirement.txt
```
+ url crawler 사용
```{python}
python crawler_test.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode tech
python crawler_test.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode general
python crawler_test.py --start_date (시작날자 (yyyy-mm-dd)) --end_date (종료날자 (yyyy-mm-dd)) --mode google
```
+ 예시
```{python}
python crawler_test.py --start_date 2019-06-01 --end_date 2019-06-26 --mode tech
python crawler_test.py --start_date 2019-06-01 --end_date 2019-06-26 --mode general
python crawler_test.py --start_date 2019-06-01 --end_date 2019-06-26 --mode google
```
+ start_date & end_date 뒤에는 시작날짜와 를 yyyy-mm-dd
+ 시작날짜와 종료날짜를 설정하지 않을 경우 오늘 날짜의 기사만 가져옴
+ mode 뒤에는 tech(IT 전문 매체) 또는 general(정론지) 또는 구글 뉴스를 설정하면 된다.

### 2. news_cralwer 사용법

+ google crawler는 안정적이지 않음
```{python}
python crawler_test.py --function article --mode tech
python crawler_test.py --function article --mode general
python crawler_test.py --function article --mode google
```

### 3. 데이터 생성기 사용법
+ 매일 매일 실행할때의 auto 기능과
+ 해당 날짜와 기간을 정핼 수 있는 manual 기능이 구분되어 있다.
+ manual로 진행할 때는 날짜를 입력해주어야 한다.
```{python}
python .\classifier\make_all_data_set.py --method auto
```
+ manual의 경우 기간의 걸쳐서 데이터를 merge할 수 있다.
```{python}
python .\classifier\make_all_data_set.py --method manual
```

### 4. 뉴스 중요 점수 산출기
+ 매일 매일 실행할때의 auto 기능
+ classifier/data/ 폴더 안에 importance.xlsx 파일이 있어야 한다.
+ manual로 진행할 때는 날짜를 입력해주어야 한다.
```{python}
python .\classifier\make_all_data_set.py --method auto
```
+ 해당 날짜를 정핼 수 있는 manual 기능
```{python}
python .\classifier\make_all_data_set.py --method manual
```
### 5. 뉴스 classifier 사용
+ 매일매일 학습을 진행할때
```{python}
python .\classifier\news_classifier.py --method auto --train True
```

+ 날짜를 지정해 새로 학습을 시킬 때
+ manual로 진행할 때는 날짜와 모델이름을 입력해주어야 한다.
```{python}
python .\classifier\news_classifier.py --method manual --train True
```
+ 날짜를 지정해 기존의 word2vec model로 진핼할 때(새로운 단어 벡터가 있을 수 있어 대부분 학습을 시켜야함)
```{python}
python .\classifier\news_classifier.py --method manual
```

### 3. batch 파일

+ url 크롤러 사용법
```{bash}
call .\batch\01.url_crawler.bat (시작날짜(yyyy-mm-dd)) (시작날짜(yyyy-mm-dd))
```

+ 예시
```{bash}
call .\batch\01.url_crawler.bat 2019-07-22 2019-07-24
```

+ news 크롤러 사용법
```{bash}
call .\batch\02.article_crawler.bat
```

+ 데이터셋 만들기
```{bash}
call .\batch\03.data_generate.bat
```
