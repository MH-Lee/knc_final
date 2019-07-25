####################################################################################################
### Project : kim & chang news alert service
### Content : K&C remove duplicate news
### Script  : 03_Analysis_code.py
### Author  : Hoon Lee & Kim
####################################################################################################
####################################################################################################
### Set up the environment
####################################################################################################
from newspaper import Article
import pandas as pd
import numpy as np
import datetime, time, os
import nltk
nltk.download('punkt')

def article_crawler(media_type='tech', date='today'):
    start_time = time.time()
    if date == 'today':
        today = datetime.date.today().strftime("%Y-%m-%d")
        today2 = datetime.date.today().strftime("%Y%m%d")
    else:
        convert_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        today = convert_date.strftime('%Y-%m-%d')
        today2 = convert_date.strftime("%Y%m%d")
    if os.path.exists('./article/') == False:
        os.mkdir('./article')
    if os.path.exists('./article/{}'.format(today)) == False:
        os.mkdir('./article/{}'.format(today))
    if media_type == 'tech':
        url_df = pd.read_csv('./source/{}/{}_techurl.csv'.format(today, today2), engine="python")
        original_row = url_df.shape[0]
    elif media_type == 'google':
        url_df = pd.read_csv('./source/{}/{}_googleurl.csv'.format(today,today2), engine="python")
        original_row = url_df.shape[0]
    elif media_type == 'general':
        url_df = pd.read_csv('./source/{}/{}_genurl.csv'.format(today,today2), engine="python")
        original_row = url_df.shape[0]
    else:
        print("잘 못된 입력 tech, google, general 중에서 입력해주세요")
    url_df.drop(np.where(url_df['Url'].isna())[0], inplace=True)
    url_df.drop_duplicates(['Url'],inplace=True)
    print("다음과 같은 중복데이터 제거: {}".format(int(original_row) - int(url_df.shape[0])))
    print(url_df.shape)
    if media_type == 'tech' or media_type == 'google':
        total_news_df = pd.DataFrame(columns=["Magazine", "Date", "Title", "Text", "Url", "Keyword"])
        for i in range(url_df.shape[0]):
            url = url_df.iloc[i]['Url']
            date = url_df.iloc[i]['Date'][0:10]
            if media_type == "google":
                magazine = "google_news"
            else:
                magazine = url_df.iloc[i]['Magazine']
            if i % 10 == 0:
                print("{}번째 & 미디어이름:{}".format(i, magazine))
            if url[-3:] in ['mp3', 'mp4']:
                print(url)
                continue
            if 'videos/' in url:
                print(url)
                continue
            if 'shopping/' in url:
                print(url)
                continue
            if 'best-deals-' in url:
                print(url)
                continue
            a = Article(url, language='en')
            a.download()
            try:
                a.parse()
                title = a.title
                text = a.text
                a.nlp()
                keyword = a.keywords
            except:
                print("{} pass".format(i))
            total_news_df = total_news_df.append({"Magazine"    : magazine,
                                                    "Date"      : date,
                                                    "Title"     : title,
                                                    "Text"      : text,
                                                    "Url"       : url,
                                                    "Keyword"   : keyword}, ignore_index=True)
        formal_row = total_news_df.shape[0]
        total_news_df.drop_duplicates(['Title'], inplace=True)
        total_news_df.drop_duplicates(['Text'], inplace=True)
        final_row = total_news_df.shape[0]
        print("최종적인 행수 : ",final_row - formal_row)
        total_news_df.to_excel('./article/{}/{}_{}_news_article.xlsx'.format(today, today2, media_type), index=False)
    # 정론지 크롤러
    elif  media_type == 'general':
        total_news_df = pd.DataFrame(columns=["Magazine", "Date", "Title", "Text", "Url"])
        url_df['Company'] = url_df['Company'].astype('category')
        excel_writer = pd.ExcelWriter('./article/{}/{}_{}_news_article_sheet.xlsx'.format(today, today2, media_type))
        for cor in url_df['Company'].cat.categories:
            company_df = pd.DataFrame(columns=["Magazine", "Date", "Title", "Text", "Url"])
            # Separate by company name
            print("현재 회사 : {}".format(cor))
            sub_url = url_df[url_df['Company'] == cor]
            print("회사별 기사 개수: ", sub_url.shape)
            for i in range(sub_url.shape[0]):
                url = sub_url.iloc[i]['Url']
                date = sub_url.iloc[i]['Date'][0:10]
                magazine = sub_url.iloc[i]['Magazine']
                if i % 10 == 0:
                    print("{}번째 & 미디어이름:{}".format(i, magazine))
                if url[-3:] in ['mp3', 'mp4']:
                    print(url)
                    continue
                if 'videos/' in url:
                    print(url)
                    continue
                if 'shopping/' in url:
                    print(url)
                    continue
                if 'best-deals-' in url:
                    print(url)
                    continue
                a = Article(url, language='en')
                a.download()
                try:
                    a.parse()
                    title = a.title
                    text = a.text
                    a.nlp()
                    keyword = a.keywords
                except:
                    print("{} pass".format(i))
                total_news_df = total_news_df.append({"Magazine"  : magazine,
                                                      "Date"      : date,
                                                      "Title"     : title,
                                                      "Text"      : text,
                                                      "Url"       : url,
                                                      "Keyword"   : keyword}, ignore_index=True)
                company_df = company_df.append({"Magazine"  : magazine,
                                                "Date"      : date,
                                                "Title"     : title,
                                                "Text"      : text,
                                                "Url"       : url,
                                                "Keyword"   : keyword}, ignore_index=True)
                company_df.drop_duplicates(['Title'], inplace=True)
                company_df.drop_duplicates(['Text'], inplace=True)
            company_df.to_excel(excel_writer,sheet_name=cor, index=False)
        excel_writer.save()
        formal_row = total_news_df.shape[0]
        total_news_df.drop_duplicates(['Title'], inplace=True)
        total_news_df.drop_duplicates(['Text'], inplace=True)
        final_row = total_news_df.shape[0]
        print("최종적인 행수 : ",final_row - formal_row)
        total_news_df.to_excel('./article/{}/{}_{}_news_article.xlsx'.format(today, today2, media_type), index=False)
    end_time = time.time()
    print(end_time - start_time)
    return "complete {}".format(end_time - start_time)
