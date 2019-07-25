import pandas as pd
import os, datetime
import datetime

today = datetime.datetime.today().strftime('%Y-%m-%d')
today2 = datetime.datetime.today().strftime('%Y%m%d')
csv_file = os.listdir('./backup/{}'.format(today))


merge_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title", "Url", "Company"])
for f in csv_file:
    print(f)
    data = pd.read_csv('./backup/{}/{}'.format(today, f))
    print(data.shape)
    merge_df = merge_df.append(data, sort=True)
    print(merge_df.shape)

merge_df.to_csv("./source/{}/{}_genurl.csv".format(today, today2), index=False, encoding='utf-8')

str1 = '2019-06-28'
convert_date = datetime.datetime.strptime(str1, "%Y-%m-%d").date()
convert_date.strftime('%Y-%m-%d')

data2 = pd.read_excel("./article/{}/{}_general_news_article.xlsx".format(today, today2))
data2.drop_duplicates(['Title'], inplace=True)
data2.drop_duplicates(['Text'], inplace=True)
data2.to_excel("./article/{}/{}_general_news_article2.xlsx".format(today, today2), index=False)
