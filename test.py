import pandas as pd
from classifier.packages.PreProcess import PreProcessing

pp_dict = PreProcessing(mode='dictionary')
pp_key = PreProcessing(mode='keywords')
article_df = pd.read_excel('./classifier/data/2019-08-19/all_article_20190819.xlsx')

article_df['Title'] = [pp_key.replace(title)for title in article_df['Title'].tolist()]
article_df['Title_keyword'] = pp_key.total_preprocess(article_df['Title'], mode='title')
article_df2 = article_df[article_df['Date'] == '2019-08-05']
article_df2[['Date','Title', 'Title_keyword']].to_excel('test1.xlsx')
article_df['Full_keyword'] = pp_dict.total_preprocess(article_df['Contents'])
