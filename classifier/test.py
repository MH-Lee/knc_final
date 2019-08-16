import pandas as pd
from classifier.packages.PreProcess import PreProcessing
from nltk.probability import FreqDist

pp = PreProcessing()

df = pd.read_excel('./article/2019-06-28/all_article_20190628.xlsx')
df['Total_text'] = df['Title'] + '\n' + df['Text']
df.dropna(inplace=True)

df['Title_keyword'] = pp.total_preprocess(df['Title'])
df['Full_keyword'] = pp.total_preprocess(df['Total_text'])

df['Title_keyword_Freq'] = df['Title_keyword'].apply(lambda x:dict(FreqDist(x)))
df['Full_keyword_Freq'] = df['Full_keyword'].apply(lambda x:dict(FreqDist(x)))
df['Title_keyword_unique'] = df['Title_keyword_Freq'].apply(lambda x:list(x.keys()))
df['Full_keyword_unique'] = df['Full_keyword_Freq'].apply(lambda x: list(x.keys()))
df.to_excel("./classifier/data/2019-06-28/article_20190628.xlsx",index=False)

knc_score = pd.read_csv('./classifier/data/word_score/knc_score.csv', engine='python')
headline_score = pd.read_csv('./classifier/data/word_score/headline_score.csv', engine='python')
total_topic_score = pd.read_csv('./classifier/data/word_score/total_keywords_score.csv', engine='python')

knc_score.set_index('Words', inplace=True)
headline_score.set_index('Word', inplace=True)
total_topic_score.set_index('Word', inplace=True)

df['common_knc'] = df['Full_keyword_unique'].apply(lambda x: list(set(x).intersection(knc_score.index.tolist())))
df['common_headline'] = df['Title_keyword_unique'].apply(lambda x: list(set(x).intersection(headline_score.index.tolist())))
df['common_headline2'] = df['Title_keyword_unique'].apply(lambda x: list(set(x).intersection(total_topic_score.index.tolist())))
df['common_full'] = df['Full_keyword_unique'].apply(lambda x: list(set(x).intersection(total_topic_score.index.tolist())))

def full_index_dict(row):
    return dict([(key, value) for key,value in row['Full_keyword_Freq'].items() if key in row['common_full']])
def title_index_dict(row):
    return dict([(key, value) for key,value in row['Title_keyword_Freq'].items() if key in row['common_headline']])

df['Full_keyword_Freq']  = df.apply(full_index_dict, axis=1)
df['Title_keyword_Freq']  = df.apply(title_index_dict, axis=1)
df['knc_score'] = df['common_knc'].apply(lambda x:len(x))
df['headline_score'] = df['common_headline'].apply(lambda x: dict(headline_score.loc[x].sum().round(2)))
df['headline_score_total'] = df['headline_score'].apply(lambda x: round(sum(x.values()),2))
df['keyword_score'] = df['common_full'].apply(lambda x: dict(total_topic_score.loc[x].sum().round(2)))
df['keyword_score_total'] = df['keyword_score'].apply(lambda x: round(sum(x.values()),2))

df.columns
df['Total_score'] = df[['knc_score', 'headline_score_total']].sum(axis=1)
df.sort_values('Total_score', ascending=False, inplace=True)
df.drop_duplicates(['Url'], inplace=True)
df.drop_duplicates(['Title'], inplace=True)
df.shape
df

# df.iloc[0:400, 0:5].to_excel("only_score.xlsx", index=False)
df.to_excel("total_scoring.xlsx", index=False)
# df.to_excel("title_scoring.xlsx", index=False)
test_Df = pd.read_excel("./classifier/data/Knc_importance.xlsx")
pred_df = df.iloc[0:len(test_Df), 0:5]
len(set(test_Df['Url'].tolist()).intersection(set(pred_df['Url'].tolist())))/len(test_Df)


data2 = pd.read_excel('./article/Article.xlsx')
data2['Category1'] = data2['Category1'].apply(lambda x: x.strip())
data2['Category2'] = data2['Category1'].apply(lambda x: x.strip())
(data2.groupby('Category1').count()/len(data2))[['Magazine']]
