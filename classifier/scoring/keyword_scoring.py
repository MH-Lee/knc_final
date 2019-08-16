import pandas as pd
from sklearn.preprocessing import MinMaxScaler

knc_word = pd.read_excel("../Data/knc_word.xlsx")
headline = pd.read_csv("../Result/keyword_headline.csv", engine="python")
total_keywords = pd.read_csv("../Result/total_topics.csv", engine="python")

knc_word['Scores'] = 2
knc_word = knc_word[['Words', 'Scores']]
knc_word.to_csv("../Score/knc_score.csv", encoding='cp949', index=False)
headline =headline[headline['Frequency'] > 1]
headline['Frequency'] = headline.groupby('Category')['Frequency'].apply(lambda x: round(((x-min(x))/(max(x)-min(x)) + 1), 2))
headline_score = headline.pivot(index='Word', columns='Category', values='Frequency').fillna(0).reset_index()
headline_score.columns = ['Word', 'International', 'Regulation', 'Corporate', 'Law', 'Market', 'Tech']
headline_score.to_csv("../Score/headline_score.csv", encoding='cp949', index=False)

total_keywords.head()
total_keywords = total_keywords.groupby(['Words', 'Category'], as_index=False).count()
total_keywords.sort_values(['Category','Topic'], ascending=[True, False], inplace=False)
total_keywords.rename(columns ={'Topic':'Score'}, inplace=True)
total_keywords['Score'] = total_keywords['Score'] / 2
total_keywords_score = total_keywords.pivot(index='Words', columns='Category', values='Score').reset_index().fillna(0)
total_keywords_score.columns = ['Word', 'International', 'Regulation', 'Corporate', 'Law', 'Market', 'Tech']
total_keywords_score.to_csv("../Score/total_keywords_score.csv", encoding='cp949', index=False)
