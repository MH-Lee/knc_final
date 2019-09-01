import pandas as pd
from packages.PreProcess import PreProcessing
from nltk.probability import FreqDist
# from gensim.summarization import keywords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import time, os
from datetime import datetime

pp = PreProcessing()

class MakeScoreData:
    def __init__(self, date='Today'):
        start_init_ = time.time()
        if date == 'Today':
            self.today1 = datetime.today().strftime("%Y-%m-%d")
            self.today2 = datetime.today().strftime("%Y%m%d")
        else:
            self.today1 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y-%m-%d")
            self.today2 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y%m%d")
        self.article_df = pd.read_excel('./classifier/data/{}/all_article_{}.xlsx'.format(self.today1, self.today2))
        self.knc_score = pd.read_csv('./classifier/results/Score/knc_score.csv', engine='python', index_col=0)
        self.headline_score = pd.read_csv('./classifier/results/Score/headline_score.csv', engine='python', index_col=0)
        self.total_topic_score = pd.read_csv('./classifier/results/Score/total_keywords_score.csv', engine='python', index_col=0)

        self.article_df['Contents'] = self.article_df['Title'] + '\n' + self.article_df['Text']
        self.article_df.dropna(inplace=True)
        end_init_ = time.time()
        print("PreProcess is done. time:{}, data dim: {}".format((end_init_ - start_init_), self.article_df.shape))

    def extract_words(self):
        start_extract_ = time.time()
        self.article_df['Title_keyword'] = pp.total_preprocess(self.article_df['Title'])
        self.article_df['Full_keyword'] = pp.total_preprocess(self.article_df['Contents'])
        self.article_df['Title_keyword_Freq'] = self.article_df['Title_keyword'].apply(lambda x:dict(FreqDist(x)))
        self.article_df['Full_keyword_Freq'] = self.article_df['Full_keyword'].apply(lambda x:dict(FreqDist(x)))
        self.article_df['Title_keyword_unique'] = self.article_df['Title_keyword_Freq'].apply(lambda x:list(x.keys()))
        self.article_df['Full_keyword_unique'] = self.article_df['Full_keyword_Freq'].apply(lambda x: list(x.keys()))
        end_extract_ = time.time()
        print(end_extract_ - start_extract_)
        return self.article_df

    def LDA_keywords_extract(self, article_df):
        start_LDA_ = time.time()
        tmp = []
        for i in range(len(article_df)):
            text = article_df.iloc[i,7]

            vectorizer = CountVectorizer(analyzer='word',
                                         stop_words='english',             # remove stop words                 # convert all words to lowercase
                                         token_pattern='[a-zA-Z0-9]{1,}')  # num chars > 1

            data_vectorized = vectorizer.fit_transform(text)
            feature_names = vectorizer.get_feature_names()

            start = time.time()
            lda_model = LatentDirichletAllocation(n_components=1,
                                                  learning_decay=0.1,
                                                  doc_topic_prior = 0.05,
                                                  learning_method='batch',
                                                  random_state=777,          # Random state
                                                  evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                                  n_jobs = -1)

            lda_model.fit(data_vectorized)
            end = time.time()
            if i % 100 == 0:
                print("{} %".format(round(i/len(article_df),3) * 100))
                print(end-start)

            for topic in lda_model.components_:
                top_word = [feature_names[i] for i in topic.argsort()[:10]]
            tmp.append(top_word)
        article_df['LDA_keywords'] = tmp
        end_LDA_ = time.time()
        print(end_LDA_ - start_LDA_)
        return article_df

    def make_score(self, article_df):
        article_df['common_knc'] = article_df['Title_keyword_unique'].apply(lambda x: list(set(x).intersection(self.knc_score.index.tolist())))
        article_df['common_headline'] = article_df['Title_keyword_unique'].apply(lambda x: list(set(x).intersection(self.headline_score.index.tolist())))
        article_df['common_full_LDA'] = article_df['LDA_keywords'].apply(lambda x: list(set(x).intersection(self.total_topic_score.index.tolist())))

        article_df['knc_score'] = article_df['common_knc'].apply(lambda x:len(x))
        article_df['headline_score'] = article_df['common_headline'].apply(lambda x: dict(self.headline_score.loc[x].sum().round(2)))
        article_df['headline_score_total'] = article_df['headline_score'].apply(lambda x: round(sum(x.values()),2))
        article_df['keyword_score_LDA'] = article_df['common_full_LDA'].apply(lambda x: dict(self.total_topic_score.loc[x].sum().round(2)))
        article_df['keyword_score_total_LDA'] = article_df['keyword_score_LDA'].apply(lambda x: round(sum(x.values()),2))

        article_df['Total_score'] = article_df[['knc_score', 'headline_score_total', 'keyword_score_total_LDA']].sum(axis=1)
        article_df.sort_values('Total_score', ascending=False, inplace=True)
        article_df.drop_duplicates(['Url'], inplace=True)
        article_df.drop_duplicates(['Title'], inplace=True)
        article_df.to_excel("./classifier/data/{}/article_score.xlsx".format(self.today1), index=False)

if __name__ == '__main__':
    date = input("날짜를 선택하세요 (Today or Date(YYYY-mm-dd))")
    ms = MakeScoreData(date=date)
    article_df = ms.extract_words()
    article_df_lda = ms.LDA_keywords_extract(article_df)
    ms.make_score(article_df_lda)
