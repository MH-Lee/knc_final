import time, os, argparse
import itertools
import pandas as pd
from packages.PreProcess import PreProcessing
from nltk.probability import FreqDist
# from gensim.summarization import keywords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from itertools import combinations
from operator import itemgetter
from collections import Counter, OrderedDict

pp_dict = PreProcessing(mode='dictionary')
pp_key = PreProcessing(mode='keywords')

def parse_args():
    # set make data parser
    parser = argparse.ArgumentParser(description='news extract')
    parser.add_argument('--method', help='select only title or lda', default='lda', type=str)
    parser.add_argument('--date', help='select date (Today or Date(YYYY-mm-dd))', default='Today', type=str)
    parser.add_argument('--rate', help='rate co-occurence score', default=0.5, type=float)
    args = parser.parse_args()
    return args

class MakeScoreData:
    def __init__(self, date='Today', rate=0.5):
        start_init_ = time.time()
        if date == 'Today':
            self.today1 = datetime.today().strftime("%Y-%m-%d")
            self.today2 = datetime.today().strftime("%Y%m%d")
        else:
            self.today1 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y-%m-%d")
            self.today2 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y%m%d")
        self.article_df = pd.read_excel('./classifier/data/{}/all_article_{}.xlsx'.format(self.today1, self.today2))
        self.headline_score = pd.read_csv('./classifier/results/Score/headline_score.csv', engine='python', index_col=0)
        self.total_topic_score = pd.read_csv('./classifier/results/Score/total_keywords_score.csv', engine='python', index_col=0)
        self.cor_list = pd.read_csv('./company_data/Company.csv')['Name'].tolist()
        self.cor_list = [cor.lower() for cor in self.cor_list]
        self.tech_list = pd.read_csv('./company_data/Tech.csv')['Tech'].tolist()
        self.etc_cor = pd.read_csv('./company_data/fortune_g.csv')['Company'].tolist()
        self.etc_cor = [cor.lower() for cor in self.etc_cor]
        self.article_df['Contents'] = self.article_df['Title'] + '\n' + self.article_df['Text']
        self.article_df.dropna(inplace=True)
        self.rate = rate
        end_init_ = time.time()
        print("PreProcess is done. time:{}, data dim: {}".format((end_init_ - start_init_), self.article_df.shape))

    def extract_words(self):
        start_extract_ = time.time()
        self.article_df['Title'] = [pp_key.replace(title)for title in self.article_df['Title'].tolist()]
        self.article_df['Title_keyword'] = pp_key.total_preprocess(self.article_df['Title'], mode='title')
        self.article_df['Full_keyword'] = pp_dict.total_preprocess(self.article_df['Contents'])
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

    def generate_co_occurence(self, corpus):
        text_data = corpus.tolist()
        varnames = tuple(sorted(set(itertools.chain(*text_data))))
        expanded = [tuple(itertools.combinations(d, 2)) for d in text_data]
        expanded = itertools.chain(*expanded)
        expanded = [tuple(sorted(d)) for d in expanded]
        oreder_c = OrderedDict(Counter(expanded))
        dict_c = dict([(key, oreder_c[key]) for key in oreder_c.keys() if oreder_c[key] > 2])
        return dict_c

    def make_score(self, article_df, method='title'):
        article_df['Company_list'] = article_df['Title_keyword_unique'].apply(lambda x:list(set(x).intersection(self.cor_list)))
        article_df['Company_list2'] = article_df['Title_keyword_unique'].apply(lambda x:list(set(x).intersection(self.etc_cor)))
        article_df['Tech_list'] = article_df['Title_keyword_unique'].apply(lambda x:list(set(x).intersection(self.tech_list)))
        article_df['common_headline'] = article_df['Title_keyword_unique'].apply(lambda x: list(set(x).intersection(self.headline_score.index.tolist())))
        article_df['headline_comb'] = article_df['common_headline'].apply(lambda x:list(combinations(x, 2)))

        co_dict = self.generate_co_occurence(article_df['common_headline'])
        article_df['Company_score'] = article_df['Company_list'].apply(lambda x:len(x))
        article_df['Company_score2'] = article_df['Company_list2'].apply(lambda x:len(x)*0.5)
        article_df['Company_score'] = article_df[['Company_score', 'Company_score2']].sum(axis=1)
        article_df.drop(['Company_score2'], axis=1, inplace=True)
        # coo_df_dict = {'Coo_Word':list(co_dict.keys()), 'freq':list(co_dict.values())}
        # coo_df =  pd.DataFrame.from_dict(coo_df_dict)
        article_df['Tech_score'] = article_df['Tech_list'].apply(lambda x:len(x))
        article_df['headline_score'] = article_df['common_headline'].apply(lambda x: self.headline_score.loc[x].values.sum())
        tuple_list = article_df['headline_comb'].tolist()

        for key in co_dict:
            co_dict[key] = self.rate * round((co_dict[key] - min(co_dict.values())) / (max(co_dict.values()) - min(co_dict.values())), 3)

        list_a = []
        for tuple_ in tuple_list:
            score_list = []
            for key in tuple_:
                try:
                    score_list.append(co_dict[key])
                except:
                    continue
            list_a.append(score_list)

        article_df['coo_score'] = list_a
        article_df['coo_score'] = article_df['coo_score'].apply(lambda x: sum(x))

        if method == 'lda':
            article_df['common_full_LDA'] = article_df['LDA_keywords'].apply(lambda x: list(set(x).intersection(self.total_topic_score.index.tolist())))
            article_df['keyword_score_LDA'] = article_df['common_full_LDA'].apply(lambda x: dict(self.total_topic_score.loc[x].sum().round(2)))
            article_df['keyword_score_total_LDA'] = article_df['keyword_score_LDA'].apply(lambda x: round(max(x.values()),2))
            article_df['Total_score'] = article_df[['Company_score', 'coo_score', 'headline_score', 'Tech_score', 'keyword_score_total_LDA']].sum(axis=1)
        else:
            article_df['Total_score'] = article_df[['Company_score', 'coo_score', 'headline_score', 'Tech_score']].sum(axis=1)
        article_df.sort_values('Total_score', ascending=False, inplace=True)
        article_df.drop_duplicates(['Url'], inplace=True)
        article_df.drop_duplicates(['Title'], inplace=True)
        # coo_df.to_excel("./classifier/data/{}/co_occurence.xlsx".format(self.today1), index=False)
        article_df.to_excel("./classifier/data/{}/article_score_{}.xlsx".format(self.today1, self.rate), index=False)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    date = args.date
    rate = args.rate
    ms = MakeScoreData(date=date, rate=rate)
    article_df = ms.extract_words()
    if args.method == 'lda':
        article_df_lda = ms.LDA_keywords_extract(article_df)
        ms.make_score(article_df_lda, method='lda')
    else:
        ms.make_score(article_df)
