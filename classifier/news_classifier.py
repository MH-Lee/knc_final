####################################################################################################
### Project : kim & chang news alert service
### Content : K&C remove duplicate news
### Author  : Leon Choi & Minsu Kim & Hoon Lee
####################################################################################################
####################################################################################################
### make preprocess class
####################################################################################################
import os
import argparse
import pandas as pd
import numpy as np
import re
import gensim
from gensim.models import Word2Vec
from gensim import models
from datetime import datetime, timedelta
import time
import multiprocessing
import ast
from packages.PreProcess import PreProcessing
cores = multiprocessing.cpu_count()
pp = PreProcessing()
####################################################################################################
### make preprocess function
####################################################################################################
def parse_args():
    # set make data parser
    parser = argparse.ArgumentParser(description='news extract')
    parser.add_argument('--method', help='select auto or manual', default='auto', type=str)
    parser.add_argument('--train', help='train or not', default=False, type=bool)
    args = parser.parse_args()
    return args

class News_classifier:

    def __init__(self, date='Today', train=False, model='Base'):
        self.train = train
        if date == 'Today':
            self.today1 = datetime.today().strftime("%Y-%m-%d")
            self.today2 = datetime.today().strftime("%Y%m%d")
        else:
            self.today1 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y-%m-%d")
            self.today2 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y%m%d")
        if model == 'Base':
            yesterday = datetime.today() - timedelta(days=1)
            self.model_name = 'W2V_news_{}'.format(yesterday.strftime("%Y%m%d"))
        else:
            self.model_name = model
        print("Today : {}, word2vec:{} ,train: {}".format(self.today1, self.model_name, self.train),"News_classifier Start!")

    def corpus_vector(self, corpus, model):
        corpus_vector = []
        for doc in corpus:
            # take average with each word vectors within document
            vector_1 = np.mean([model[word] for word in doc], axis = 0)
            corpus_vector.append(vector_1)
        corpus_vector = np.asarray(corpus_vector)
        return(corpus_vector)

    def title_vector(self, title, model):
        title = pp.total_preprocess(title)
        title_vec = self.corpus_vector(title, model)
        return (title_vec)

    def text_vector(self, text, model):
        text_vec = self.corpus_vector(text, model)
        return (text_vec)

    def title_corpus_vector(self, df, model, alpha = 0.3):
        title =  df.reset_index(drop = True)['Title']
        text = df.reset_index(drop = True)['Full_keyword'].tolist()
        title_vec = self.title_vector(title, model)
        text_vec = self.text_vector(text, model)
        print(title_vec.shape, text_vec.shape)
        new_corpus = (title_vec * alpha) + (text_vec * (1 - alpha))
        return(new_corpus)

    def cosine_corpus(self, vector_1, vector_2):
        import scipy
        cosine = np.dot(vector_1, vector_2) / (np.sqrt(np.dot(vector_1, vector_1)) * np.sqrt(np.dot(vector_2, vector_2)))
        if np.isnan(np.sum(cosine)):
            return 0
        return(cosine)

    def cosine_mat(self, corpus):
        dist_mat = []
        for doc1 in corpus:
            results = []
            for doc2 in corpus:
                sim = self.cosine_corpus(doc1, doc2)
                results.append(sim)
            dist_mat.append(results)
        dist_mat = np.asmatrix(dist_mat)
        return(dist_mat)

    def merge_list_by_col(self, row):
        # print(row['Title_keyword'])
        total_list = list(row['Title_keyword']) + list(row['Full_keyword'])
        return total_list

    def make_result(self):
        data = pd.read_excel("./classifier/data/{}/article_score.xlsx".format(self.today1))
        date = data.Date.astype('category').cat.categories.tolist()
        data.dropna(inplace=True)
        print("날짜: ", len(date), "데이터 차원: ", data.shape)
        data.reset_index(inplace=True, drop=True)
        data['Title_keyword'] = data['Title_keyword'].apply(lambda x: ast.literal_eval(x))
        data['Title_keyword_len'] = data['Title_keyword'].apply(lambda x: len(x))
        data['Full_keyword'] = data['Full_keyword'].apply(lambda x: ast.literal_eval(x))
        data['Full_keyword'] = data.apply(self.merge_list_by_col, axis=1)
        data = data[data['Title_keyword_len'] > 0]
        if self.train == True:
            print("setting train")
            start_time1 = time.time()
            newwiki  =  data.Full_keyword.tolist()
            end_time1 = time.time()
            print(end_time1 - start_time1)
            print("start train!")
            start_time2 = time.time()
            model = gensim.models.Word2Vec.load('./classifier/models/{}'.format(self.model_name))
            model.build_vocab(newwiki, update=True)
            total_examples = model.corpus_count
            model.train(newwiki, total_examples = total_examples, epochs = 10)
            model.save('./classifier/models/W2V_news_{}'.format(self.today2))
            end_time2 = time.time()
            print(end_time2 - start_time2)
            w2v_model = model.wv
            del model
        else:
            model = gensim.models.Word2Vec.load('./classifier/models/{}'.format(self.model_name))
            w2v_model = model.wv
            del model
        if os.path.exists('./classifier/results/') == False:
            os.mkdir('./classifier/results/')
        if os.path.exists('./classifier/results/article/') == False:
            os.mkdir('./classifier/results/article/')
        if os.path.exists('./classifier/results/article/{}'.format(date[-1])) == False:
            os.mkdir('./classifier/results/article/{}'.format(date[-1]))
        data = data[data.Total_score > 3.5]
        make_data_start = True
        for day in date:
            data_day = data[data.Date == day]
            data_day.sort_values('Total_score', ascending=False , inplace=True)
            data_day.reset_index(inplace=True, drop=True)
            print(data_day.shape)
            init_row = data_day.shape[0]
            news_cv = self.title_corpus_vector(data_day, model= w2v_model, alpha = 0.2)
            cosine_day = self.cosine_mat(news_cv)
            print(cosine_day)
            # cosin similarity 0.8이상의 index
            # data.sort_values('Date', inplace=True)
            for i in data_day.index:
                try:
                    data_day.drop(np.where(cosine_day[i] > 0.75)[1][1:], axis=0, inplace=True)
                    # data03.reset_index(inplace=True, drop=True)
                except KeyError:
                    print(i, "pass")
                    # print()
                    continue
            if make_data_start == True:
                total_df = data_day
                make_data_start = False
            else:
                total_df = total_df.append(data_day)
            last_row = data_day.shape[0]
            print("제거된 중복뉴스 수: {}".format(init_row - last_row))
            data_day.to_excel('./classifier/results/article/{}/{}.xlsx'.format(date[-1], day), index=False)
        total_df.to_excel('./classifier/results/article/{}/total_df.xlsx'.format(date[-1]), index=False)

if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.method == 'auto':
        if args.train == True:
            nc = News_classifier(train=True)
        else:
            nc = News_classifier()
    else:
        date = input("날짜(YYYY-mm-dd): ")
        model = input("word2vec model name: ")
        if args.train == True:
            nc = News_classifier(date=date, train=True, model=model)
        else:
            nc = News_classifier(date=date, model=model)
    nc.make_result()
