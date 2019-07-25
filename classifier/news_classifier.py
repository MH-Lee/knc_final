####################################################################################################
### Project : kim & chang news alert service
### Content : K&C remove duplicate news
### Author  : Leon Choi & Hoon Lee
####################################################################################################
####################################################################################################
### make preprocess class
####################################################################################################
import os
import argparse
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.summarization import keywords
from gensim.models import Word2Vec
from gensim import models
from datetime import datetime, timedelta
import time
import multiprocessing
# nltk.download('wordnet')
cores = multiprocessing.cpu_count()
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
            self.today2 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y-%m-%d")
        if model == 'Base':
            yesterday = datetime.today() - timedelta(days=1)
            self.model_name = 'W2V_news_{}'.format(yesterday.strftime("%Y-%m-%d"))
        else:
            self.model_name = model
        print("Today : {}, word2vec:{} ,train: {}".format(self.today1, self.model_name, self.train),"News_classifier Start!")

    def preprocess(self, corpus):
        stop_words = set(stopwords.words('english'))
        n = WordNetLemmatizer()
        corpus_preprocess= []
        for i in range(0,len(corpus)):
            #print(i)
            # Special Characters
            text = re.sub(r"\'s", " 's ", str(corpus[i]))
            text = re.sub(r"\'ve", " 've ", text)
            text = re.sub(r"n\'t", " 't ", text )
            text = re.sub(r"\'re", " 're ",text)
            text = re.sub(r"\'d", " 'd ", text )
            text = re.sub(r"\'ll", " 'll ", text)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            text = re.sub(r"!", " ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"\)", " ) ", text)
            text= re.sub(r"\?", " ", text)
            text = re.sub(r"\s{2,}", " ", text) # change 2 white spaces as 1 white space
            text = re.sub("[^a-zA-Z]"," ", text) # change "match all strings that contain a non-letter" as 1 white spaced
            word_tokens = word_tokenize(text.lower())
            result = []
            for w in word_tokens:
                if w not in stop_words:
                    result.append(w)
            result = [n.lemmatize(w) for w in result]
            # dd = " ".join(result)
            corpus_preprocess.append(result)
        return(corpus_preprocess)

    def corpus_vector(self, corpus, model):
        corpus_vector = []
        for doc in corpus:
            # take average with each word vectors within document
            vector_1 = np.mean([model[word] for word in doc], axis = 0)
            corpus_vector.append(vector_1)
        corpus_vector = np.asarray(corpus_vector)
        return(corpus_vector)

    def title_vector(self, title, model):
        pre_title = self.preprocess(title)
        title_vec = self.corpus_vector(pre_title, model)
        return (title_vec)

    def text_vector(self, text, model):
        pre_text = self.preprocess(text)
        text_vec = self.corpus_vector(pre_text, model)
        return (text_vec)

    def title_corpus_vector(self, df, model, alpha = 0.3):
        title =  df.reset_index(drop = True)['Title']
        text = df.reset_index(drop = True)['Text']
        title_vec = self.title_vector(title, model)
        text_vec = self.text_vector(text, model)
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

    def make_result(self):
        data = pd.read_excel("./classifier/data/{}/article_score.xlsx".format(self.today1))
        date = data.Date.astype('category').cat.categories.tolist()
        print(len(date))
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)
        data['Contents'] = data['Title'] + "\n" + data["Text"]
        if self.train == True:
            print("setting train")
            start_time1 = time.time()
            newwiki  = self.preprocess(data.Contents)
            end_time1 = time.time()
            print(end_time1 - start_time1)
            print("start train!")
            start_time2 = time.time()
            model = gensim.models.Word2Vec.load('./classifier/models/{}'.format(self.model_name))
            model.build_vocab(newwiki, update=True)
            total_examples = model.corpus_count
            model.train(newwiki, total_examples = total_examples, epochs = 10)
            model.save('./classifier/models/W2V_news_{}'.format(self.today1))
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
        if os.path.exists('./classifier/results/{}'.format(date[-1])) == False:
            os.mkdir('./classifier/results/{}'.format(date[-1]))
        data = data[data.important_score >=2]
        for day in date:
            data_day = data[data.Date == day]
            data_day.sort_values('important_score', ascending=False , inplace=True)
            data_day.reset_index(inplace=True, drop=True)
            print(data_day.shape)
            init_row = data_day.shape[0]
            news_cv = self.title_corpus_vector(data_day, model= w2v_model, alpha = 0.2)
            cosine_day = self.cosine_mat(news_cv)
            # cosin similarity 0.8이상의 index
            # data.sort_values('Date', inplace=True)
            for i in data_day.index:
                try:
                    data_day.drop(np.where(cosine_day[i] > 0.75)[1][1:], axis=0, inplace=True)
                    # data03.reset_index(inplace=True, drop=True)
                except KeyError:
                    print(i)
                    print("pass")
                    continue
            last_row = data_day.shape[0]
            print("제거된 중복뉴스 수: {}".format(init_row - last_row))
            data_day.to_excel('./classifier/results/{}/{}.xlsx'.format(date[-1], day), index=False)


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