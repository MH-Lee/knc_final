from packages.PreProcess import PreProcessing
import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import pyLDAvis.sklearn as sklvis
import time, os
import argparse

pp = PreProcessing(mode='dictionary')
np.random.seed(777)

def parse_args():
    # set make data parser
    parser = argparse.ArgumentParser(description='news extract')
    parser.add_argument('--method', help='select only title or lda', default='lda', type=str)
    args = parser.parse_args()
    return args

class MakeKewordDict:
    def __init__(self, month=None, mode="title"):
        # make resutls directory
        if mode == 'title':
            self.data = pd.read_excel("./classifier/data/important_article/knc_importance.xlsx")
        else:
            self.data = pd.read_excel("./classifier/data/important_article/category/knc_importance_{}.xlsx".format(month))
        if os.path.exists('./classifier/results/') == False:
            os.mkdir('./classifier/results/')
        if os.path.exists('./classifier/results/Score') == False:
            os.mkdir('./classifier/results/Score')
        if os.path.exists('./classifier/results/Dictionary') == False:
            os.mkdir('./classifier/results/Dictionary')
        if os.path.exists('./classifier/results/LDA_html') == False:
            os.mkdir('./classifier/results/LDA_html')
        if os.path.exists('./classifier/results/Dictionary/Topic_csv') == False:
            os.mkdir('./classifier/results/Dictionary/Topic_csv')
        if os.path.exists('./classifier/results/Dictionary/likelihood') == False:
            os.mkdir('./classifier/results/Dictionary/likelihood')
        if mode=='lda':
            self.month=month
            if os.path.exists('./classifier/results/Dictionary/Topic_csv/{}'.format(month)) == False:
                os.mkdir('./classifier/results/Dictionary/Topic_csv/{}'.format(month))
            if os.path.exists('./classifier/results/Dictionary/likelihood/{}'.format(month)) == False:
                os.mkdir('./classifier/results/Dictionary/likelihood/{}'.format(month))
            if os.path.exists('./classifier/results/LDA_html/{}'.format(month)) == False:
                os.mkdir('./classifier/results/LDA_html/{}'.format(month))
        print("start!")

    # pre-process
    def title_keyword_preprocess_(self):
        title_data = self.data
        title = title_data["Title"].tolist()
        # Step 01. Change all characters into lower case
        corpus = pp.total_preprocess(title)
        title_data["Title2"] = [" ".join(c) for c in corpus]
        # Re pre processing
        tmp = title_data["Title2"].tolist()
        tmp = pp.tokenize(tmp)
        tmp = pp.remv_total_stopwords(tmp)
        tmp = pp.pos_tag(tmp, mode="title")
        tmp = pp.remv_total_stopwords(tmp)
        tmp = pp.remv_short_words(tmp)
        title_data["Title2"] = tmp
        return title_data

    def title_keword_extract(self):
        title_data = self.title_keyword_preprocess_()
        title_keyword = pd.DataFrame(columns=["Word", "Frequency"])
        tmp = title_data["Title2"].tolist()
        corpus = []
        for text in tmp:
            tmp2 = " ".join(text)
            corpus.append(tmp2)

        # make tf matrix
        corpus_tf = CountVectorizer().fit(corpus)
        count = corpus_tf.transform(corpus).toarray().sum(axis=0)
        # sort index by word frequency
        idx = np.argsort(-count)

        # tf matrix to list
        count = count[idx]
        feature_name = np.array(corpus_tf.get_feature_names())[idx]
        corpus_tf = list(zip(feature_name, count))

        # make frequency dataframe
        for i in range(0, len(corpus_tf)):
            title_keyword = title_keyword.append({'Word':corpus_tf[i][0],\
                                                'Frequency':corpus_tf[i][1]}, ignore_index=True)
        title_keyword.to_csv("./classifier/results/Dictionary/keyword_headline.csv", index=False, encoding="cp949")

    ########################################################################################################
    ### LDA로 뽑은 토픽에 Likelihood를 그려주는 함수
    ########################################################################################################
    def make_liklihood_plot(self, result_df, cat, n_topics):
        # 0.01, 0.05, 0.1, 0.125, 0.25
        log_likelihood_1 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.01) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_2 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.05) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_3 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.1) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_4 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.125) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_5 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.25) & (result_df.loc[idx,'params']['learning_decay']==0.5)]

        # Show likelihood graph
        plt.figure(figsize=(12, 8))
        plt.plot(n_topics, log_likelihood_1, label='0.01')
        plt.plot(n_topics, log_likelihood_2, label='0.05')
        plt.plot(n_topics, log_likelihood_3, label='0.1')
        plt.plot(n_topics, log_likelihood_4, label='0.125')
        plt.plot(n_topics, log_likelihood_5, label='0.25')
        plt.title("Choosing Optimal LDA Model")
        plt.xlabel("Num Topics")
        plt.ylabel("Log Likelyhood Scores")
        plt.legend(title='Alpha', loc='best')
        plt.savefig('./classifier/results/Dictionary/likelihood/{}/{}_best.jpg'.format(month, cat))
        plt.ioff()
        plt.close()

    def make_catagory_Dict(self):
        data =  self.data
        data['Category1'] = data['Category1'].astype('category')
        data['Category2'] = data['Category2'].astype('category')
        ####################################################################################################
        ### The process of extracting key words through LDA from articles categorized by category
        ### category1 또는 2로 분류된 중요기사에서 상위 15개의 단어를 추출하는 작업
        ####################################################################################################
        category = data['Category1'].cat.categories.tolist()
        category = [cat.strip() for cat in category]
        category = list(set(category))
        print(category)
        # gridserch할 토픽 갯수 리스트
        n_topics = [2, 3, 4]
        start_1 = time.time()
        make_data_first = True
        try:
            total_df = pd.read_csv('./classifier/results/Dictionary/total_topics.csv', engine='python', encoding='cp949')
            tmp_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
            make_data_first = False
        except FileNotFoundError:
            tmp_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
        for cat in category:
            print(cat)
            df = data[(data['Category1'] == cat) | (data['Category2'] == cat)]
            print("data dim: ",df.shape)

            text = df["Text"].values.tolist()
            corpus = pp.total_preprocess(text)
            corpus_join = pp.merge_text_list(corpus)

            try:
                vectorizer = CountVectorizer(analyzer='word',
                                             min_df=10,                       # minimum reqd occurences of a word
                                             stop_words='english',             # remove stop words
                                             token_pattern='[a-zA-Z0-9]{1,}')  # num chars > 1

                data_vectorized = vectorizer.fit_transform(corpus_join)
                feature_names = vectorizer.get_feature_names()
                # Build LDA Model
                lda_model = LatentDirichletAllocation(learning_method='batch',
                                                      random_state=777,          # Random state
                                                      evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                                      n_jobs = -1)               # Use all available CPUs
                # 최적의 토픽 수를 찾기위한 GridSearch의 Parameters
                search_params = {'n_components': n_topics, \
                                 'learning_decay': [.5, .7, .9],\
                                 'doc_topic_prior':[0.01, 0.05, 0.1, 0.125, 0.25]}

                start = time.time()
                model = GridSearchCV(lda_model, param_grid=search_params)
                model.fit(data_vectorized)
                end = time.time()
                best_lda_model = model.best_estimator_
                print("Best Model's Params:", model.best_params_)
                print(end - start)
                print("Model Perplexity:", best_lda_model.perplexity(data_vectorized))
                print("Best Log Likelihood Score:", model.best_score_)
                cat_data_file = cat.replace("/", "_")

                data_df = pp.display_topics(best_lda_model, feature_names, 15, cat)
                data_df.to_csv('./classifier/results/Dictionary/Topic_csv/{}/{}_topics.csv'.format(month, cat_data_file), index=False, encoding='cp949')
                tmp_df = tmp_df.append(data_df)
                panel = sklvis.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
                pyLDAvis.save_html(panel, './classifier/results/LDA_html/{}/lda_{}.html'.format(month, cat_data_file))

                result_df = pd.DataFrame(model.cv_results_)
                self.make_liklihood_plot(result_df=result_df, cat=cat_data_file, n_topics=n_topics)
            except ValueError:
                continue
        end_1 = time.time()
        print(end_1 - start_1)
        # keywords 사전에서 월마다 단어를 업데이트하는 과정
        if make_data_first:
            tmp_df['Month'] = month
            total_df = tmp_df
        else:
            tmp_df['Month'] = month
            total_df = total_df.append(tmp_df)
        total_df.to_csv('./classifier/results/Dictionary/total_topics.csv', index=False, encoding='cp949')

    def make_keyword_score(self, method='title'):
        headline = pd.read_csv("./classifier/results/Dictionary/keyword_headline.csv", engine="python")
        total_keywords = pd.read_csv("./classifier/results/Dictionary/total_topics.csv", engine="python")
        # Frequency가 1이상인 단어만 filter
        headline = headline[headline['Frequency'] > 1]

        # headline score를 1~2사이로 normalize
        headline['Frequency'] = headline['Frequency'].apply(lambda x: round(((x-headline['Frequency'].min())/(headline['Frequency'].max()-headline['Frequency'].min()) + 0.5), 2))
        headline.to_csv("./classifier/results/Score/headline_score.csv", index=False)

        if method == 'lda':
            total_keywords = total_keywords.groupby(['Words', 'Category'], as_index=False).count()
            total_keywords.sort_values(['Category','Topic'], ascending=[True, False], inplace=False)
            total_keywords.rename(columns ={'Topic':'Score'}, inplace=True)
            # 두개의 토픽에 다 있으면 1점 한개에 토픽에 단어만 있으면 0.5점이므로 2로 나누어준다.
            total_keywords['Score'] = total_keywords['Score'] / 2
            total_keywords_score = total_keywords.pivot(index='Words', columns='Category', values='Score').reset_index().fillna(0)
            total_keywords_score.columns = ['Word', 'International', 'Regulation', 'Corporate', 'Law', 'Market', 'Tech']
            total_keywords_score.to_csv("./classifier/results/Score/total_keywords_score.csv", encoding='cp949', index=False)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.method == 'title':
        md = MakeKewordDict(mode="title")
        md.title_keword_extract()
        md.make_keyword_score()
    elif args.method == 'lda':
        month = input("Topic keyword를 갱신하려는 달을 입력하세요 ex) 01 : ")
        md = MakeKewordDict(month=month, mode="lda")
        md.title_keword_extract()
        md.make_catagory_Dict()
        md.make_keyword_score(method='lda')
    else:
        print("title or lda 중에 선택하세요")
