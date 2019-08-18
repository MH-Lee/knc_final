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

pp = PreProcessing()
np.random.seed(777)

class MakeKewordDict:
    def __init__(self):
        self.data = pd.read_excel("./classifier/data/important_article/knc_importance.xlsx")
        if os.path.exists('./classifier/results/') == False:
            os.mkdir('./classifier/results/')
        if os.path.exists('./classifier/results/Dictionary') == False:
            os.mkdir('./classifier/results/Dictionary')
        if os.path.exists('./classifier/results/LDA_html') == False:
            os.mkdir('./classifier/results/LDA_html')
        if os.path.exists('./classifier/results/Dictionary/Topic_csv') == False:
            os.mkdir('./classifier/results/Dictionary/Topic_csv')
        if os.path.exists('./classifier/results/Dictionary/likelihood') == False:
            os.mkdir('./classifier/results/Dictionary/likelihood')
        print("start!")

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
        title_data['Category1'] = title_data['Category1'].apply(lambda x: x.strip())
        title_data['Category2'] = title_data['Category2'].apply(lambda x: str(x).strip())
        category = title_data['Category1'].astype('category').cat.categories.tolist()
        category = [cat.strip() for cat in category]
        category = list(set(category))

        title_keyword = pd.DataFrame(columns=["Word", "Frequency", "Category"])

        for cat in category:
            tmp = title_data.loc[(title_data["Category1"]==cat ) | (title_data["Category2"]==cat), "Title2"].tolist()
            corpus = []
            for text in tmp:
                tmp2 = " ".join(text)
                corpus.append(tmp2)

            corpus_tf = CountVectorizer().fit(corpus)
            count = corpus_tf.transform(corpus).toarray().sum(axis=0)
            idx = np.argsort(-count)

            count = count[idx]
            feature_name = np.array(corpus_tf.get_feature_names())[idx]
            corpus_tf = list(zip(feature_name, count))

            for i in range(0, len(corpus_tf)):
                title_keyword = title_keyword.append({'Word':corpus_tf[i][0],\
                                                    'Frequency':corpus_tf[i][1],\
                                                    'Category':cat}, ignore_index=True)
        title_keyword.to_csv("./classifier/results/Dictionary/keyword_headline.csv", index=False, encoding="cp949")

    def make_liklihood_plot(self, result_df, cat, n_topics):
        # 0.01, 0.05, 0.1, 0.125, 0.25
        log_likelihood_1 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.01) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_2 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.05) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_3 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.1) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_4 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.125) & (result_df.loc[idx,'params']['learning_decay']==0.5)]
        log_likelihood_5 = [round(result_df.loc[idx,'mean_test_score']) for idx in range(result_df.shape[0]) if (result_df.loc[idx,'params']['doc_topic_prior']==0.25) & (result_df.loc[idx,'params']['learning_decay']==0.5)]

        # Show graph
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
        plt.savefig('./classifier/results/Dictionary/likelihood/{}_best.jpg'.format(cat))
        plt.ioff()
        plt.close()


    def make_catagory_Dict(self):
        data =  self.data
        data['Category1'] = data['Category1'].astype('category')
        data['Category2'] = data['Category2'].astype('category')
        ####################################################################################################
        ### Pre-processing
        ####################################################################################################
        category = data['Category1'].cat.categories.tolist()
        category = [cat.strip() for cat in category]
        category = list(set(category))
        print(category)
        n_topics = [2, 3, 4]
        start_1 = time.time()
        total_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
        for cat in category:
            print(cat)
            df = data[(data['Category1'] == cat) | (data['Category2'] == cat)]
            print("data dim: ",df.shape)

            text = df["Text"].values.tolist()
            corpus = pp.total_preprocess(text)
            corpus_join = pp.merge_text_list(corpus)

            vectorizer = CountVectorizer(analyzer='word',
                                         min_df=10,                       # minimum reqd occurences of a word
                                         stop_words='english',             # remove stop words                 # convert all words to lowercase
                                         token_pattern='[a-zA-Z0-9]{1,}')  # num chars > 1

            data_vectorized = vectorizer.fit_transform(corpus_join)
            feature_names = vectorizer.get_feature_names()
            # Build LDA Model
            lda_model = LatentDirichletAllocation(learning_method='batch',
                                                  random_state=777,          # Random state
                                                  evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                                  n_jobs = -1)               # Use all available CPUs


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
            data_df.to_csv('./classifier/results/Dictionary/Topic_csv/{}_topics.csv'.format(cat_data_file), index=False, encoding='cp949')
            total_df = total_df.append(data_df)
            panel = sklvis.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
            pyLDAvis.save_html(panel, './classifier/results/LDA_html/lda_{}.html'.format(cat_data_file))

            result_df = pd.DataFrame(model.cv_results_)
            self.make_liklihood_plot(result_df=result_df, cat=cat_data_file, n_topics=n_topics)
        end_1 = time.time()
        print(end_1 - start_1)
        total_df.to_csv('./classifier/results/Dictionary/total_topics.csv', index=False, encoding='cp949')

    def make_keyword_score(self):
        knc_word = pd.read_excel("./classifier/data/knc_word.xlsx")
        headline = pd.read_csv("./classifier/results/Dictionary/keyword_headline.csv", engine="python")
        total_keywords = pd.read_csv("./classifier/results/Dictionary/total_topics.csv", engine="python")

        knc_word['Scores'] = 1
        knc_word = knc_word[['Words', 'Scores']]
        knc_word.to_csv("./classifier/results/Score/knc_score.csv", encoding='cp949', index=False)
        headline =headline[headline['Frequency'] > 1]
        headline['Frequency'] = headline.groupby('Category')['Frequency'].apply(lambda x: round(((x-min(x))/(max(x)-min(x)) + 1), 2))
        headline_score = headline.pivot(index='Word', columns='Category', values='Frequency').fillna(0).reset_index()
        headline_score.columns = ['Word', 'International', 'Regulation', 'Corporate', 'Law', 'Market', 'Tech']
        headline_score.to_csv("./classifier/results/Score/headline_score.csv", encoding='cp949', index=False)

        total_keywords = total_keywords.groupby(['Words', 'Category'], as_index=False).count()
        total_keywords.sort_values(['Category','Topic'], ascending=[True, False], inplace=False)
        total_keywords.rename(columns ={'Topic':'Score'}, inplace=True)
        total_keywords['Score'] = total_keywords['Score'] / 2
        total_keywords_score = total_keywords.pivot(index='Words', columns='Category', values='Score').reset_index().fillna(0)
        total_keywords_score.columns = ['Word', 'International', 'Regulation', 'Corporate', 'Law', 'Market', 'Tech']
        total_keywords_score.to_csv("./classifier/results/Score/total_keywords_score.csv", encoding='cp949', index=False)

if __name__ == '__main__':
    md = MakeKewordDict()
    md.title_keword_extract()
    md.make_catagory_Dict()
    md.make_keyword_score()
