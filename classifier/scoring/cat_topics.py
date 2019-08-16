from PreProcess.PreProcess import PreProcessing
import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as gensimvis
import pyLDAvis.sklearn as sklvis
from sklearn.feature_extraction.text import CountVectorizer
import time

data = pd.read_excel("../Data/Article.xlsx")
data['Category1'] = data['Category1'].astype('category')
data['Category2'] = data['Category2'].astype('category')
pp = PreProcessing()
np.random.seed(777)
####################################################################################################
### Pre-processing
####################################################################################################
category = data['Category1'].cat.categories.tolist()
category = [cat.strip() for cat in category]
category = list(set(category))
print(category)

# corpus, dictionary = pp.implement_preprocessing()
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

    search_params = {'n_components': [2, 3, 4], 'learning_decay': [.5, .7, .9], 'doc_topic_prior':[0.01, 0.05, 0.1, 0.125, 0.25]}
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
    data_df.to_csv('./Topic_csv/{}_topics.csv'.format(cat_data_file), index=False, encoding='cp949')
    total_df = total_df.append(data_df)
    # pyLDAvis.enable_notebook()
    panel = sklvis.prepare(best_lda_model, data_vectorized, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, '../Result/LDA_html/lda_{}.html'.format(cat_data_file))

    result_df = pd.DataFrame(model.cv_results_)
    n_topics = [2, 3, 4]
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
    plt.savefig('../Result/likelihood/{}_best.jpg'.format(cat_data_file))
    plt.ioff()
    plt.close()
end_1 = time.time()
print(end_1 - start_1)
total_df.to_csv('../Result/total_topics.csv', index=False, encoding='cp949')
