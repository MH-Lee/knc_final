import operator
import time, os, argparse
import pandas as pd
# ./classifier/packages에 있는 전처리 기능
from packages.PreProcess import PreProcessing
from nltk.probability import FreqDist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from collections import Counter

pp_dict = PreProcessing(mode='dictionary')
pp_key = PreProcessing(mode='keywords')

################################################################################
##### 단순이 단일 title과 text로 점수를 산출할 경우에는
##### Co-occurence matrix를 만들 수 없어 coo_score가 제외됩니다.
################################################################################

################################################################################
##### 해당기능을 따로 분리하고 싶으시면
##### ./classifier/packages PreProcessing.py
##### ./classifier/results/Score/headline_score.csv
##### ./classifier/results/Score/total_keywords_score.csv
##### company_data에 폴더안에 데이터가 필요합니다.
################################################################################

class RetrunScore:
    def __init__(self, method='title'):
        start_init_ = time.time()
        self.method = method
        self.headline_score = pd.read_csv('./classifier/results/Score/headline_score.csv', engine='python', index_col=0)
        self.total_topic_score = pd.read_csv('./classifier/results/Score/total_keywords_score.csv', engine='python', index_col=0)
        # company list selected by k.c
        self.cor_list = pd.read_csv('./company_data/Company.csv')['Name'].tolist()
        self.cor_list = [cor.lower() for cor in self.cor_list]
        # load tech list
        self.tech_list = pd.read_csv('./company_data/Tech.csv')['Tech'].tolist()
        # load fortune global 500 company list
        self.etc_cor = pd.read_csv('./company_data/fortune_g.csv')['Company'].tolist()
        self.etc_cor = [cor.lower() for cor in self.etc_cor]
        end_init_ = time.time()
        print("PreProcess is done. time:{}".format(end_init_ - start_init_))

    def extract_words(self, title, text=None):
        # PreProcess
        self.info_dict = dict()
        start_extract_ = time.time()
        self.info_dict['Title'] = pp_key.replace(title)
        self.info_dict['Title_keyword'] = pp_key.text_preprocess(self.info_dict['Title'])
        self.info_dict['Title_keyword_Freq'] = dict(FreqDist(self.info_dict['Title_keyword']))
        self.info_dict['Title_keyword_unique'] = self.info_dict['Title_keyword_Freq'].keys()
        if self.method == 'lda':
            if text==None:
                raise Exception('method가 LDA일 경우 Text를 반드시 입력하세요')
            self.info_dict['Full_text'] = pp_dict.replace(text)
        end_extract_ = time.time()
        print(end_extract_ - start_extract_)
        return self

    # LDA keywords extraction for full text scoring
    def LDA_keywords_extract(self):
        start_LDA_ = time.time()
        text = self.info_dict['Full_text']
        vectorizer = CountVectorizer(analyzer='word',
                                     stop_words='english',             # remove stop words
                                     token_pattern='[a-zA-Z0-9]{1,}')  # num chars > 1

        data_vectorized = vectorizer.fit_transform([text])
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
        # Show lda keywords extraction task progress
        #  Extract 10 keywords
        for topic in lda_model.components_:
            top_word = [feature_names[i] for i in topic.argsort()[:10]]
        self.info_dict['LDA_keywords'] = top_word
        end_LDA_ = time.time()
        print(end_LDA_ - start_LDA_)
        return self

    def make_score(self):
        self.info_dict['common_headline'] = list(set(self.info_dict['Title_keyword_unique']).intersection(self.headline_score.index.tolist()))
        # belong to the list of k.c companies, give 1 points.
        tech_score1  = len(set(self.info_dict['Title_keyword_unique']).intersection(self.cor_list))
        # belong to the list of Fortune 500 companies, give 0.5 points.
        tech_score2 = len(set(self.info_dict['Title_keyword_unique']).intersection(self.etc_cor))*0.5

        # merge k.c company socre and fortune company score
        self.info_dict['Company_score'] = tech_score1 + tech_score2
        self.info_dict['Tech_score'] = len(set(self.info_dict['Title_keyword_unique']).intersection(self.etc_cor))
        # title keywords score 산출
        self.info_dict['headline_score'] = round(sum([self.headline_score.loc[x].values for x in self.info_dict['common_headline']])[0],2)

        if self.method == 'lda':
            # 사전과 full text keywords 사이에 교집합 counting
            self.info_dict['common_full_LDA'] = list(set(self.info_dict['LDA_keywords']).intersection(self.total_topic_score.index.tolist()))
            first_loop = True
            for x in self.info_dict['common_full_LDA']:
                if first_loop:
                    tmp = Counter(dict(self.total_topic_score.loc[x]))
                else:
                    lda_category = dict(tmp + Counter(dict(self.total_topic_score.loc[x])))
                    self.info_dict['keyword_score_LDA'] = lda_category
                    self.info_dict['category1'] = max(lda_category, key=lda_category.get)
                    second_max_keys = list(dict(sorted(lda_category.items(), key=lambda kv: kv[1], reverse=True)).keys())[1]
                    if lda_category[second_max_keys] > 3:
                        self.info_dict['category2'] = second_max_keys
                first_loop = False
            self.info_dict['keyword_score_total_LDA'] = round(max(self.info_dict['keyword_score_LDA'].values()),2)
            self.info_dict['Total_score'] = round(sum([self.info_dict[x] for x in ['Company_score', 'Tech_score', 'headline_score', 'keyword_score_total_LDA']]),3)
        else:
            self.info_dict['Total_score'] = round(sum([self.info_dict[x] for x in ['Company_score', 'Tech_score', 'headline_score']]),3)
        return self.info_dict

    def score_result(self, title, text=None):
        self.extract_words(title, text)
        score_result = dict()
        if self.method == 'title':
            result_idx = ['Company_score', 'Tech_score', 'headline_score',  'Total_score']
        else:
            self.LDA_keywords_extract()
            result_idx = ['Company_score', 'Tech_score', 'headline_score',  'category1', 'keyword_score_total_LDA', 'Total_score']
        info_dict = self.make_score()
        for idx in result_idx:
            score_result[idx] = info_dict[idx]
        return info_dict, score_result

title = "Samsung’s Galaxy Fold concierge service is live in the US for those who need it – TechCrunch"

text = """
"Part of Samsung’s reboot of the Galaxy Fold was the announcement of a Premiere Service. Along with a reinforced version of the phone and a lot more warning labels, the company announced that it would also be a 24/7 care service…just in case something happened with the device.
I had some issues with my in just over a day, after not running into any trouble with the original version of the phone. Given how gingerly the company insists users act with the device, my issue doesn’t appear to be particularly widespread — good news for Samsung on that front. Even so, this sort of things feels pretty necessary for a $2,000 (and up) phone that is effectively in mass beta testing.
Two weeks after making the device available in the States, Premier Service has gone live. Sammobile noted the addition of Fold Concierge via a new software update, bringing with it support via phone or video chat. The list of potentially helpful features ranges from on-boarding with the device to a $149, same-day screen replacement service. That can be accommodated in person at a number of locations.
It’s a pretty unique offer from a big consumer electronics company — though the Fold is nothing if not unique, I suppose. I’ve got a fuller write up of my impressions of the handset here. The TLDR version is the I can’t recommend the purchase of what is very much a first generation device that’s double the price of a standard flagship. If you’re so inclined, however, Samsung’s got a hotline for you."
"""
###############################################################################
###### info_dict : 전체 분석 정보
###### score_result : 점수결과 dictionary
###############################################################################


###############################################################################
###### Test용 코드
###############################################################################
rs = RetrunScore(method='lda')
info_dict, score_result = rs.score_result(title, text)
score_result
print("Title :",info_dict['Title'])
print("Text :",info_dict['Full_text'])
print("Full_text results :",score_result)
print("===============================")
rs = RetrunScore(method='title')
info_dict, score_result = rs.score_result(title)
score_result
print("Title results :",score_result)
