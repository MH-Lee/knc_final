import pandas as pd
import numpy as np
import ast
import operator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from gensim.summarization import keywords
from datetime import datetime
import inflect
import argparse
p = inflect.engine()
nltk.download('punkt')
nltk.download("stopwords")
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
###############################################################################
### 1. Set stop words
###############################################################################
def parse_args():
    # set make data parser
    parser = argparse.ArgumentParser(description='score data 만들기')
    parser.add_argument('--method', help='select auto or manual', default='auto', type=str)
    args = parser.parse_args()
    return args

class MakeScoreData:
    def __init__(self, date='Today'):
        if date == 'Today':
            self.today1 = datetime.today().strftime("%Y-%m-%d")
            self.today2 = datetime.today().strftime("%Y%m%d")
        else:
            self.today1 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y-%m-%d")
            self.today2 = datetime.strptime(date, "%Y-%m-%d").date().strftime("%Y%m%d")
        print("{}".format(self.today1), "extract start!")

    def filter_wordtoken(self, text):
        word_token = word_tokenize(text)
        pos_list = ['JJ','JJR','JJS','NN', 'NNP', 'VB', 'VBG', 'VBP', 'VBN', 'VBZ',\
                    'RB','RBR','RBS']
        stop_words = list(set(stopwords.words('english')))
        stop_words_add = [',','.','’','reuters', 'photo', 'file', 'inc.', 'corp.', \
                        'techcrunch','u.s.', 'u.k', 'york', 'mac']
        cor_list = pd.read_csv('./Company.csv')['Name'].tolist()
        cor_list = [cor.lower() for cor in cor_list]
        stop_words.extend(stop_words_add)
        filtered_sentence = [w for w in word_token if not w.lower() in stop_words]
        filtered_sentence = [w for w in filtered_sentence if not w.lower() in cor_list]
        filtered_sentence = [word for word in filtered_sentence if len(word) > 1]
        filtered_sentence = [word for word in filtered_sentence if not word.isnumeric()]
        filtered_sentence = [word for word, pos in pos_tag(filtered_sentence) if pos in pos_list]
        return filtered_sentence

    def plural_to_singular(self, word_token):
        return list(set([p.singular_noun(word) if p.singular_noun(word) != False else word for word in word_token]))

    ###############################################################################
    ###############################################################################
    ### 2. make score
    ###############################################################################
    def article_scoring(self):
        article_train = pd.read_excel('./classifier/data/{}/all_article_{}.xlsx'.format(self.today1, self.today2))
        imp_word =pd.read_csv('./classifier/data/important_word.csv')
        cor_list = pd.read_csv('./Company.csv')['Name'].tolist()
        imp_word.fillna(0, inplace=True)
        cor_list = [cor.lower() for cor in cor_list]
        cor_list.extend(['amazon', 'lg','hynix'])

        importance_word = imp_word[imp_word['importance'] == 1]['words'].tolist()
        crypto_c_word = imp_word[imp_word['crypto'] == 1]['words'].tolist()
        law_word = imp_word[imp_word['law'] == 1]['words'].tolist()
        senti_word = imp_word[imp_word['sentiment'] == 1]['words'].tolist()
        tech_word = imp_word[imp_word['tech'] == 1]['words'].tolist()
        regulation_word = imp_word[imp_word['regulation'] == 1]['words'].tolist()

        article_train['Keyword'] = article_train['Keyword'].apply(lambda x: ast.literal_eval(x))
        # word plural to singular
        article_train['Keyword'] = article_train['Keyword'].apply(lambda word_token:self.plural_to_singular(word_token))
        col_dict = {'com_word':cor_list, 'imp_word':importance_word, 'sent_word':senti_word,\
                    'tech_word':tech_word, 'law_word':law_word, 'reg_word':regulation_word,\
                    'crypto_c_word':crypto_c_word}
        for col in col_dict:
            article_train[col] = article_train['Keyword'].apply(lambda x:list(set(x).intersection(col_dict[col])))
            article_train['{}_len'.format(col)] = article_train[col].apply(lambda x:len(x))
        print(article_train.columns)
        article_train['important_score'] = article_train[['com_word_len','imp_word_len', 'crypto_c_word_len',\
                                                            'sent_word_len', 'tech_word_len', 'reg_word_len',\
                                                            'law_word_len']].sum(axis=1)
        article_train['important_score'].mean()
        article_train['important_score'].describe()
        article_train.to_excel('./classifier/data/{}/article_score.xlsx'.format(self.today1), index=False)


###############################################################################
### 3. exe setting
###############################################################################
if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.method == 'auto':
        ms = MakeScoreData()
    else:
        date = input("날짜(YYYY-mm-dd)")
        ms = MakeScoreData(date=date)
    ms.article_scoring()
