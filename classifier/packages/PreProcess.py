####################################################################################################
### Project  : Kim and Chang - News recommend system
### Script   : PreProcesser.py
### Contents : Text pre-processing
####################################################################################################

####################################################################################################
### Setting up environment
####################################################################################################

# Packages
import pandas as pd
import gensim
import re, os
import nltk

from nltk.corpus   import stopwords, wordnet
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag      import pos_tag
# Etc
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")

class PreProcessing:
    def __init__(self):

        # Stopwords of English
        self.en_stopwords = stopwords.words("english")
        self.en_stopwords.remove("against")
        # User-defined stopwords
        # print(os.getcwd())
        my_stopwords = pd.read_csv("./classifier/packages/Data/Stopwords.csv", engine="python")["Stopwords"].tolist()
        company      = pd.read_csv("./classifier/packages/Data/Company.csv",   engine="python")["Company"].tolist()
        magazine     = pd.read_csv("./classifier/packages/Data/Magazine.csv",  engine="python")["Magazine"].tolist()
        self.total_stopwords = my_stopwords + self.en_stopwords + company + magazine
        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

   # Step1. Change all characters into lower case
    def base_process(self, input_text):
        result = []
        for text in input_text:
            text = text.lower()
            text = re.sub(r"\s{2,}", " ", text)
            text = re.sub("\n", " ", text)
            text = re.sub("[^A-Za-z]+", " ", text)
            result.append(text.lower())
        return result

    # Step 03. Tokenize
    def tokenize(self, input_text):
        result = []
        for text in input_text:
            result.append(nltk.word_tokenize(text))
        return result

    # Step 04. Remove english stopwords
    def remv_en_stopwords(self, input_text):
        result = []
        for text in input_text:
            tmp = []
            for word in text:
                if word not in self.en_stopwords:
                    tmp.append(word)
            result.append(tmp)
        return result

    # Step 05. Remove punctuation
    def remv_punc(self, input_text):
        result = []
        for text in input_text:
            tmp=[]
            for word in text:
                tmp.append(re.sub("[^A-Za-z]+", " ", word))
            result.append(tmp)
        return result

    # Step 06. Remove white space
    def remv_white_space(self, input_text):
        result = []
        for text in input_text:
            tmp=[]
            for word in text:
                if (word != " "):
                    tmp.append(word)
            result.append(tmp)
        return result

    # Step 09. Lemmatization
    def get_wordnet_pos(self,treebank_tag):
        if treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('J'):
            return wordnet.NOUN
        else:
            return wordnet.ADJ

    def pos_tag(self,tokens, mode='article'):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        # CD & JJ(adjective or numeral, ordinal(ex: first))
        pos_tokens = [nltk.pos_tag(token) for token in tokens]
        if mode == 'article':
            # lemmatization using pos tagg
            # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
            pos_tokens = [[self.lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)) for word, pos_tag in token_
                           if  pos_tag.startswith('V') or pos_tag.startswith('N') or pos_tag.startswith('v') or pos_tag.startswith('n')] for token_ in pos_tokens]
        else:
            pos_tokens = [[self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)) for word, pos_tag in token_] for token_ in pos_tokens]
        return pos_tokens

    # Step 10. Remove my stopwords
    def remv_total_stopwords(self, input_text):
        result = []
        for text in input_text:
            tmp=[]
            for word in text:
                if (word not in self.total_stopwords):
                    tmp.append(word)
            result.append(tmp)
        return result

    # Step 11. Remove one-length character (length of words < 2)
    def remv_short_words(self, input_text):
        result = []
        for text in input_text:
            tmp=[]
            for word in text:
                if (len(word) > 1):
                    tmp.append(word)
            result.append(tmp)
        return result

    # etc
    def merge_text_list(self, input_text):
        result = []
        for text in input_text:
            tmp = " ".join(text)
            result.append(tmp)
        return result

    def display_topics(self, model, feature_names, no_top_words, category):
        topic_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
        for topic_idx, topic in enumerate(model.components_):
            tmp_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
            tmp_df['Words'] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            tmp_df['Topic'] = "Topic {}".format(topic_idx)
            topic_df = topic_df.append(tmp_df)
        topic_df['Category'] = category
        return topic_df

    def total_preprocess(self, text):
        step1 = self.base_process(text)
        step2 = self.tokenize(step1)
        step3 = self.remv_en_stopwords(step2)
        step4 = self.remv_punc(step3)
        step5 = self.remv_white_space(step4)
        step6 = self.remv_total_stopwords(step5)
        step7 = self.pos_tag(step6, mode='article')
        step8 = self.remv_total_stopwords(step7)
        corpus_preprocess = self.remv_short_words(step8)
        return corpus_preprocess
