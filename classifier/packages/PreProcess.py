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
# 처음 실행할 때 주석을 제거하여 설치 해주어야함
# nltk.download("stopwords")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("wordnet")

# 영어의 약어 표현을 바꾸어주는 정규 표현식
replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
    (r'(\w+)’s', '\g<1>'),
]

class PreProcessing:
    def __init__(self, mode='dictionary', patterns=replacement_patterns):
        # Stopwords of English
        self.en_stopwords = stopwords.words("english")
        self.en_stopwords.remove("against")
        self.pattern = patterns
        # User-defined stopwords
        # print(os.getcwd())
        # 미리 설정해둔 불용어 리스트를 로딩 상황에 맞게 단어를 추가해줄 수 있음
        my_stopwords = pd.read_csv("./classifier/packages/Data/Stopwords.csv", engine="python")["Stopwords"].tolist()
        company      = pd.read_csv("./classifier/packages/Data/Company.csv",   engine="python")["Company"].tolist()
        magazine     = pd.read_csv("./classifier/packages/Data/Magazine.csv",  engine="python")["Magazine"].tolist()
        if mode == 'dictionary':
            self.total_stopwords = my_stopwords + self.en_stopwords + magazine + company
        else:
            self.total_stopwords = my_stopwords + self.en_stopwords + magazine
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

    # corpus convert to one sting data format(using LDA keywords extract)
    def merge_text_list(self, input_text):
        result = []
        for text in input_text:
            tmp = " ".join(text)
            result.append(tmp)
        return result

    # make lda dataframe
    def display_topics(self, model, feature_names, no_top_words, category):
        topic_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
        for topic_idx, topic in enumerate(model.components_):
            tmp_df = pd.DataFrame(columns=['Words', 'Category', 'Topic'])
            tmp_df['Words'] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            tmp_df['Topic'] = "Topic {}".format(topic_idx)
            topic_df = topic_df.append(tmp_df)
        topic_df['Category'] = category
        return topic_df

    #  Integrate the whole preprocess
    def total_preprocess(self, text, mode='article'):
        step1 = self.base_process(text)
        step2 = self.tokenize(step1)
        step3 = self.remv_en_stopwords(step2)
        step4 = self.remv_punc(step3)
        step5 = self.remv_white_space(step4)
        step6 = self.remv_total_stopwords(step5)
        if mode == 'article':
            step7 = self.pos_tag(step6, mode='article')
        else:
            step7 = self.pos_tag(step6, mode=mode)
        step8 = self.remv_total_stopwords(step7)
        corpus_preprocess = self.remv_short_words(step8)
        return corpus_preprocess

    # replace the contractions to the original form
    def replace(self, text):
        patterns = [(re.compile(regex), repl) for (regex, repl) in self.pattern]
        s = text
        for (pattern, repl) in patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

    # Pre-process for full text
    def corpus_preprocess(self, corpus):
        n = WordNetLemmatizer()
        corpus_preprocess= []
        for sentence in corpus:
            #print(i)
            # Special Characters
            try:
                text = self.replace(str(sentence))
            except KeyError:
                print(sentence)
            text = re.sub(r",", " ", text)
            text = re.sub(r"\.", " ", text)
            text = re.sub(r"!", " ", text)
            text = re.sub(r"\(", " ( ", text)
            text = re.sub(r"\)", " ) ", text)
            text= re.sub(r"\?", " ", text)
            text = re.sub("[^A-Za-z]"," ", text) # change "match all strings that contain a non-letter" as 1 white spaced
            text = re.sub(r"\s{2,}", " ", text) # change 2 white spaces as 1 white space
            word_tokens = word_tokenize(text.lower())
            result = []
            for w in word_tokens:
                if w not in self.total_stopwords:
                    result.append(w)
            result = [n.lemmatize(w) for w in result]
            # dd = " ".join(result)
            corpus_preprocess.append(result)
        return(corpus_preprocess)
