####################################################################################################
### Project  : Kim and Chang - News recommend system
### Script   : Keyword_Headline.py
### Contents : Keyword extraction from headline
####################################################################################################

####################################################################################################
### Setting up environment
####################################################################################################

# Pakcages
from sklearn.feature_extraction.text import CountVectorizer
from PreProcess.PreProcess           import PreProcessing
import pandas as pd
import numpy  as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus   import stopwords, wordnet
# Pre Processing class
pp = PreProcessing()

####################################################################################################
### Setting up data
####################################################################################################

df = pd.read_excel("../Data/Article.xlsx")

# Check the data
df.head()
df.shape
df.isnull().sum()

####################################################################################################
### Pre process the title of article
####################################################################################################

title = df["Title"].tolist()

# Step 01. Change all characters into lower case
corpus = pp.total_preprocess(title)

# Attach the pre-processed data into dataframe
df["Title2"] = None
for i in range(0, len(df)):
    df["Title2"][i] = " ".join(corpus[i])

# Re pre processing
tmp = df["Title2"].tolist()
tmp = pp.tokenize(tmp)
tmp = pp.remv_total_stopwords(tmp)
tmp = pp.pos_tag(tmp, mode="title")
tmp = pp.remv_total_stopwords(tmp)
tmp = pp.remv_short_words(tmp)
df["Title2"] = tmp

df
####################################################################################################
### Keyword extraction for each category
####################################################################################################

title_keyword = pd.DataFrame(columns=["Word", "Frequency", "Category"])

# "국제" 카테고리
tmp = df.loc[(df["Category1"]=="국제" ) | (df["Category2"]=="국제"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"국제"}, ignore_index=True)

# "규제/규제기관" 카테고리
tmp = df.loc[(df["Category1"]=="규제/규제기관" ) | (df["Category2"]=="규제/규제기관"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"규제/규제기관"}, ignore_index=True)

# "기업" 카테고리
tmp = df.loc[(df["Category1"]=="기업" ) | (df["Category2"]=="기업"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"기업"}, ignore_index=True)

# "시장" 카테고리
tmp = df.loc[(df["Category1"]=="시장" ) | (df["Category2"]=="시장"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"시장"}, ignore_index=True)

# "법/분쟁/소송" 카테고리
tmp = df.loc[(df["Category1"]=="법/분쟁/소송" ) | (df["Category2"]=="법/분쟁/소송"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"법/분쟁/소송"}, ignore_index=True)

# "제품/기술/서비스" 카테고리
tmp = df.loc[(df["Category1"]=="제품/기술/서비스" ) | (df["Category2"]=="제품/기술/서비스"), "Title2"].tolist()
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
    title_keyword = title_keyword.append({'Word':corpus_tf[i][0], 'Frequency':corpus_tf[i][1], 'Category':"제품/기술/서비스"}, ignore_index=True)


####################################################################################################
### Save result into csv file
####################################################################################################
title_keyword.to_csv("../Result/keyword_headline.csv", index=False, encoding="cp949")
