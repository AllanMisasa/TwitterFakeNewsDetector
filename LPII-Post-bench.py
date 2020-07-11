import json
import glob
import numpy as np
from pathlib import Path
import emoji
from textblob import TextBlob
from textblob import Word
import re
from time import time
import pandas as pd
import gensim
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

DIREC = Path("en")


def removeexcess(string):
    """
    Description: removes all the excess from the tweets'
    Input: File with tweets
    Output: Clean list of tweets

    """
    tweet_list = []
    for tweet in string:
        tweet_list.append(tweet[19:-14])
    return tweet_list


def remove_stopwords(string):
    """
    DOCSTRING: Removes stopwords from a string using SPACY
    Input: A string
    Output: The same string with all stopwords removed

    """
    cleaned_lemmas = []
    for tokens in string:
        # Does not append all words in the stopwords list provided by SPACY
        if tokens not in stopwords:
            cleaned_lemmas.append(tokens)
    # returns list without stopwords
    return cleaned_lemmas


# this removes symbols from a string list
def remove_symbols_list(string_list):
    """
    DOCSTRING: removes symbols and the word "url"
    Input: a list
    Output: the same list with all symbols and the word "url" removed

    """
    # joins list as string
    list_clean = " ".join(string_list)
    # removes all symbols from string
    list_clean = re.sub(r'[^\w]', ' ', list_clean)
    # removes the word url from string
    list_clean = re.sub(r' url ', " ", list_clean)
    # returns list
    return list_clean.split()


# this function removes symbols from a string, this one is relevant for the sentiment with tex
def remove_symbols_string(string):
    """
    DOCSTRING: removes symbols and the word "url"
    Input: a string
    Output: the same string with all symbols and the word "url" removed

    """
    # removes all symbols from string
    string_clean = re.sub(r'[^\w]', ' ', string)
    # removes the word url from string

    string_clean = string_clean.replace("#HASHTAG#", '')
    string_clean = string_clean.replace("#URL#", '')
    string_clean = string_clean.replace("#USER#", '')
    string_clean = string_clean.replace("'", '')
    # returns string
    return string_clean


# combines the list of all the pre-processed tweets together into one string

def exclamation_count(string):
    return string.count('!')


def question_count(string):
    return string.count('?')


def hashtag_count(string):
    return string.count('#HASHTAG#')


def user_count(string):
    return string.count('#USER#')


def tweet_length(string):
    return len(string)


def all_tweets(inputs):
    """
    DOCSTRING: Puts all the tweets together into one string
    Input: a list of strings
    Output: one string containing all elements from list
    """
    all_tweets = []
    for tweet in inputs:
        all_tweets.append(tweet)
    return "".join(all_tweets)

'''
def profanity(doc):
	if doc.contains_profanity() == True:
		return 1
	else:
		return 0
'''

# Features
# Emoji extraction raw
def extract_emojis(string):
    """
    Description: This uses the emoticon library to extract emojis from a string
    necessary: import emoji

    if needed:
    pip install emoji

    input: string
    output: emojis in string
    """
    return ''.join(c for c in string if c in emoji.UNICODE_EMOJI)


# Percentage of upper case letter extraction
def upper_percent_extraction(string):
    """
    Description: Gives the percent of a string that is upper case.

    Input: a string
    Output: the percent of a string that is capitalized as a float
    """
    upper_count = 0
    for letter in string:
        if letter.isupper():
            upper_count += 1
    return (upper_count / len(string))


# Named entity extraction using Textblob
def named_entities_labels(string):
    """
    DOCSTRING: Uses Spacy to return two lists of the named entity text and labels from a string
    Input: A string
    Output: The named entities and labels in two lists
    REQUIREMENTs: The string needs to have been passed through SPACY (nlp)
    """
    named_entities_labels = []
    for ent in string.ents:
        named_entities_labels.append(ent.label_)
    return named_entities_labels


# Propernoun extraction using Spacy
def ProperNoun_extraction(pos):
    """
    Description: Shows count of Proper Nouns from SPACY pos
    Input: Pos of Tweet data
    Output: Amount of Proper Nouns

    Requires: Spacy PROPN

    """
    count = 0
    for word in pos:
        if word is "PROPN":
            count += 1
    return count


def emo_sum(doc):
    '''
    input: spacy document
    output: 10x1 vector with emotion counts
    '''
    tmp = [0] * 10
    for token in doc:
        if token.lemma_ in emolex_words.word.values:
            tmp += emolex_words[emolex_words.word == token.lemma_].iloc[0][1:11]
    return dict(tmp)


def count_pos(doc):
    '''
    input: spacy document
    output: percentage of nouns, adjectives etc. in the document
    '''
    dc = dict()
    c = Counter(([token.pos_ for token in doc]))
    sbase = sum(c.values())
    for el, cnt in c.items():
        dc[el] = cnt / sbase
    return dc


# Spell checker, sensitive to symbols (make sure to remove symbols before using)
def spell_checker(string):
    """
    DOCSTRING: Checks the spelling of words and returns a count of misspelled words
    Input: A string
    Output: An integer of misspelled words
    REQUIREMENTS: Word from TextBlob

    BE AWARE: Does not work with symbols in string, for example (,) (") (!)
    """
    count = 0
    for word in string.split():
        if Word(word).spellcheck()[0][1] < 1:
            count += 1
        elif Word(word).spellcheck()[:][0][0] != word:
            count += 1
    return count


class TfidfEmbeddingVectorizer(object):
    '''
    a class that trains word embeddings based on a certain corpus, and then transforms them based on the tf-idf value
    '''

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 25

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


'''
load the spacy english file
load the emotion lexicon dataset
change the shape of emolex for easier searching, and then fill empty fields with 0
'''
nlp = spacy.load('en_core_web_sm')
filepath = "C:/Data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], skiprows=45, sep='\t')
emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
emolex_words.fillna(0)
'''
# Loading Clickbait.txt file for use - currently directory for mac!
Clickbait_results = "D:/Clickbait_results.txt"
open_clickbait_text = open(Clickbait_results)
Click_res = open_clickbait_text.read()
Click_res = json.loads(Clickbait_results)
open_clickbait_text.close()
'''

def process_user(FILE):
    f = open(FILE, encoding='utf-8')
    l = f.read()
    f.close()

    tweets = re.findall(r'<document>.+?</document>', l)
    tweets_refined = removeexcess(tweets)
    all_tweets_together1 = all_tweets(tweets_refined[:])
    dc = dict()

    dc['tweet length'] = tweet_length(all_tweets_together1)
    dc['qmarks'] = question_count(all_tweets_together1)
    dc['epoint'] = exclamation_count(all_tweets_together1)
    dc['hashtags'] = hashtag_count(all_tweets_together1)
    dc['users'] = user_count(all_tweets_together1)

    all_tweets_together = remove_symbols_string(all_tweets_together1)  # remove weird symbols
    all_tweets_together = re.sub(' +', ' ', all_tweets_together)  # convert multiple spaces into one space

    dc['upper'] = upper_percent_extraction(all_tweets_together)
    all_tweets_together = all_tweets_together.lower()  # optional?

    doc = nlp(all_tweets_together)  # read the cleaned up list of tweets intop spacy

    '''
    note: the function below take a long time to run. comment them out to speed up processing
    '''
    dc.update(count_pos(doc).items())  # merge a dictionary containing pos counts into dc
    dc.update(emo_sum(doc).items())  # merge a dictionary containing emotion counts into dc

    TXB = TextBlob(all_tweets_together)
    dc['polarity'] = TXB.sentiment[0]
    dc['subjectivity'] = TXB.sentiment[1]

    # Named Entities Labels
    # Descriptions retrieved from: https://spacy.io/api/annotation#named-entities

    # Companies, agencies, institutions, etc.
    dc['ORG'] = named_entities_labels(doc).count('ORG')
    # Countries, cities, states.
    dc['GPE'] = named_entities_labels(doc).count('GPE')
    # Monetary values, including unit.
    dc['MONEY'] = named_entities_labels(doc).count('MONEY')
    # People, including fictional.
    dc['PERSON'] = named_entities_labels(doc).count('PERSON')
    # Absolute or relative dates or periods.
    dc['DATE'] = named_entities_labels(doc).count('DATE')
    # Nationalities or religious or political groups.
    dc['NORP'] = named_entities_labels(doc).count('NORP')
    # Buildings, airports, highways, bridges, etc.
    dc['FAC'] = named_entities_labels(doc).count('FAC')
    # Non-GPE locations, mountain ranges, bodies of water.
    dc['LOC'] = named_entities_labels(doc).count('LOC')
    # Objects, vehicles, foods, etc. (Not services.)
    dc['PRODUCT'] = named_entities_labels(doc).count('PRODUCT')
    # Named hurricanes, battles, wars, sports events, etc.
    dc['EVENT'] = named_entities_labels(doc).count('EVENT')
    # Titles of books, songs, etc.
    dc['WORK_OF_ART'] = named_entities_labels(doc).count('WORK_OF_ART')
    # Named documents made into laws.
    dc['LAW'] = named_entities_labels(doc).count('LAW')
    # Any named language.
    dc['LANGUAGE'] = named_entities_labels(doc).count('LANGUAGE')
    # Times smaller than a day.
    dc['TIME'] = named_entities_labels(doc).count('TIME')
    # Percentage, including ”%“.
    dc['PERCENT'] = named_entities_labels(doc).count('PERCENT')
    # Measurements, as of weight or distance.
    dc['QUANTITY'] = named_entities_labels(doc).count('QUANTITY')
    # “first”, “second”, etc.
    dc['ORDINAL'] = named_entities_labels(doc).count('ORDINAL')
    # Numerals that do not fall under another type.
    dc['CARDINAL'] = named_entities_labels(doc).count('CARDINAL')

    """
    Spelling mistake check (below) is very time consuming so it is currently commented out
    """
    #dc['spelling mistakes'] = spell_checker(TXB)

    return dc, doc


GT = DIREC / "truth.txt"
true_values = {}
f = open(GT)
for line in f:
    linev = line.strip().split(":::")
    true_values[linev[0]] = linev[1]
f.close()

texts = []
X = []
y = []
c = 0

for FILE in DIREC.glob("*.xml"):
    # The split command below gets just the file name,
    # without the whole address. The last slicing part [:-4]
    # removes .xml from the name, so that to get the user code
    USERCODE = str(FILE).split("/")[-1][3:-4]

    # This function should return a vectorial representation of a user
    repr, doc = process_user(FILE)

    # adding in clickbait feature, not in the function!
    #repr['Clickbait Percent'] = Click_res[USERCODE]

    # We append the representation of the user to the X variable
    # and the class to the y vector
    X.append(repr)
    texts.append([word.lemma_ for word in doc])
    y.append(true_values[USERCODE])
texts = np.array(texts)
y = np.array(y)

df = pd.DataFrame(X)
df = df.dropna(axis='columns')

'''
train word embeddings based on all the documents
'''
model = gensim.models.Word2Vec(texts, size=25)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))
sss = TfidfEmbeddingVectorizer(w2v)
sss.fit(texts, y)

'''
create tf-idf weighted embeddings of each user
'''
embeddings = sss.transform(texts)
accs = []
df = pd.DataFrame(pd.np.column_stack([df, embeddings]))
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(df, y):
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]

    xg = XGBClassifier(random_state=42)
    xg.fit(X_train, y_train)
    predictions = xg.predict(X_test)
    plot_roc_curve(xg, X_test, y_test)
    plt.show()


'''
for train_index, test_index in skf.split(df, y):
    y_train, y_test = y[train_index], y[test_index]
    X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    #ext = ExtraTreesClassifier()
    #rf = RandomForestClassifier(n_estimators = 50, min_samples_split = 5, min_samples_leaf= 4, max_features = 'auto', max_depth= 80, bootstrap= True, n_jobs=-1, random_state=42)
    #ada = AdaBoostClassifier(n_estimators=50, random_state=42)
    #gd = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xg = XGBClassifier(random_state=42)
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)

    #rf.fit(X_train, y_train)
    xg.fit(X_train, y_train)
    #gd.fit(X_train, y_train)
    #logmodel = LogisticRegression(solver='lbfgs')
    #predictions = rf.predict(X_test)
    predictions = xg.predict(X_test)
    accscore= accuracy_score(y_test, predictions)
    #print(accuracy_score(y_test, predictions))
    accs.append(accscore)
    print(confusion_matrix(y_test, predictions))
print(np.mean(accs))

# Backtranslater not in use!
# def back_translate_es(string):
#    """
#    DOCSTRING: Translates string to spanish and then English
#    Input: A string in any language
#    Output: A TextBlob in English
#    REQUIREMENTS: Textblob as TextBlob
#    """
#    espanol = TextBlob(string).translate(to = "es")
#    english = espanol.translate(to = "en")
#    return str(english)

def benchmark(clf):
    accuracies = []
    for train_index, test_index in skf.split(df, y):
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    score = accuracy_score(y_test, pred)
    accuracies.append(score)
    print("accuracy:   %0.3f" % score)
    clf_descr = str(clf).split('(')[0]
    return clf_descr, np.mean(accuracies), train_time, test_time


results = []

for clf, name in (
        (RandomForestClassifier(random_state=42), "Unoptimized RF"),
        (RandomForestClassifier(n_estimators = 50, min_samples_split = 5, min_samples_leaf= 4, max_features = 'auto', max_depth= 80, bootstrap= True, n_jobs=-1, random_state=42), "Optimized RF"),
        (AdaBoostClassifier(n_estimators=50, random_state=42), "AdaBoost Classifier"),
        (XGBClassifier(random_state=42), "XG Boost Classifier"),
        (LinearSVC(random_state=42), "Linear SVC")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time", color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
'''