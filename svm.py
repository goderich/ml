#!/usr/bin/env python3

from math import log
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
from datetime import timedelta
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams


stopwords = stopwords.words('english')

def ngramize(lines, n):
    ls = LancasterStemmer()
    d = {}
    wc = 0
    lc = 0

    for line in lines:
        lc += 1
        score = int(line[0])
        # My functional programming background
        # is painfully obvious from this code.
        # If only Python had function composition,
        # a pipelining operator, or threading macros...
        n_grams = list(
            map(tuple,
                map(sorted,
                    ngrams(
                        map(ls.stem,
                            filter(lambda x: x not in ',.?'
                                   and x not in stopwords,
                                   word_tokenize(line[2:]))),
                        n))))
        wc += len(n_grams)
        d[lc] = (score, n_grams)

    return d, wc


def calculate_tfidf(d, wc):
    tfidf_dict = {}
    # calculate how many times each word (n-gram)
    # appears in the dataset
    for linum, (_, wds) in d.items():
        for w in wds:
            if w in tfidf_dict:
                tfidf_dict[w].append(linum)
            else:
                tfidf_dict[w] = [linum]

    lc = len(d)
    # based on the first loop, calculate the
    # tf-idf of each n-gram
    for w, linums in tfidf_dict.items():
        tf = len(linums) / wc
        # i'm currently only using tf, not tf-idf,
        # for ease of calculation with a split dataset
        # idf = log(lc / len(set(linums)))
        tfidf_dict[w] = tf # * idf

    # normalize tf-idf values
    min_v = min(tfidf_dict.values())
    max_v = max(tfidf_dict.values())
    tfidf_dict = {k: ((v-min_v)/(max_v-min_v))
                  for k, v in tfidf_dict.items()}

    return tfidf_dict


def review2vec(ngram_dict, tfidf_dict):
    scores = []
    vectors = []

    for _, (score, n_gram) in ngram_dict.items():
        vector = np.fromiter(
            (v if k in n_gram
             else 0
             for k, v in tfidf_dict.items()),
            float)
        scores.append(score)
        vectors.append(vector)

    np_scores = np.array(scores, int)
    np_vectors = np.array(vectors, float)

    return (np_scores, np_vectors)

with open('data/rt-polarity', 'r', encoding='latin-1') as f:
    lines = [l.strip() for l in f.readlines()]
    train_data, test_data = train_test_split(
        lines, test_size=0.2, random_state=42)

for n in [2, 3]:
    ngram_dict_train, wc_train = ngramize(train_data, n)
    ngram_dict_test, wc_test = ngramize(test_data, n)
    tfidf_train = calculate_tfidf(ngram_dict_train, wc_train)
    s_train, v_train = review2vec(ngram_dict_train, tfidf_train)
    s_test, v_test = review2vec(ngram_dict_test, tfidf_train)
    print(f'Total number of n-grams:'
          f' {wc_train + wc_test}.\n'
          f'({wc_train} in training data '
          f'and {wc_test} in testing data.)')

    for kernel in ['linear', 'rbf']:
        for C in [1]:
            print(f'Training dataset with unordered {n}-grams,'
                  f' C = {C},'
                  f' and {kernel} kernel.')
            start_time = time.time()
            clf = svm.SVC(C=C, kernel=kernel)
            clf.fit(v_train, s_train)
            end_time = time.time()
            t = timedelta(seconds = end_time - start_time)
            score = clf.score(v_test, s_test)
            print(f'Score for test data in batch: {score}\n'
                f'Training time: {t}\n')
