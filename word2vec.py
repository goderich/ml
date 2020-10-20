#!/usr/bin/env python3

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
from datetime import timedelta
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.models

stopwords = stopwords.words('english')
w2v = gensim.models.Word2Vec.load('data/model')


def safe_get_line_vector(wordlist):
    vectors = []
    for w in wordlist:
        try:
            v = w2v.wv[w]
            vectors.append(v)
        except:
            pass
    return np.mean(vectors, axis=0)


def review2vec(lines):
    scores = []
    vectors = []
    for line in lines:
        score = int(line[0])
        vector = safe_get_line_vector(
            word_tokenize(line[2:]))
        scores.append(score)
        vectors.append(vector)

    np_scores = np.array(scores, int)
    np_vectors = np.array(vectors)

    return (np_vectors, np_scores)


with open('data/rt-polarity', 'r', encoding='latin-1') as f:
    lines = [l.strip() for l in f.readlines()]
    train_data, test_data = train_test_split(
        lines, test_size=0.2, random_state=42)

v_train, s_train = review2vec(train_data)
v_test, s_test = review2vec(test_data)

for kernel in ['linear', 'rbf']:
    for C in [1, 0.1, 0.01]:
        print(f'Training dataset with C = {C},'
              f' and {kernel} kernel.')
        start_time = time.time()
        clf = svm.SVC(C=C, kernel=kernel)
        clf.fit(v_train, s_train)
        end_time = time.time()
        t = timedelta(seconds = end_time - start_time)
        score = clf.score(v_test, s_test)
        print(f'Score for test data in batch: {score}\n'
            f'Training time: {t}\n')
