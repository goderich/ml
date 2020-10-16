# Intro

I started learning to apply Machine Learning to real problems with
this dataset from @JoliLin.
The dataset includes just over 10K reviews from IMDB,
almost all of them in English
(I did see some Portuguese in there as well, but we'll leave it in
and call it 'adversarial learning').
The reviews are coded with `0` for negative,
and `1` for positive.
The goal is for the algorithm to predict whether the review
is negative or positive from the text alone.

At the behest of my mentor Joli, I began with the very basics,
and gradually move towards more sophisticated solutions.
So far the plan looks like this:

- SVM, including:
  - Tokenizing
  - TF-IDF
  - N-grams
  - Stemming
- Word embeddings (word2vec)
- CNNs
- Deep Learning (BERT)

# SVM

Support Vector Machines. A staple of Machine Learning for a long time,
up until the Deep Learning revolution.
SVMs are built on some very complicated math, but luckily it is not
required to actually use them.
I used SVMs from the [scikit-learn library](https://scikit-learn.org/stable/modules/svm.html),
but there are other alternatives as well.

I began with the simplest solution of all:
a hand-written TF-IDF calculation on all words in the dataset
except a list of stopwords and punctuation.
The TF-IDF values were stored in a dictionary
which was then used to vectorize each individual review
(`0` if a given dictionary word was not in the review
and TF-IDF if it was in the review).

I tried out different parameters with my model, specifically
C (the regularization parameter) and different SVM kernels.
Linear and RBF kernels gave the best results with C = 1:
around 65-66% with less than 20 mins training time on our server.
It should be noted that a C = 1 may lead to overfitting, and
usually a lower C is preferred. C tends to be set in tenfold
decrements, so 0.1, 0.01, etc.

The next iteration of the code involved some complicated steps
and resulted in a complete rewrite. I did the following:

- Separated training and testing data from the beginning.
  In real life situations, you don't see the data that will be used
  with your algorithm while you're still training it.
  The algorithm should not break in OOV (out-of-vocabulary) situations.
- Used stemming. Stemming removes inflexional morphology from words,
  sometimes successfully. It is used to calculate the occurrence of
  a single root with multiple inflections as multiple occurrences of
  the same root, and not as completely separate words.
  I chose the Lancaster Stemmer instead of the more commonly used
  Porter Stemmer, because it worked better with certain examples I saw
  (for example, correctly removing gemination from 'programmer').
- Used N-grams. In order to make predictions based on word co-occurrence,
  N-grams are used in preprocessing. N-grams are collections of N words
  that occur continuously in the input, for any N >= 1. I used N = 2 and
  N = 3.

In order to slightly simplify the vectorization, I used TF instead of
TF-IDF values. The problem isn't calculating IDF (that part is trivial),
but rather which number to use for D (that is, total amount of documents)
when calculating these values for testing data.

Mostly because of N-grams (which greatly increased the number of dimensions
in my vector representations), these updated models ran for about 2 hours
each. In the end the results were even worse than the original (around
50%), which could be due to a number of factors:
the early separation of training and testing data,
N-grams on insufficiently large data, using TF values,
or other things.

Still, that was a valuable learning experience,
mostly in that it taught me that writing all of the above by hand
is a huge pain in the ass.
I am now ready to move on to word2vec.
