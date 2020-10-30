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
- Word embeddings
  - wikipedia2vec
  - gensim
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
I am now ready to move on to word embeddings.

# Word embeddings

More specifically, [wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).
This is a collection of pre-trained vectors for words in a dozen languages,
all trained using Wikipedia data.
The vectors come in 100, 300, and 500 dimensions, with 300 being the most popular.
This is a lot less than my 100K-dimensional vectors for naive bigrams.

## Wikipedia2Vec

These vectors encode some semantic information about words,
for instance the famous example `king - man + woman = queen`
(first mentioned in [Mikolov et al 2013](https://www.aclweb.org/anthology/N13-1090/)).
[NB: The closest vector will actually be `king`, but `queen` is number two.
I'd say that's still quite impressive, especially when compared to techniques like TF-IDF.]

The vector collection size is absolutely massive: around 10GB.
This includes such things as plurals and other derived forms,
so stemming is not necessary with this library.

Of course, the difference from the previous method
is that these vectors are for individual words.
There are two ways I could calculate a vector for a document:

1. Put all the word vectors from a single document (in my case, a review)
   together in one big vector. Naturally this means that reviews with fewer
   words will have shorter vectors. These have to be padded with zeros.
2. Calculate the average of all word vectors in a single document
   and have it be the value for that document.

The latter is apparently used more often and still gives the best results
while also being the quickest to execute.

Using word embeddings with vector averages means I'm manipulating 300-dimensional vectors, a lot smaller than the many tens of thousands of dimensions I had in the naive implementation. This makes learning very quick: a linear kernel takes around 20 seconds on our server, and an rbf kernel around 30 seconds.

The results are also a lot better: 73-75% correct. This drops with a C of 0.01 or lower due to underfitting.

## Training my models with Gensim

I also trained my own models on the (unscored) data with the help of the training script in `w2v-train.py`. On bigger datasets, this can lead to further improvements of results. Evidently, my dataset was not big enough, because the results got a lot worse instead (hovering around 55%). Still, that was an experience in training my own embedding model, and I could come back and play around with it later.

When training a model, there are many parameters to choose from. So far I stuck with the following:

``` python
model = word2vec.Word2Vec(sentences, size=300, min_count=1, sg=1, window=3)
```

The parameters are as follows:

- `size` is obviously just the dimensions of the resulting vectors. I set it to 300 to mimic what wikipedia2vec was doing.
- `min_count` is the smallest amount of times a word can occur in the dataset to be included in the model. My dataset is very small, so I want to include everything, even hapax legomena.
- `sg` is the training algorithm. This can be either skip-gram (`sg=1`) or continuous bag-of-words (CBOW, `sg=0`).
- `window` is how far behind and ahead the algorithm looks for context, in the number of words. The default is 5, but apparently 2 or 3 works best here. Indeed, with my data 3 gave the best results.

# Deep Learning

Finally it's time to do some proper deep learning. For this I am no longer writing my algorithms from scratch, but instead relying on Joli's code from the `classification_sample` directory (with some debugging of stale code).

## Deep Learning with pretrained embeddings

The first step was to use DL with pretrained word embeddings.
For this I used two models:

- A Google News vector model from Joli's repo,
- and the Wikipedia2Vec model from before.

With this method, learning is iterative
and divided into epochs (capped at 300),
with an early stop
if the `val_loss` value does not shrink for 7 epochs.
Training time is around 3 seconds per epoch on our server.
With both models, training stopped at around 110-120 epochs.

Final accuracy was around 73-75%.
Not great, but not bad either.
