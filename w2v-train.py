from gensim.models import word2vec
import sys

sentences = word2vec.LineSentence(sys.argv[1])
model = word2vec.Word2Vec(sentences, size=300, min_count=1, sg=1, window=3)
model.save(sys.argv[2])
