from gensim.models import word2vec
import sys

sentences = word2vec.LineSentence(sys.argv[1])
model = word2vec.Word2Vec(sentences, size=1, sg=1)#, min_count=1)
model.save(sys.argv[2])
