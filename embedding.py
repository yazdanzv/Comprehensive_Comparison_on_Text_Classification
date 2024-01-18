from gensim.models import Word2Vec
import numpy as np


class Embedding:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test

    @staticmethod
    def get_word_vector(word, model):
        try:
            return model.wv[word]
        except KeyError:  # Word not in the model
            return np.zeros(model.vector_size)  # Return a zero vector

    def word2vec(self):
        model = Word2Vec(self.X_train, vector_size=100, window=5, min_count=1, workers=4)
        train_vectors = [[Embedding.get_word_vector(word, model) for word in sentence] for sentence in self.X_train]
        test_vectors = [[Embedding.get_word_vector(word, model) for word in sentence] for sentence in self.X_test]
    