from gensim.models import Word2Vec, FastText
import numpy as np


class Embedding:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test

    @staticmethod
    def get_word_vector(word, model):  # For Word2Vec embedding (According to documentation)
        try:
            return model.wv[word]
        except KeyError:  # Word not in the model
            return np.zeros(model.vector_size)  # Return a zero vector

    @staticmethod
    def tokens_to_vectors(tokens, model):  # For FastText embedding (According to documentations)
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if not vectors:  # Handle case with no words in the model
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)  # Averaging word vectors

    def word2vec(self):
        model = Word2Vec(self.X_train, vector_size=100, window=5, min_count=1, workers=4)
        train_vectors = [[Embedding.get_word_vector(word, model) for word in sentence] for sentence in self.X_train]
        test_vectors = [[Embedding.get_word_vector(word, model) for word in sentence] for sentence in self.X_test]
        return train_vectors, test_vectors

    def glove(self):
        pass

    def fasttext(self):
        model = FastText(self.X_train, vector_size=100, window=3, min_count=1, workers=4, sg=1)
        train_vectors = [Embedding.tokens_to_vectors(sentence, model) for sentence in self.X_train]
        test_vectors = [Embedding.tokens_to_vectors(sentence, model) for sentence in self.X_test]
        return train_vectors, test_vectors
