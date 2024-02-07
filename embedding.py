from gensim.models import Word2Vec, FastText
import numpy as np
import gensim.downloader as api


class Embedding:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test

    @staticmethod
    def get_word_embedding(word, model):  # For Word2Vec embedding (According to documentation)
        try:
            return model.wv[word]
        except KeyError:  # Word not in the model
            return np.zeros(model.vector_size)  # Return a zero vector

    @staticmethod
    def get_word_embedding_glove(word, model):  # For GloVe embedding (According to documentation)
        try:
            return model[word]
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
        train_vectors = [[Embedding.get_word_embedding(word, model) for word in sentence] for sentence in self.X_train]
        test_vectors = [[Embedding.get_word_embedding(word, model) for word in sentence] for sentence in self.X_test]
        return train_vectors, test_vectors

    def glove(self):
        glove_model = api.load('glove-wiki-gigaword-100')
        train_vectors = [[Embedding.get_word_embedding_glove(word, glove_model) for word in sentence] for sentence in self.X_train]
        test_vectors = [[Embedding.get_word_embedding_glove(word, glove_model) for word in sentence] for sentence in self.X_test]
        return train_vectors, test_vectors

    def fasttext(self):
        model = FastText(self.X_train, vector_size=100, window=3, min_count=1, workers=4, sg=1)
        train_vectors = [Embedding.tokens_to_vectors(sentence, model) for sentence in self.X_train]
        test_vectors = [Embedding.tokens_to_vectors(sentence, model) for sentence in self.X_test]
        return train_vectors, test_vectors

    @staticmethod
    def tfidf_weighted_document_embedding(word_embeddings, tfidf_weights):
        # Checks that word embeddings & TFIDF weights are not empty
        if not all(word_embeddings) or not all(tfidf_weights):
            raise ValueError("Word embeddings and TF-IDF weights lists cannot be empty.")

        # Checks that the length of them are equal
        if len(word_embeddings) != len(tfidf_weights):
            raise ValueError("The lengths of word embeddings and TF-IDF weights lists must be the same.")

        # Initialize the first numpy array os weighted embeddings
        weighted_embeddings = np.zeros(len(word_embeddings[0]))
        total_weight = 0

        # Make weighted embeddings
        for embedding, weight in zip(word_embeddings, tfidf_weights):
            weighted_embeddings += embedding * weight
            total_weight += weight

        # Normalize the weights
        if total_weight != 0:
            weighted_embeddings /= total_weight

        return weighted_embeddings

    def document_embedding(self, word_embeddings, tfidf_weights):
        document_embedding = self.tfidf_weighted_document_embedding(word_embeddings, tfidf_weights)
        return document_embedding
