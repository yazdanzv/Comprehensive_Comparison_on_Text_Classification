from gensim.models import Word2Vec, FastText
import numpy as np
import gensim.downloader as api
import copy


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
        model = FastText(self.X_train, vector_size=100, window=5, min_count=1, workers=4, sg=4)
        train_vectors = [[Embedding.tokens_to_vectors(word, model) for word in sentence] for sentence in self.X_train]
        test_vectors = [[Embedding.tokens_to_vectors(word, model) for word in sentence] for sentence in self.X_test]
        return train_vectors, test_vectors

    @staticmethod
    def tfidf_weighted_document_embedding(word_embeddings, tfidf_weights):
        # (Optional) to make program more robust

        # # Checks that word embeddings & TFIDF weights are not empty
        # if len(word_embeddings) == 0 or len(tfidf_weights) == 0:
        #     raise ValueError("Word embeddings and TF-IDF weights lists cannot be empty.")

        # # Checks that the length of them are equal
        # if len(word_embeddings) != len(tfidf_weights):
        #     raise ValueError("The lengths of word embeddings and TF-IDF weights lists must be the same.")


        # Initialize the first numpy array as weighted embeddings
        document_embeddings = []
        weighted_embeddings = np.zeros(len(word_embeddings[0][0]))
        total_weight = 0

        # Make weighted embeddings
        for doc_e, doc_w in zip(word_embeddings, tfidf_weights):
            for embedding, weight in zip(doc_e, doc_w):
                weighted_embeddings += embedding * float(weight)
            document_embeddings.append(copy.deepcopy(weighted_embeddings))

        return document_embeddings
