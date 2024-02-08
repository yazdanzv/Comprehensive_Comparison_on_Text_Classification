import copy
import math
from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test
        self.tfidf_vecs_train = None
        self.tfidf_vecs_test = None
        self.train_tfidf_weights = list()
        self.test_tfidf_weights = list()
        self.df_matrix_train = dict()
        self.df_matrix_test = dict()

    def vectorize(self):  # Vectorize documents and returns TFIDF scores
        model_train = TfidfVectorizer(stop_words='english')
        model_test = TfidfVectorizer(stop_words='english')
        self.tfidf_vecs_train = model_train.fit_transform([" ".join(tokens) for tokens in self.X_train])
        self.tfidf_vecs_test = model_test.fit_transform([" ".join(tokens) for tokens in self.X_train])

    def word_weight(self):  # For transformation function on document embeddings creation
        # Calculate DF matrix
        self.df_calculator()
        # Convert the TF-IDF matrix for a specific document to a dictionary
        # Train data
        for doc in self.X_train:
            temp = []
            for term in doc:
                df = len(self.df_matrix_train[term])
                temp.append(Vectorizer.tfidf_calculator(doc.count(term), df, len(self.X_train)))
            self.train_tfidf_weights.append(copy.deepcopy(temp))

        # Test data
        for doc in self.X_test:
            temp = []
            for term in doc:
                df = len(self.df_matrix_test[term])              
                temp.append(Vectorizer.tfidf_calculator(doc.count(term), df, len(self.X_test)))
            self.test_tfidf_weights.append(copy.deepcopy(temp))

    @staticmethod
    def tfidf_calculator(tf, df, N):
        return tf * math.log2(N / df)
    
    def df_calculator(self):
        # Initialize df matrix
        # Train data
        for doc in self.X_train:
            for term in doc:
                self.df_matrix_train[term] = []

        for doc in self.X_train:
            for term in doc:
                self.df_matrix_train[term].append(self.X_train.index(doc))
            self.df_matrix_train[term] = list(set(self.df_matrix_train[term]))

        # Test data
        for doc in self.X_test:
            for term in doc:
                self.df_matrix_test[term] = []

        for doc in self.X_test:
            for term in doc:
                self.df_matrix_test[term].append(self.X_test.index(doc))
            self.df_matrix_test[term] = list(set(self.df_matrix_test[term]))
