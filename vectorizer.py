from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test
        self.tfidf_vecs_train = None
        self.tfidf_vecs_test = None
        self.model_train = None
        self.model_test = None
        self.train_tfidf_weights = None
        self.test_tfidf_weights = None

    def vectorize(self):  # Vectorize documents and returns TFIDF scores
        self.model_train = TfidfVectorizer(stop_words='english')
        self.model_test = TfidfVectorizer(stop_words='english')
        self.tfidf_vecs_train = self.model_train.fit_transform([" ".join(tokens) for tokens in self.X_train])
        self.tfidf_vecs_test = self.model_test.fit_transform([" ".join(tokens) for tokens in self.X_train])

    def word_weight(self):  # For transformation function on document embeddings creation
        # Convert the TF-IDF matrix for a specific document to a dictionary
        # Train data
        train_feature_names = self.model_train.get_feature_names_out()
        doc_index = 0  # for the first document
        tfidf_scores = self.tfidf_vecs_train[doc_index].toarray().flatten()
        self.train_tfidf_weights = dict(zip(train_feature_names, tfidf_scores))

        # Test data
        test_feature_names = self.model_test.get_feature_names_out()
        doc_index = 0  # for the first document
        tfidf_scores = self.tfidf_vecs_train[doc_index].toarray().flatten()
        self.test_tfidf_weights = dict(zip(test_feature_names, tfidf_scores))
