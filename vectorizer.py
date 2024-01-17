from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test
        self.tfidf_vecs_train = None
        self.tfidf_vecs_test = None

    def vectorize(self):  # Vectorize documents and returns TFIDF scores
        vectorizer_model_train = TfidfVectorizer(stop_words='english')
        vectorizer_model_test = TfidfVectorizer(stop_words='english')
        self.tfidf_vecs_train = vectorizer_model_train.fit_transform([" ".join(tokens) for tokens in self.X_train])
        self.tfidf_vecs_test = vectorizer_model_test.fit_transform([" ".join(tokens) for tokens in self.X_train])

