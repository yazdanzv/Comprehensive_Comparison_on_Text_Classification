class Embedding:
    def __init__(self, X_train: list, X_test: list):
        self.X_train = X_train
        self.X_test = X_test

    def word2vec(self):
