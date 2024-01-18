from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import copy
import os
import re

# In case the nltk library got you with error like ..... package not found use download method for your desired package, all packages that are needed were included here
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Dataset Directories
TRAIN_FILEPATH_POS = 'Dataset/aclImdb/train/pos'
TRAIN_FILEPATH_NEG = 'Dataset/aclImdb/train/neg'
TEST_FILEPATH_POS = 'Dataset/aclImdb/test/pos'
TEST_FILEPATH_NEG = 'Dataset/aclImdb/test/neg'


class PreProcess:

    def __init__(self):
        self.X_train = list()  # Reviews of train data
        self.y_train = list()  # Labels of train data
        self.X_test = list()  # Reviews of test data
        self.y_test = list()  # Labels of test data

    @staticmethod
    def load_data():
        # Train files
        X_train = []  # Preprocessing not applied
        y_train = []  # Preprocessing not applied
        # Positive labels files
        files_train_pos = list(os.walk(TRAIN_FILEPATH_POS))[0][-1]
        for file in range(len(files_train_pos)):
            with open(TRAIN_FILEPATH_POS + "/" + str(files_train_pos[file]), 'r') as f:
                X_train.append(f.readlines())
                y_train.append(1)  # 1 is the labels of positive reviews

        # Negative labels files
        files_train_neg = list(os.walk(TRAIN_FILEPATH_NEG))[0][-1]
        for file in range(len(files_train_neg)):
            with open(TRAIN_FILEPATH_NEG + "/" + str(files_train_neg[file]), 'r') as f:
                X_train.append(f.readlines())
                y_train.append(-1)  # -1 is the labels of negative reviews

        # Test files
        X_test = []  # Preprocessing not applied
        y_test = []  # Preprocessing not applied
        # Positive labels files
        files_test_pos = list(os.walk(TEST_FILEPATH_POS))[0][-1]
        for file in range(len(files_test_pos)):
            with open(TEST_FILEPATH_POS + "/" + str(files_test_pos[file]), 'r') as f:
                X_test.append(f.readlines())
                y_test.append(1)  # 1 is the labels of positive reviews

        # Negative labels files
        files_test_neg = list(os.walk(TEST_FILEPATH_NEG))[0][-1]
        for file in range(len(files_test_neg)):
            with open(TEST_FILEPATH_NEG + "/" + str(files_test_neg[file]), 'r') as f:
                X_test.append(f.readlines())
                y_test.append(-1)  # -1 is the labels of negative reviews

        return X_train, X_test, y_train, y_test

    @staticmethod
    def case_folding(text: str):  # Handle upper case characters
        new_word = text.lower()
        return new_word

    @staticmethod
    def special_characters_remover(text: str):  # Eliminates all the special characters like {, . : ; }
        normalized_word = re.sub(r'[^\w\s]', '', text)
        return normalized_word

    @staticmethod
    def tokenizer(text: str):  # Tokenize the text
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def stop_word_remover(tokens: list):  # Eliminate stop words
        stop_words = set(stopwords.words('english'))
        new_tokens = []
        for token in tokens:
            if token not in stop_words:
                new_tokens.append(token)
        return copy.deepcopy(new_tokens)

    @staticmethod
    def stemmer(tokens: list):  # Stemming the tokens
        stemmer_obj = PorterStemmer()
        stemmed_tokens = [stemmer_obj.stem(token) for token in tokens]
        return stemmed_tokens

    @staticmethod
    def lemmatizer(tokens: list):  # Lemmatize the tokens
        lemmatizer_obj = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer_obj.lemmatize(token) for token in tokens]
        return lemmatized_tokens

    def preprocess(self):
        # Get raw data loaded by load_data method
        X_train, X_test, y_train, y_test = PreProcess.load_data()

        # Preprocess

        # Train set
        temp = []
        for data in X_train:
            # Case folding
            data = PreProcess.case_folding(copy.deepcopy(data[0]))
            # Special characters remover
            data = PreProcess.special_characters_remover(copy.deepcopy(data))
            # Tokenizing
            data = PreProcess.tokenizer(copy.deepcopy(data))
            # Stop word remover
            data = PreProcess.stop_word_remover(copy.deepcopy(data))
            # Stemming
            data = PreProcess.stemmer(copy.deepcopy(data))
            # Lemmatization
            data = PreProcess.lemmatizer(copy.deepcopy(data))
            # Append processed data
            temp.append(copy.deepcopy(data))
        X_train = copy.deepcopy(temp)

        # Test set
        temp = []
        for data in X_test:
            # Case folding
            data = PreProcess.case_folding(copy.deepcopy(data[0]))
            # Special characters remover
            data = PreProcess.special_characters_remover(copy.deepcopy(data))
            # Tokenizing
            data = PreProcess.tokenizer(copy.deepcopy(data))
            # Stop word remover
            data = PreProcess.stop_word_remover(copy.deepcopy(data))
            # Stemming
            data = PreProcess.stemmer(copy.deepcopy(data))
            # Lemmatization
            data = PreProcess.lemmatizer(copy.deepcopy(data))
            # Append processed data
            temp.append(copy.deepcopy(data))
        X_test = copy.deepcopy(temp)

        # Initializing properties
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

a = PreProcess()
a.preprocess()
print(len(a.y_train))