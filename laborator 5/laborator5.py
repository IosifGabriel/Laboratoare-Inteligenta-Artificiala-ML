import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score, classification_report


training_data = np.load('training_sentences.npy', allow_pickle=True)
training_labels = np.load('training_labels.npy', allow_pickle=True)
test_data = np.load('test_sentences.npy', allow_pickle=True)
test_labels = np.load('test_labels.npy', allow_pickle=True)

def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1' or type == 'l2':
        scaler = preprocessing.Normalizer(norm=type)

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return scaled_train_data, scaled_test_data
    else:
        return train_data, test_data


class BagOfWords:

    def __init__(self):
        self.vocab = {}
        self.words = []

    def build_vocabulary(self, train_data):
        for sentence in train_data:
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = len(self.words)
                    self.words.append(word)
        return len(self.words)

    def get_features(self, data):
        result = np.zeros((data.shape[0], len(self.words)),dtype='uint16')
        for idx, sentence in enumerate(data):
            for word in sentence:
                if word in self.vocab:
                    result[idx, self.vocab[word]] += 1
        return result

bow_model = BagOfWords()
bow_model.build_vocabulary(training_data)

#Transforming text into numeric
training_features = bow_model.get_features(training_data)
testing_features = bow_model.get_features(test_data)

norm_train, norm_test = normalize_data(training_features, testing_features, type='l2')

model = svm.SVC(C=1.0, kernel='linear')
model.fit(norm_train, training_labels)
test_preds = model.predict(norm_test)
accuracy_score(test_labels, test_preds)
f1_score(test_labels, test_preds)
print(classification_report(test_labels, test_preds))


#printing the SPAM and good words
weights = np.squeeze(model.coef_)
indexes = np.argsort(weights)
words = np.array(bow_model.words)
print('NEGATIVE', words[indexes[:10]])
print('POSITIVE', words[indexes[-10:]])
