import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import MultinomialNB

# ex1

antrenare = [(160, 'F'), (165, 'F'), (155, 'F'), (172, 'F'), (175, 'B'), (180, 'B'), (177, 'B'), (190, 'B')]
barbatiInterval = 0
femeiInterval = 0
barbatiTotal = 0
femeiTotal = 0
interval = 0
for i in antrenare:
    if i[1] == 'F':
        femeiTotal = femeiTotal + 1
    else:
        barbtiTotal = barbatiTotal + 1
    if i[0] > 171 and i[0] < 180:
        interval = interval + 1
        if i[1] == 'F':
            femeiInterval = femeiInterval + 1
        else:
            barbatiInterval = barbatiInterval + 1
Pc = femeiTotal / (femeiTotal + barbatiTotal)
Pxc = femeiInterval / (femeiInterval + barbatiInterval)
print('P(femeie|178) = ', Pc * Pxc * interval / (femeiTotal + barbatiTotal))

#ex2
train_images = np.loadtxt('train_images.txt')
train_labels = np.loadtxt('train_labels.txt', 'int')
test_images = np.loadtxt('test_images.txt')
test_labels = np.loadtxt('test_labels.txt', 'int')

def values_to_bins (data_matrix, num_bins):
    bins = np.linspace(start = 0, stop = 255, num = num_bins)
    return np.digitize(data_matrix, bins) - 1

x_train = values_to_bins(train_images, 5)
x_test = values_to_bins(test_images, 5)

# ex3
clf = MultinomialNB()
clf.fit(x_train, train_labels)#xtrain f, labeluri dupa care se corecteaza
print(clf.score(x_test, test_labels))

Rangeofbins = [3, 5, 7, 9, 11]

#ex 4
def test():
    score_values = []
    for i in [3, 5, 7, 9, 11]:
        new_xtrain = values_to_bins(train_images, i)
        new_xtest = values_to_bins(test_images, i)
        clf.fit(new_xtrain, train_labels)
        score_values.append(clf.score(new_xtest, test_labels))
    return score_values

#ex 5
def miss():
    max = np.argmax(test())
    predict = clf.predict(values_to_bins(test_images,Rangeofbins[max]))
    miss = []
    miss.append(np.where(predict != test_labels))
    for i in range(1,10):
        image = np.reshape(x_test[i], (28, 28))
        plt.imshow(image.astype(np.uint8), cmap='gray')
        plt.show()

miss()

#ex6
from sklearn.metrics import confusion_matrix
predict_array = clf.predict(x_test)
my_matrix = sklearn.metrics.confusion_matrix(test_labels,predict_array)
print(my_matrix)