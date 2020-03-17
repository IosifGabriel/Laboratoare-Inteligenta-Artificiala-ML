import numpy as np
import matplotlib.pyplot as plt

trainimages = np.loadtxt('train_images.txt')
trainlabes = np.loadtxt('train_labels.txt', 'int')
testimages = np.loadtxt('test_images.txt')
testlabels = np.loadtxt('test_labels.txt', 'int')


def values_to_bins(DataMatrix, num_bins):
    bins = np.linspace(start=0, stop=255, num=num_bins)
    return np.digitize(DataMatrix, bins) - 1


xtrain = values_to_bins(trainimages, 5)
xtest = values_to_bins(testimages, 5)


class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):

        distances = []
        if (metric == 'l1'):
            distances = np.sum(abs(self.train_images - test_image), axis=1)
        else:
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1))

        distance2 = np.argsort(distances)
        k_mindistance = distance2[:num_neighbors]
        selectedlabels = self.train_labels[k_mindistance]
        frequency = np.bincount(selectedlabels)

        return np.argmax(frequency)

    def classify_images(self, test_images, numneighbourns, metric):
        f = open("predictii_3nn_l2_mnist.txt", "w+")
        predictlist = np.zeros(testimages.shape[0])
        for i in range(testimages.shape[0]):
            predictlist[i] = self.classify_image(test_images[i], numneighbourns, metric )

        from sklearn.metrics import accuracy_score
        f.write(str(accuracy_score(testlabels , predictlist)) +"\n")
        return accuracy_score(testlabels, predictlist)

if __name__ == '__main__':
    Knn = KnnClassifier(trainimages, trainlabes)
    Knn.classify_images(testimages,3,'l2')

    moreneighbourns = []
    neighbournsl1 =[]
    for i in [1,3,5,7,9]:
        moreneighbourns.append(Knn.classify_images(testimages, i, 'l2'))
        neighbournsl1.append(Knn.classify_images(testimages, i, 'l1'))

    print(moreneighbourns)
    plt.plot(np.array(moreneighbourns))
    plt.plot(np.array(neighbournsl1))
    plt.show()
