import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error

data   = np.load("training_data.npy")
prices = np.load("prices.npy")

#1
def normalize_data (train_data, test_data):
    sc = preprocessing.StandardScaler()
    sc.fit(train_data)
    sc_train= sc.transform(train_data)
    sc_test= sc.transform(test_data)
    return (sc_train, sc_test)

def mse_and_mae (model, train_data, train_labels, test_data, test_labels):
    norm_train, norm_test = normalize_data(train_data, test_data)
    fitter = model.fit(norm_train, train_labels)
    predictions = fitter.predict(norm_test)
    mse = mean_squared_error (test_labels, predictions)
    mae = mean_absolute_error(test_labels, predictions)
    return mse, mae

# randomizer
data, prices = shuffle(data, prices, random_state=0)
num_samples_fold = len(data) // 3

# split data in 3 folds
data1 = data[:num_samples_fold]
prices1 = prices [:num_samples_fold]
data2 = data[ num_samples_fold : 2 * num_samples_fold]
prices2 = prices [ num_samples_fold : 2 * num_samples_fold]
data3 = data   [2 * num_samples_fold:]
prices3 = prices [2 * num_samples_fold:]



#2
model = LinearRegression()
mse12, mae12 = mse_and_mae(model, np.concatenate((data1, data2)), np.concatenate((prices1, prices2)), data3, prices3)
mse13, mae13 = mse_and_mae(model, np.concatenate((data1, data3)), np.concatenate((prices1, prices3)), data2, prices2)
mse23, mae23 = mse_and_mae(model,np.concatenate((data2, data3)), np.concatenate((prices2, prices3)), data1, prices1)

mean_mse = (mse12 + mse13 + mse23) / 3
mean_mae = (mae12 + mae13 + mae23) / 3
print(mean_mse, mean_mae)



#3
best_mae    = 0
best_mse    = -1
best_alpha  = 0

for alpha in [1, 10, 100, 1000]:
    model = Ridge(alpha = alpha)
    mse12, mae12 = mse_and_mae(model, np.concatenate((data1, data2)), np.concatenate((prices1, prices2)), data3, prices3)
    mse13, mae13 = mse_and_mae(model, np.concatenate((data1, data3)), np.concatenate((prices1, prices3)), data2, prices2)
    mse23, mae23 = mse_and_mae(model, np.concatenate((data2, data3)), np.concatenate((prices2, prices3)), data1, prices1)
    mean_mse = (mse12 + mse13 + mse23) / 3
    mean_mae = (mae12 + mae13 + mae23) / 3

    if  best_mse == -1 or (mean_mae + mean_mse) / 2 < (best_mae + best_mse) / 2:
        best_mae = mean_mae
        best_mse = mean_mse
        best_alpha = alpha

print("Cel mai bun alpha si mse/mae aferente:", best_alpha, best_mse,
                                                best_mae)


#4
model       = Ridge(best_alpha)
scaler      = preprocessing.StandardScaler()
scaler.fit(data)
norm_data   = scaler.transform(data)
model.fit(norm_data, prices)

attributes = ["Year", "Kilometers Driven", "Mileage",
            "Engine", "Power", "Seats", "Owner Type",
            "Fuel Type", "Transmission"]

print("Coeficientii sunt: ",     model.coef_)
print("Biasul regresiei este: ", model.intercept_)

print("Cel mai semnificativ atribut este: ",
      attributes[np.argmax(np.abs(model.coef_))])

print("Al doilea cel mai semnificativ atribut este: ",
        attributes[np.argmax(np.abs(model.coef_)) + 1])

print("Cel mai nesemnificativ atribut este:",
      attributes[np.argmin(np.abs(model.coef_))])