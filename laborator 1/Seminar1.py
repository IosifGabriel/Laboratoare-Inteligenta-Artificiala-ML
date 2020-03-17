y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]




def accuracy_score(y_true, y_pred):
    contor = 0
    for i in range(len(y_true)):
        if y_pred[i]==y_true[i]:
            contor += 1

    return contor/len(y_pred)

print(accuracy_score(y_true,y_pred))

def precision_recall_score(y_true, y_pred):
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_pred[i]==1 and  y_true[i]==1:
            tp += 1
        elif y_pred[i]==1 and y_true[i]==0:
            fp += 1
        elif y_pred[i]==0 and y_true[i]==1:
            fn += 1

    Precizie =  tp / (tp+fp)
    Recall = tp / (tp+fn)
    return Precizie, Recall

print(precision_recall_score(y_true,y_pred))

def mse(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += (y_pred[i] - y_true[i])**2

    return sum/len(y_true)

print(mse(y_true,y_pred))


def mae(y_true, y_pred):
    sum = 0
    for i in range(len(y_true)):
        sum += abs(y_pred[i] - y_true[i])

    return sum / len(y_true)


print(mae(y_true, y_pred))
