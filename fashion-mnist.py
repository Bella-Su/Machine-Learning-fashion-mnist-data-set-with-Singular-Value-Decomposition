import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ------------------------------------------------------------------------------
# load the training and test data set
# ------------------------------------------------------------------------------
data_train = pd.read_csv("fashion-mnist_train.csv", sep = ',')
data_test = pd.read_csv("fashion-mnist_test.csv", sep = ',')
##print(data_train.head())
# label  pixel1  pixel2  pixel3  pixel4  ...  pixel780  pixel781  pixel782  pixel783  pixel784
# 0      2       0       0       0       0  ...         0         0         0         0         0
# 1      9       0       0       0       0  ...         0         0         0         0         0
# 2      6       0       0       0       0  ...         0         0         0         0         0
# 3      0       0       0       0       1  ...         1         0         0         0         0
# 4      3       0       0       0       0  ...         0         0         0         0         0

X_train, y_train = data_train.drop(['label'], axis=1),data_train['label']
X_test, y_test = data_test.drop(['label'], axis=1),data_test['label']

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# (60000,)
# (60000, 784)
# (10000,)
# (10000, 784)


# ------------------------------------------------------------------------------
# apply SVD
# ------------------------------------------------------------------------------

## Normalize the data
X_train = X_train/255
X_test = X_test/255
# print(X_train)
# [0.00784314 0.03529412 0.02352941 ... 0.03137255 0.03137255 0.02745098] shape of (60000, 784)

## scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

## reduce the number of dimensions with SVD
svd = TruncatedSVD(n_components=300)
svd.fit(X_train)
print(svd.explained_variance_ratio_)

X_train_new = svd.transform(X_train)
X_test_new = svd.transform(X_test)


# ------------------------------------------------------------------------------
# training models after SVD
# ------------------------------------------------------------------------------
print('\n\n------------------After SVD---------------------')

## logistic regression
clf = LogisticRegression(C=10, max_iter=500)
clf.fit(X_train_new, y_train)
print(f'\nLogistic Regression accuracy after SVD = {clf.score(X_test_new,y_test)}')

## gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train_new,y_train)
print(f'Gussian Naive Bayes accuracy after SVD = {clf.score(X_test_new,y_test)}')

## K-NN
# convert to np for spilte the dataset
X_train_new = np.array(X_train_new) 
y_train = np.array(y_train) #shape pf (60000, )

# get the validation set from the trainning set
X_train_new_knn, X_validation_new_knn, y_train_knn, y_validation_knn=train_test_split(X_train_new,y_train, train_size=0.7)

# tuning process for finding best k
best_k = 1
best_acc = 0
for i in range(1,10,2):
    clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    clf.fit(X_train_new_knn,y_train_knn)
    accuracy = clf.score(X_validation_new_knn,y_validation_knn)

    if accuracy > best_acc:
        best_acc = accuracy
        best_k = i

print("\nThe best k is: ", best_k)
print("----------------------------------------")

# apply K-NN
clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
clf.fit(X_train_new,y_train)
print(f'K-NN accuracy after SVD = {clf.score(X_test_new,y_test)}')


# ------------------------------------------------------------------------------
# training models before SVD
# ------------------------------------------------------------------------------
print('\n\n------------------Before SVD---------------------')

## logistic regression
clf = LogisticRegression(C=10, max_iter=500)
clf.fit(X_train, y_train)
print(f'\nLogistic Regression accuracy before SVD = {clf.score(X_test,y_test)}')

## gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train,y_train)
print(f'Gussian Naive Bayes accuracy before SVD = {clf.score(X_test,y_test)}')

## K-NN
# get the validation set from the trainning set
X_train_knn, X_validation_knn, y_train_knn, y_validation_knn=train_test_split(X_train,y_train, train_size=0.7)

# tuning process for finding best k
best_k = 1
best_acc = 0
for i in range(1,10,2):
    clf = KNeighborsClassifier(n_neighbors=i,n_jobs=-1)
    clf.fit(X_train_knn,y_train_knn)
    accuracy = clf.score(X_validation_knn,y_validation_knn)

    if accuracy > best_acc:
        best_acc = accuracy
        best_k = i

# apply K-NN
print("\nThe best k is: ", best_k)
print("----------------------------------------")
clf = KNeighborsClassifier(n_neighbors=best_k,n_jobs=-1)
clf.fit(X_train,y_train)
print(f'K-NN accuracy before SVD = {clf.score(X_test,y_test)}')