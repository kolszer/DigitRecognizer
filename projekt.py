
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neural_network, linear_model, svm, tree, cluster, naive_bayes, neighbors, ensemble
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score


# In[2]:

#Import csv z danymi treningowymi
train = pd.read_csv("train.csv")


# In[3]:

#X- Kolumny zawierajace wartosci pixeli
X = train[train.columns[1:]].values
#y- Kolumna zawierajaca etykiety
y = train[train.columns[0]].values


# In[4]:

#Zamiana wartości pixeli z 0-255 na 0-7 dla danych treningowych(w celu poprawy wydajnosci)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i][j] < 32:
            X[i][j] = 0
        elif X[i][j] >= 32 and X[i][j] < 64:
            X[i][j] = 1
        elif X[i][j] >= 64 and X[i][j] < 96:
            X[i][j] = 2
        elif X[i][j] >= 96 and X[i][j] < 128:
            X[i][j] = 3
        elif X[i][j] >= 128 and X[i][j] < 160:
            X[i][j] = 4
        elif X[i][j] >= 160 and X[i][j] < 192:
            X[i][j] = 5
        elif X[i][j] >= 192 and X[i][j] < 224:
            X[i][j] = 6
        elif X[i][j] >= 224:
            X[i][j] = 7


# In[5]:

#Wyswietlenie cyfry z 22 wiersza
x = train.values[22,1:]
x = np.reshape(x, (28, 28))
plt.imshow(x)
plt.show()
print (train.values[22,0])


# In[6]:

#Funkcja train_test_split() losuje, dzieli i zwraca zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[7]:

model = neural_network.MLPClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[8]:

model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[9]:

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[10]:

model = svm.LinearSVC()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[11]:

model = linear_model.SGDClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[12]:

model = naive_bayes.BernoulliNB()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[13]:

model = cluster.KMeans(n_clusters=10)
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[14]:

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[15]:

model = ensemble.ExtraTreesClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[16]:

model = ensemble.AdaBoostClassifier()
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[17]:

#Dopracowanie najlepszego klasyfikatora
model = neural_network.MLPClassifier(max_iter=100000, hidden_layer_sizes=(1000, ))
model.fit(X_train, y_train)
test = model.predict(X_test)
print (confusion_matrix(y_test, test))
print (accuracy_score(y_test, test))
print (classification_report(y_test, test))
print (precision_score(y_test, test,average=None))


# In[18]:

#Import danych testowych
test = pd.read_csv("test.csv")


# In[19]:

#Zamiana wartości pixeli z 0-255 na 0-7 dla danych testowych(w celu poprawy wydajnosci)
X_test = test.values
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if X_test[i][j] < 32:
            X_test[i][j] = 0
        elif X_test[i][j] >= 32 and X_test[i][j] < 64:
            X_test[i][j] = 1
        elif X_test[i][j] >= 64 and X_test[i][j] < 96:
            X_test[i][j] = 2
        elif X_test[i][j] >= 96 and X_test[i][j] < 128:
            X_test[i][j] = 3
        elif X_test[i][j] >= 128 and X_test[i][j] < 160:
            X_test[i][j] = 4
        elif X_test[i][j] >= 160 and X_test[i][j] < 192:
            X_test[i][j] = 5
        elif X_test[i][j] >= 192 and X_test[i][j] < 224:
            X_test[i][j] = 6
        elif X_test[i][j] >= 224:
            X_test[i][j] = 7


# In[20]:

#Stworzenie modelu sieci neuronowej z maksymalna liczba iteracji 100 000 i z 1 000 ukrytych warstw
model = neural_network.MLPClassifier(max_iter=100000, hidden_layer_sizes=(1000, ))
#Uczenie sieci neuronowej
model.fit(X, y)


# In[21]:

#Klasyfikacja danych testowych
test = model.predict(X_test)


# In[22]:

#Zapis do CSV
dfTest = pd.DataFrame(test)
dfTest.index += 1
dfTest.to_csv("testPredict.csv")

