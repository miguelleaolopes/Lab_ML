from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

global x_train, x_test, y_train, y_test 
x_import = np.load('data/Xtrain_Classification1.npy')
y_import = np.load('data/Ytrain_Classification1.npy')
x_train, x_test, y_train, y_test =  train_test_split(x_import, y_import, test_size=0.2)
x_train, x_test = x_train / 255.0, x_test / 255.0



class svm_model:

    def __init__(self,kernel,gamma='auto'):
        self.model = svm.SVC(kernel=kernel,gamma=gamma)
    

    def fit(self):
        self.model.fit(x_train, y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(x_test)

    def calculate_metrics(self):
        self.acc = metrics.accuracy_score(y_test, self.y_pred)
        self.f1 = metrics.f1_score(y_test, self.y_pred)
        

gammas = np.linspace(0.0001,.01,5)

for gamma  in gammas:
    # model = svm_model(kernel='poly')
    # model = svm_model(kernel='linear')
    model = svm_model(kernel='rbf',gamma=gamma)

    model.fit()
    model.predict()
    model.calculate_metrics()

    print('=== Gamma:', gamma)
    print('=== Score F1:',model.f1)
    print('== Score ACC:',model.acc)


    