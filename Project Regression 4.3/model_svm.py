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



class linear_svm:

    def __init__(self):
        self.model = svm.SVC(kernel='linear')
    

    def fit(self):
        self.model.fit(x_train, y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(x_test)

    def calculate_score(self):
        self.acc = metrics.accuracy_score(y_test, self.y_pred)


mod = linear_svm()
mod.fit()
mod.predict()
mod.calculate_score()
print(mod.acc)


    