from random import Random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

global x_train, x_test, y_train, y_test 
x_import = np.load('data/Xtrain_Classification1.npy')
y_import = np.load('data/Ytrain_Classification1.npy')
x_train, x_test, y_train, y_test =  train_test_split(x_import, y_import, test_size=0.2)
x_train, x_test = x_train / 255.0, x_test / 255.0

class RandomForestModel:

    def __init__(self):      
        self.model = RandomForestClassifier()
 

    def fit(self):
        self.model.fit(x_train,y_train)
    
    def predict(self):
        self.y_pred = self.model.predict(x_test)

    def calculate_score(self):
        self.score = cross_val_score(self.model, x_test, y_test, cv=10, scoring='f1')

    def print_scores(self):
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, self.y_pred))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(y_test, self.y_pred))
        print('\n')
        print("=== All AUC Scores ===")
        print(self.score)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", self.score.mean())



class RandomForestModelCV:

    def __init__(self):
        self.n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
        self.max_depth = [int(x) for x in np.linspace(100, 500, num = 5)]
        self.max_depth.append(None) 
        self.random_grid = {'n_estimators': self.n_estimators,'max_depth': self.max_depth}
        self.model = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = self.random_grid,n_iter = 10, cv = 3, verbose=2, n_jobs = -1)
    
    def fit(self):
        self.model.fit(x_train,y_train)
    
    def print_bestparameters(self):
        print('=== Best Parameters ===')
        print(self.model.best_params_)

    def predict(self):
        self.y_pred = self.model.predict(x_test)

    def calculate_score(self):
        self.score = cross_val_score(self.model, x_test, y_test, cv=10, scoring='f1')

    def print_scores(self):
        print("=== Confusion Matrix ===")
        print(confusion_matrix(y_test, self.y_pred))
        print('\n')
        print("=== Classification Report ===")
        print(classification_report(y_test, self.y_pred))
        print('\n')
        print("=== All AUC Scores ===")
        print(self.score)
        print('\n')
        print("=== Mean AUC Score ===")
        print("Mean AUC Score - Random Forest: ", self.score.mean())



mod = RandomForestModelCV()
mod.fit()
mod.print_bestparameters()
mod.predict()
mod.calculate_score()
mod.print_scores()




