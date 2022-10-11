# Import Libraries
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from tqdm import trange

# Import variables
x_import = np.load("data/Xtrain_Regression2.npy")
y_import = np.load("data/Ytrain_Regression2.npy")
x_train,y_train,x_test,y_test, x_train_s, x_test_s = [], [], [], [], [], []

# Essential functions

def split_data(N_list,test_size):

    for i in range(N_list):
        result = train_test_split(x_import, x_import, test_size=test_size)
        x_train.append(result[0])
        x_test.append(result[1])
        y_train.append(result[2])
        y_test.append(result[3])

        x_train_s.append(StandardScaler().fit_transform(result[0]))
        x_test_s.append(StandardScaler().fit_transform(result[1]))

        x_train_s.append(StandardScaler().fit_transform(result[0]))
        x_test_s.append(StandardScaler().fit_transform(result[1]))