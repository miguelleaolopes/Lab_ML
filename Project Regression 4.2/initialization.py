# Import Libraries
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from tqdm import tqdm


# Import variables
x_import = np.load("data/Xtrain_Regression2.npy")
y_import = np.load("data/Ytrain_Regression2.npy")