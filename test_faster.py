import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from tqdm import trange

X_import = np.load("data/Xtrain_Regression1.npy")
Y_import = np.load("data/Ytrain_Regression1.npy")
# create global functions to store the splitted data
X_train,Y_train,X_test,Y_test = 0,0,0,0

# NValidations = 1000
# TestFraction = 0.2

# Xtrain0_Rows, Xtrain0_Cols = X_import.shape
# Ytrain0_Rows, Ytrain0_Cols = Y_import.shape

# TestDimension  = int(TestFraction * Xtrain0_Rows)
# TrainDimension = Xtrain0_Rows - TestDimension

# print("\nCreating {} validation sets witwh a test dimension of {} ({} %)".format(NValidations, TestDimension, TestFraction*100))
# X_train, Y_train = np.empty((NValidations, TrainDimension, Xtrain0_Cols)), np.empty((NValidations, TrainDimension, Ytrain0_Cols))
# X_test , Y_test  = np.empty((NValidations, TestDimension , Xtrain0_Cols)), np.empty((NValidations, TestDimension , Ytrain0_Cols))

def calc_SSE(y_pred,y):
    '''Calculates the SSE from the predicted data'''
    #return np.sum((y_pred - y)**2)
    return mean_squared_error(y_pred,y)*np.shape(y)[0]

NValidations = 1000
alphalist = np.logspace(start=-3, stop=-1, num=100)
model = RidgeCV(alphas=alphalist)

TestAlpha = []
TestScore = []
TestMSE   = []
TestSSE   = []

# for i in range(NValidations):
#     x1,x2,x3,x4 = train_test_split(X_import, Y_import, test_size=0.2)
#     X_train.append(x1), X_test.append(x2), Y_train.append(x3), Y_test.append(x4)


for i in trange(NValidations):

    X_train, X_test, Y_train, Y_test = train_test_split(X_import, Y_import, test_size=0.2)
    
    model.fit(X_train, Y_train)

    TestAlpha.append(model.alpha_)
    TestScore.append(model.best_score_)
    TestMSE.append(mean_squared_error(Y_test,model.predict(X_test)))
    TestSSE.append(calc_SSE(model.predict(X_test),Y_test))

print("α   =", np.mean(TestAlpha), "±", np.std(TestAlpha))
print("R²  =", np.mean(TestScore), "±", np.std(TestScore))
print("MSE =", np.mean(TestMSE), "±", np.std(TestMSE))
print("SSE =", np.mean(TestSSE), "±", np.std(TestSSE))