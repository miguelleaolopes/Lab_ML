# Import Libraries
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures,  StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from tqdm import trange
from model_ridge import *
from model_lasso import *
from model_linear import *


# Import variables
x_import = np.load("data/Xtrain_Regression2.npy")
y_import = np.load("data/Ytrain_Regression2.npy")
x_train,y_train,x_test,y_test, x_train_s, x_test_s = [], [], [], [], [], []

# Essential functions
def split_data(x,y,N_list,test_size):

    for i in range(N_list):
        result = train_test_split(x, y, test_size=test_size)
        x_train.append(result[0])
        x_test.append(result[1])
        y_train.append(result[2])
        y_test.append(result[3])

        x_train_s.append(StandardScaler().fit_transform(result[0]))
        x_test_s.append(StandardScaler().fit_transform(result[1]))

def determine_best_model(x_imp,y_imp,N_val,alpha_list,Centered=False):
    '''This function tests all models'''
    split_data(x_imp,y_imp,N_val,0.2)
    models, mse_mean, mse_var, best_alphas = ['Linear', 'Ridge', 'Lasso'], [], [], []

    if not Centered:
        print('Not centered features')
        x_trn = x_train
        x_tst = x_test
    else:
        print('Centered features')
        x_trn = x_train_s
        x_tst = x_test_s

    
    # _________________________________________________________________________________

    print('Calculating mse for linear model ....')
    mse_list = []
    for i in trange(N_val):
        lin_model = linear_model(x_train[i],y_train[i]) #For linear we do not center
        y_pred_lin = lin_model.predict(x_test[i])
        mse_list.append(mean_squared_error(y_pred_lin,y_test[i]))

    mse_mean.append(np.mean(mse_list))
    mse_var.append(np.var(mse_list))
    best_alphas.append(0) #Just to maintain the index consistent

    # _________________________________________________________________________________

    print('Calculating best alpha for ridge model ....')
    best_ridalpha_lis = []
    for i in trange(N_val):
        ridcv_model = ridge_modelcv(x_trn[i],y_train[i],alpha_list,None,fit_int=False,solv=False)
        # best_ridalpha_lis.append(ridcv_model[1]['alpha'])
        best_ridalpha_lis.append(ridcv_model[1])

    best_alphas.append(np.mean(best_ridalpha_lis))
    print(best_alphas[1],"±",np.std(best_ridalpha_lis))

    print('Calculating mse for ridge model ....')
    mse_list = []
    for i in trange(N_val):
        rid_model = ridge_model(x_trn[i],y_train[i],best_alphas[1],solver='auto',fit_intercept=True)
        y_pred_rid = rid_model.predict(x_tst[i])
        mse_list.append(mean_squared_error(y_pred_rid,y_test[i]))

    mse_mean.append(np.mean(mse_list))
    mse_var.append(np.var(mse_list))

    # _________________________________________________________________________________

    print('Calculating best alpha for lasso model ....')
    best_lasalpha_lis = []
    for i in trange(N_val):
        lascv_model = lasso_modelcv(x_trn[i],y_train[i],alpha_list,cv=None,fit_int=False)
        # best_lasalpha_lis.append(lascv_model[1]['alpha'])
        best_lasalpha_lis.append(lascv_model[1])

    best_alphas.append(np.mean(best_lasalpha_lis))
    print(best_alphas[2],"±",np.std(best_lasalpha_lis))

    print('Calculating mse for lasso model ....')
    mse_list = []
    for i in trange(N_val):
        las_model = ridge_model(x_trn[i],y_train[i],best_alphas[2],fit_intercept=True)
        y_pred_las = las_model.predict(x_tst[i])
        mse_list.append(mean_squared_error(y_pred_las,y_test[i]))

    mse_mean.append(np.mean(mse_list))
    mse_var.append(np.var(mse_list))

    for i in range(len(models)):
        print('MSE',models[i],':', mse_mean[i],"±",mse_var[i])
    
    best_mse = np.min(mse_mean)
    best_index = np.where(mse_mean == best_mse)[0][0]
    print('##### best model is',models[best_index],'#####')

    return models, best_alphas, best_index   
