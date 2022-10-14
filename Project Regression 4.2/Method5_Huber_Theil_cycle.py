from model_linear import *
from determine_best_model import *
from sklearn.linear_model import TheilSenRegressor, HuberRegressor

import statsmodels.api as sm
from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *

print("Determining outliars using Huber method")

alpha_list = np.linspace(0.001,3,50)
N_val = 200
MSE_l = []
eps_val=[]
for i in range(10):
    # Initializing the model
    Huber = HuberRegressor(epsilon=1.1+0.1*i, max_iter=10000,tol=1e-05)
    eps_val.append(1.1+0.1*i)
    print(eps_val[i])
    # training the model
    Huber.fit(x_import, y_import)
    Huber.score(x_import, y_import)

    # inlier mask
    outlier_mask = Huber.outliers_
    inlier_mask = np.where(outlier_mask == False, True, False)
    # print(inlier_mask)

    # for loop to count
    count = 0
    index = []
    index_i = 0
    for j in inlier_mask:
        if j==False:
            index.append(index_i)
            count +=1
        index_i +=1

    # printing
    print("Total datapoints were : ", len(inlier_mask))
    print("Total outliers detected  were : ", count)
    print("Outliers:\n",index)

    x_import_wo_ransac = x_import[inlier_mask,:]
    y_import_wo_ransac = y_import[inlier_mask,:] 
    # print("Shape of Xinlier:", x_import_wo_ransac.shape)
    # print("Shape of Yinlier:", y_import_wo_ransac.shape)

    split_data(x_import_wo_ransac,y_import_wo_ransac,N_val,0.2)

    print('Calculating mse for linear model ....')
    mse_list = []
    for k in range(N_val):
        lin_model = linear_model(x_train[k],y_train[k]) #For linear we do not center
        y_pred_lin = lin_model.predict(x_test[k])
        mse_list.append(mean_squared_error(y_pred_lin,y_test[k]))

    mse_lin = np.mean(mse_list)
    mse_lin_var = np.var(mse_list)
    print('MSE linear:', mse_lin,"±",mse_lin_var)
    MSE_l.append(mse_lin)


print(MSE_l)
print(eps_val)