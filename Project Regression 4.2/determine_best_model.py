from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *

def determine_best_model(x_imp,y_imp,N_val):

    split_data(x_imp,y_imp,N_val,0.2)

    print('Calculating mse for linear model ....')
    for i in trange(N_val):
        mse_list = []
        lin_model = linear_model(x_train[i],y_train[i])
        y_pred_lin = lin_model.predict(x_test[i])
        mse_list.append(mean_squared_error(y_pred_lin,y_test[i]))

    mse_lin = np.mean(mse_list)

    print('Calculating best alpha for ridge model ....')
    best_ridalpha_lis = []
    for i in trange(N_val):
        ridcv_model = ridge_modelcv(x_train[i],y_train[i],np.linspace(0.001,0.1,20),None,fit_int=False,solv=False)
        best_ridalpha_lis.append(ridcv_model[1]['alpha'])

    best_ridalpha = np.mean(best_ridalpha_lis)
    print(best_ridalpha,"±",np.std(best_ridalpha_lis))

    print('Calculating mse for ridge model ....')
    for i in trange(N_val):
        mse_list = []
        rid_model = ridge_model(x_train[i],y_train[i],best_ridalpha,solver='auto',fit_intercept=True)
        y_pred_rid = rid_model.predict(x_test[i])
        mse_list.append(mean_squared_error(y_pred_lin,y_test[i]))

    mse_rid = np.mean(mse_list)

    print('Calculating best alpha for lasso model ....')
    best_lasalpha_lis = []
    for i in trange(N_val):
        lascv_model = lasso_modelcv(x_train[i],y_train[i],np.linspace(0.001,0.1,20),cv=None,fit_int=False)
        best_lasalpha_lis.append(lascv_model[1]['alpha'])

    best_lasalpha = np.mean(best_lasalpha_lis)
    print(best_lasalpha,"±",np.std(best_lasalpha_lis))

    print('Calculating mse for lasso model ....')
    for i in trange(N_val):
        mse_list = []
        las_model = ridge_model(x_train[i],y_train[i],best_lasalpha,fit_intercept=True)
        y_pred_las = las_model.predict(x_test[i])
        mse_list.append(mean_squared_error(y_pred_las,y_test[i]))

    mse_las = np.mean(mse_list)


    print('MSE linear:', mse_lin)
    print('MSE ridge:', mse_rid)
    print('MSE lasso:', mse_las)

# print('This program determines the best model and parameters without clearing training data with outliers')
# determine_best_model(x_import,y_import,500)