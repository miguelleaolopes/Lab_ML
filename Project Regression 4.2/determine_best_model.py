from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *


def determine_best_model(x_imp,y_imp,N_val,alpha_list,Centered=False):
    '''This function tests all models'''
    split_data(x_imp,y_imp,N_val,0.2)

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

    mse_lin = np.mean(mse_list)
    mse_lin_var = np.var(mse_list)
    # plt.scatter(list(range(np.shape(mse_list)[0])),mse_list)
    # plt.title('Squared Error vs Data Index', fontsize=14)
    # plt.xlabel('x')
    # plt.ylabel('Cooks Distance')
    # plt.show()
    print('MSE linear:', mse_lin,"±",mse_lin_var)

    # _________________________________________________________________________________

    print('Calculating best alpha for ridge model ....')
    best_ridalpha_lis = []
    for i in trange(N_val):
        ridcv_model = ridge_modelcv(x_trn[i],y_train[i],alpha_list,None,fit_int=False,solv=False)
        # best_ridalpha_lis.append(ridcv_model[1]['alpha'])
        best_ridalpha_lis.append(ridcv_model[1])

    best_ridalpha = np.mean(best_ridalpha_lis)
    print(best_ridalpha,"±",np.std(best_ridalpha_lis))

    print('Calculating mse for ridge model ....')
    mse_list = []
    for i in trange(N_val):
        rid_model = ridge_model(x_trn[i],y_train[i],best_ridalpha,solver='auto',fit_intercept=True)
        y_pred_rid = rid_model.predict(x_tst[i])
        mse_list.append(mean_squared_error(y_pred_rid,y_test[i]))

    mse_rid = np.mean(mse_list)
    mse_rid_var = np.var(mse_list)
    print('MSE ridge:', mse_rid,"±",mse_rid_var)

    # _________________________________________________________________________________

    print('Calculating best alpha for lasso model ....')
    best_lasalpha_lis = []
    for i in trange(N_val):
        lascv_model = lasso_modelcv(x_trn[i],y_train[i],alpha_list,cv=None,fit_int=False)
        # best_lasalpha_lis.append(lascv_model[1]['alpha'])
        best_lasalpha_lis.append(lascv_model[1])

    best_lasalpha = np.mean(best_lasalpha_lis)
    print(best_lasalpha,"±",np.std(best_lasalpha_lis))

    print('Calculating mse for lasso model ....')
    mse_list = []
    for i in trange(N_val):
        las_model = ridge_model(x_trn[i],y_train[i],best_lasalpha,fit_intercept=True)
        y_pred_las = las_model.predict(x_tst[i])
        mse_list.append(mean_squared_error(y_pred_las,y_test[i]))

    mse_las = np.mean(mse_list)
    mse_las_var = np.var(mse_list)
    # print('MSE lasso:', mse_las,"±",mse_las_var)


    print('MSE linear:', mse_lin,"±",mse_lin_var)
    print('MSE ridge:', mse_rid,"±",mse_rid_var)
    print('MSE lasso:', mse_las,"±",mse_las_var)


# print('This program determines the best model and parameters without clearing training data with outliers')
# determine_best_model(x_import,y_import,500,True)