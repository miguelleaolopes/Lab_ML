from initialization import *
from sklearn.linear_model import TheilSenRegressor
from sklearn.datasets import make_regression



N_val = 200
x_train,y_train,x_test,y_test, x_train_s, x_test_s = split_data(x_import,y_import,N_val,0.2)

mse_list=[]
for i in trange(N_val):
    reg = TheilSenRegressor().fit(x_train[i], np.ravel(y_train[i]))
    y_pred = reg.predict(x_test[i])
    mse_list.append(mean_squared_error(y_pred,y_test[i]))

print(np.mean(mse_list))