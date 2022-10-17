from initialization import *
from method1_se_removal import *
from method2_cook_distance import *
from method3_isolation_forest import *
from method4_ransac import *
from method5_huber_theil import *
from method6_dffits import *

mse_list, min_mse = [], []
methods = ['method 1','method 1.1','method 2','method 2.1','method 3','method 4','method 5','method 6']
models = ['linear', 'ridge', 'lasso']

alpha_list = np.linspace(0.001,1,300)
N_validation = 1000

m1 = method1_se_removal(alpha_list=alpha_list,N_val=N_validation)
m1.remove_outliers()
m1.test_method()

m1_1 = method1_se_removal(alpha_list=alpha_list,N_val=N_validation)
m1_1.remove_outliers_cyclical()
m1_1.test_method()

m2 = method2_cooks_distance(alpha_list=alpha_list,N_val=N_validation)
m2.remove_outliers()
m2.test_method()

m2_1 = method2_cooks_distance(alpha_list=alpha_list,N_val=N_validation)
m2_1.remove_outliers_cyclical()
m2_1.test_method()

m3 = method3_isolation_forest(alpha_list=alpha_list,N_val=N_validation)
m3.remove_outliers()
m3.test_method()

m4 = method4_ransac(alpha_list=alpha_list,N_val=N_validation)
m4.remove_outliers()
m4.test_method()

m5 = method5_huber_theil(alpha_list=alpha_list,N_val=N_validation)
m5.remove_outliers()
m5.test_method()

m6 = method6_dffits(alpha_list=alpha_list,N_val=N_validation)
m6.remove_outliers_cyclical()
m6.test_method()

methodson, all_outliers = [m1,m1_1,m2,m2_1,m3,m4,m5,m6], []
for mth in methodson:
    mse_list.append(mth.mse_mean)

for mth in methodson:
    all_outliers.append(mth.out_list)


lowest_mse = np.min(mse_list)
lowest_location = np.where(mse_list == lowest_mse)
outliers = all_outliers[lowest_location[0][0]]
print('The best method is',methods[lowest_location[0][0]],'with',models[lowest_location[1][0]],'model with',len(outliers),'outliers')
print('Outliers:', outliers)


x_import_wo = np.delete(x_import, outliers, axis=0)
y_import_wo = np.delete(y_import, outliers, axis=0)

print('Determining best model considering the previous outliers')
models, best_alphas, best_index, mse_mean = determine_best_model(x_import_wo,y_import_wo,10000,np.linspace(0.001,0.1,600),Centered=False)
print('The best model is',models[best_index],'with and alpha of',best_alphas[best_index])



