from model_linear import *
from determine_best_model import *
from sklearn.linear_model import TheilSenRegressor, HuberRegressor

import statsmodels.api as sm


print("Determining outliars using Huber method")

# Initializing the model
Huber = HuberRegressor(epsilon=1., max_iter=10000,tol=1e-05)

# training the model
Huber.fit(x_import, y_import)
Huber.score(x_import, y_import)

# inlier mask
inlier_mask = Huber.outliers_
# print(inlier_mask)

# for loop to count
count = 0
index = []
index_i = 0
for i in inlier_mask:
    if i==False:
        index.append(index_i)
        count +=1
    index_i +=1

# printing
print("Total datapoints were : ", len(inlier_mask))
print("Total outliers detected  were : ", count)
print("Outliers:\n",index)

x_import_wo_ransac = x_import[inlier_mask,:]
y_import_wo_ransac = y_import[inlier_mask,:] 
print("Shape of Xinlier:", x_import_wo_ransac.shape)
print("Shape of Yinlier:", y_import_wo_ransac.shape)

# Testing SSE and MSE directly from all data (before cv)
huber_linear_model = linear_model(x_import,y_import)
print("SSE for Huber before CV:", (np.linalg.norm(y_import_wo_ransac-huber_linear_model.predict(x_import_wo_ransac)))**2)
print("MSE for Huber before CV:", mean_squared_error(y_import_wo_ransac, huber_linear_model.predict(x_import_wo_ransac)))


alpha_list = np.linspace(0.001,3,50)
determine_best_model(x_import_wo_ransac,y_import_wo_ransac,200,alpha_list)