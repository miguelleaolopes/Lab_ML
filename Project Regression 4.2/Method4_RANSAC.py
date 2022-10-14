from model_linear import *
from determine_best_model import *
from sklearn.linear_model import RANSACRegressor

import statsmodels.api as sm


print("Determining outliars using RANSAC method")

# Initializing the model
# Ransac = RANSACRegressor()
Ransac = RANSACRegressor(min_samples=(x_import.shape[1]+1), max_trials=10000000, loss='squared_error') #random_state=42, residual_threshold=10

# training the model
Ransac.fit(x_import, y_import)
Ransac.score(x_import, y_import)

# inlier mask
inlier_mask = Ransac.inlier_mask_
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
print("SSE for RANSAC before CV:", (np.linalg.norm(y_import_wo_ransac-Ransac.predict(x_import_wo_ransac)))**2)
print("MSE for RANSAC before CV:", mean_squared_error(y_import_wo_ransac, Ransac.predict(x_import_wo_ransac)))


alpha_list = np.linspace(0.001,3,50)
determine_best_model(x_import_wo_ransac,y_import_wo_ransac,200,alpha_list)