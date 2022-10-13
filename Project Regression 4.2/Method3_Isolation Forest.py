from sklearn.ensemble import IsolationForest
from initialization import *
from ridge import *
from lasso import *
from linear import *
from determine_best_model import determine_best_model


model = IsolationForest()
y_pred = model.fit_predict(x_import)

# select all rows that are not outliers
mask = y_pred != -1
x_import_wo, y_import_wo = x_import[mask, :], y_import[mask]

out_list = []

for i in range(len(mask)):
    if not mask[i]:
        out_list.append(i)



n_outliers = np.shape(x_import)[0] - np.shape(x_import_wo)[0]
print('There are',len(out_list),'outliers.\n',out_list)
print(n_outliers)

