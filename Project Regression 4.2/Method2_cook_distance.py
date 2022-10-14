from model_linear import *
from determine_best_model import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot

import statsmodels.api as sm


print("Determining outliars using Cooks Distance")

# Adding collum of ones to x_import
col_ones = np.ones((np.shape(x_import)[0],1))
Xbig_import = np.hstack([col_ones,x_import])
# Xbig_import = sm.add_constant(x_import)

#fit linear regression model from statsmodels
model = sm.OLS(y_import, Xbig_import).fit() 
np.set_printoptions(suppress=True)

#create instance of influence
influence = model.get_influence()
summary_influence = influence.summary_frame()

#obtain Cook's distance for each observation
cooks = influence.cooks_distance

# Indices of points with the distance above a certain threshold I>4/n(/2.2)
threshold_cook = 4/np.shape(x_import)[0]/2.2
indices = [i for i,v in enumerate(cooks[0]) if v > threshold_cook]

print(" There are",np.shape(indices)[0], "outliers:\n",indices)


cooks_filter1 = np.delete(cooks[0],indices,axis=0)
x_import_wo_cook = np.delete(x_import,indices,axis=0)
y_import_wo_cook = np.delete(y_import,indices,axis=0)


cooks_out1 = []
for i in indices:
    cooks_out1.append(cooks[0][i])

# print("Values of Outliars:",cooks_out1)

plt.plot(list(range(np.shape(cooks)[1])), cooks[0])
plt.hlines(threshold_cook,0,np.shape(x_import)[0], color='red')
plt.xlabel('x')
plt.ylabel('Cooks Distance')
plt.show()




alpha_list = np.linspace(0.001,1,500)
determine_best_model(x_import_wo_cook,y_import_wo_cook,500,alpha_list)