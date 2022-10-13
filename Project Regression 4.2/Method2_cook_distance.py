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

print(" There are",np.shape(indices), "outliars:")
print("Indices: ",indices)

cooks_filter1 = np.delete(cooks[0],indices,axis=0)
x_import_wo_cook = np.delete(x_import,indices,axis=0)
y_import_wo_cook = np.delete(y_import,indices,axis=0)

# print(np.shape(x_import_wo_cook))
# print(np.shape(y_import_wo_cook))

cooks_out1 = []
for i in indices:
    cooks_out1.append(cooks[0][i])

print("Values of Outliars:",cooks_out1)

plt.plot(list(range(np.shape(cooks)[1])), cooks[0])
plt.hlines(threshold_cook,0,np.shape(x_import)[0], color='red')
plt.xlabel('x')
plt.ylabel('Cooks Distance')
plt.show()


#######################################################
# --- Trying to implement the method from the root ---
#######################################################

# p_size_features = np.shape(x_import)[1]
# n_observations = np.shape(x_import)[0]

# class CooksDistance(Visualizer):
    
#     def fit(self, X, y):
#         # Leverage is computed as the diagonal of the projection matrix of X 
#         # TODO: whiten X before computing leverage
#         self.leverage_ = (X * np.linalg.pinv(X).T).sum(1)
        
#         # Compute the MSE
#         rank = np.linalg.matrix_rank(X)
#         df = X.shape[0] - rank
        
#         resid = y - LinearRegression().fit(X, y).predict(X)
#         mse = np.dot(resid, np.transpose(resid)) / df 
        
#         resid_studentized_internal = resid / np.sqrt(mse) / np.sqrt(1-self.leverage_)
        
#         self.distance_ = resid_studentized_internal**2 / X.shape[1]
#         self.distance_ *= self.leverage_ / (1 - self.leverage_)

#         self.p_values_ = sp.stats.f.sf(self.distance_, X.shape[1], df)
        
#         self.draw()
#         return self

# viz = CooksDistance().fit(x_import, y_import[:,0])
# viz.show()
# viz.finalize()

################################################

# cook_linear_model = linear_model(x_import,y_import)
# influence = cook_linear_model

# visualizer = CooksDistance()
# visualizer.fit(Xbig_import, y_import[:,0])
# visualizer.show()

# models = LinearRegression()
# visualizer_residuals = ResidualsPlot(models,qqplot=True, hist=False,)
# visualizer_residuals.fit(x_import, y_import[:,0])
# visualizer_residuals.show()

alpha_list = np.linspace(0.001,1,50)
determine_best_model(x_import_wo_cook,y_import_wo_cook,500)