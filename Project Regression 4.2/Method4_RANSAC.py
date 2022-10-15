from initialization import *
from sklearn.linear_model import RANSACRegressor
import statsmodels.api as sm



class method4_ransac:

    def __init__(self,silent=True,N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.outliers_removed = False
        self.N_val = N_val
        self.alpha_list = alpha_list
    
    def remove_outliers(self):
        print("\n\n\nMethod 4: Remove outliers with RANSAC method")

        # Initializing the model
        # Ransac = RANSACRegressor()
        Ransac = RANSACRegressor(min_samples=(x_import.shape[1]+1), max_trials=10000000, loss='squared_error') #random_state=42, residual_threshold=10

        # training the model
        Ransac.fit(x_import, y_import)
        Ransac.score(x_import, y_import)

        self.out_list = []
        self.inlier_mask = Ransac.inlier_mask_
        # print(inlier_mask)



        for i in range(len(self.inlier_mask)):
            if self.inlier_mask[i]==False: self.out_list.append(i)
                


        print("Total datapoints were : ", len(self.inlier_mask))
        print('There are',len(self.out_list),'outliers.\n',self.out_list)
        

        self.x_import_wo = x_import[self.inlier_mask,:]
        self.y_import_wo = y_import[self.inlier_mask,:]

        if not self.silent:
            print("Shape of Xinlier:", self.x_import_wo.shape)
            print("Shape of Yinlier:", self.y_import_wo.shape)

            print("SSE for RANSAC before CV:", (np.linalg.norm(self.y_import_wo - Ransac.predict(self.x_import_wo)))**2)
            print("MSE for RANSAC before CV:", mean_squared_error(self.y_import_wo, Ransac.predict(self.x_import_wo)))

        self.outliers_removed = True

    def test_method(self):
    
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index = determine_best_model(self.x_import_wo, self.y_import_wo,self.N_val,self.alpha_list)
        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')