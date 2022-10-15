from sklearn.ensemble import IsolationForest
from initialization import *


class method3_isolation_forest:

    def __init__(self,silent=True,N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.outliers_removed = False
        self.N_val = N_val
        self.alpha_list = alpha_list

    def remove_outliers(self):
        print("\n\n\nMethod 3: Remove outliers with isolation forest")
        model = IsolationForest()
        y_pred = model.fit_predict(x_import)
        mask = y_pred != -1 # select all rows that are not outliers
        self.x_import_wo, self.y_import_wo = x_import[mask, :], y_import[mask]
        self.out_list = []

        for i in range(len(mask)):
            if not mask[i]:
                self.out_list.append(i)

        self.out_list = np.sort(self.out_list)
        print('There are',len(self.out_list),'outliers.\n',self.out_list)
        
        self.outliers_removed = True

    def test_method(self):
    
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index = determine_best_model(self.x_import_wo, self.y_import_wo,self.N_val,self.alpha_list)
        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')