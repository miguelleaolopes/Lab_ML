from initialization import *


class method1_1_cyclical_se_removal:

    def __init__(self,verbose=False):
        self.verbose = verbose
        self.outliers_removed = False

    def remove_outliers_cyclical(self):
        print('Method 1.1: Removing outliers with highest standard error one by one')
        self.thd_passed = False
        self.out_list = []
        self.x_import_wo, self.y_import_wo = x_import.copy(), y_import.copy()

        while not self.thd_passed and len(self.out_list) < 20:

            modelwout = linear_model(self.x_import_wo,self.y_import_wo)
            y_pred = modelwout.predict(self.x_import_wo)
            SE_list = []

            for i in range(len(self.x_import_wo)):
                SE_list = np.hstack([SE_list,(self.y_import_wo[i]-y_pred[i])**2])
            
            threshold = np.mean(SE_list) + 1*np.std(SE_list)
            outlier = np.amax(SE_list)
            outlier_index = np.where(SE_list == outlier)[0][0]

        
            if SE_list[outlier_index] > threshold:
                if self.verbose: print('Outlier removed:',outlier_index)
                self.out_list.append(np.where(x_import == self.x_import_wo[outlier_index])[0][0])
                self.x_import_wo = np.delete(self.x_import_wo,outlier_index,axis=0)
                self.y_import_wo = np.delete(self.y_import_wo,outlier_index,axis=0)
            else:
                self.thd_passed = True

        if self.thd_passed: print('Threshold reached')
        else: print('Outlier limit reached')

        # self.out_list = np.sort(self.out_list)
        print(len(self.out_list),'outliers found:\n',self.out_list)

        self.outliers_removed = True

    def test_method(self):
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index = determine_best_model(self.x_import_wo,self.y_import_wo,500,np.linspace(0.001,2,50))

        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')
