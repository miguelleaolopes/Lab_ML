from initialization import *


class method1_se_removal:

    def __init__(self,silent=True,show_plt=False,N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.show_plt = show_plt
        self.outliers_removed = False
        self.N_val = N_val
        self.alpha_list = alpha_list

    def remove_outliers(self):
        print('\n\n\nMethod 1: Removing outliers with highest standard error all at once')
        modelwout = linear_model(x_import,y_import)
        SE_list, self.out_list = [], []
        index_list = [i for i in range(np.shape(x_import)[0])]
        y_pred = modelwout.predict(x_import)

        for i in range(len(x_import)):
            SE_list.append((y_import[i]-y_pred[i])**2)

        self.sigma = np.std(SE_list)
        self.threshold = np.mean(SE_list) + 0.1*self.sigma

        if self.show_plt:
            plt.plot(self.index_list, SE_list, color='blue', marker='o')
            plt.hlines(self.threshold,self.index_list[0],index_list[-1], color='red')
            plt.title('Squared Error vs Data Index', fontsize=14)
            plt.ylabel('SE', fontsize=14)
            plt.xlabel('Data Index', fontsize=14)
            plt.grid(True)
            plt.show()


        for i in range(len(x_import)):
            if SE_list[i] > self.threshold:
                self.out_list.append(i)

        print('There are',len(self.out_list),' outliers :\n',self.out_list)

        self.x_import_wo = np.delete(x_import, self.out_list, axis=0)
        self.y_import_wo = np.delete(y_import, self.out_list, axis=0)
        self.outliers_removed = True


    def remove_outliers_cyclical(self):
        print('\n\n\nMethod 1.1: Removing outliers with highest standard error one by one')
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
                if not self.silent: print('Outlier removed:',outlier_index)
                self.out_list.append(np.where(x_import == self.x_import_wo[outlier_index])[0][0])
                self.x_import_wo = np.delete(self.x_import_wo,outlier_index,axis=0)
                self.y_import_wo = np.delete(self.y_import_wo,outlier_index,axis=0)
            else:
                self.thd_passed = True

        if self.thd_passed: print('Threshold reached')
        else: print('Outlier limit reached')

        self.out_list = np.sort(self.out_list)
        print(len(self.out_list),'outliers found:\n',self.out_list)

        self.outliers_removed = True


    
    def test_method(self):
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index, self.mse_mean = determine_best_model(self.x_import_wo, self.y_import_wo, self.N_val, self.alpha_list)

        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')


