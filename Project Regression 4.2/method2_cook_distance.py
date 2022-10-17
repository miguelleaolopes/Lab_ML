from initialization import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm


class method2_cooks_distance:

    def __init__(self,silent=True,show_plt=False,N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.show_plt = show_plt
        self.outliers_removed = False
        self.N_val = N_val
        self.alpha_list = alpha_list

    def remove_outliers(self):
        print("\n\n\nMethod 2: Remove outliers with highest cook's distance at once")

        col_ones = np.ones((np.shape(x_import)[0],1))
        Xbig_import = np.hstack([col_ones,x_import])

        model = sm.OLS(y_import, Xbig_import).fit() 
        np.set_printoptions(suppress=True)

        influence = model.get_influence()
        summary_influence = influence.summary_frame()
        cooks = influence.cooks_distance

        self.threshold = 4/np.shape(x_import)[0]/2.2
        self.out_list = [i for i,v in enumerate(cooks[0]) if v > self.threshold]

        print(" There are",len(self.out_list), "outliers:\n",self.out_list)

        cooks_filter1 = np.delete(cooks[0],self.out_list,axis=0)
        self.x_import_wo = np.delete(x_import,self.out_list,axis=0)
        self.y_import_wo = np.delete(y_import,self.out_list,axis=0)


        self.out_vals = []
        for i in self.out_list:
            self.out_vals.append(cooks[0][i])

        if not self.silent: print("Values of Outliars:\n",self.out_vals)

        if self.show_plt:
            plt.plot(list(range(np.shape(cooks)[1])), cooks[0])
            plt.hlines(self.threshold,0,np.shape(x_import)[0], color='red')
            plt.xlabel('x')
            plt.ylabel('Cooks Distance')
            plt.show()
        
        self.outliers_removed = True

    def remove_outliers_cyclical(self):
        print("\n\n\nMethod 2.2: Remove outliers with highest cook's distance on at the time")

        self.thd_passed = False
        self.out_list = []
        self.x_import_wo, self.y_import_wo = x_import.copy(), y_import.copy()

        while not self.thd_passed and len(self.out_list) < 20:
            col_ones = np.ones((np.shape(self.x_import_wo)[0],1))
            Xbig_import = np.hstack([col_ones,self.x_import_wo])
            model = sm.OLS(self.y_import_wo, Xbig_import).fit() 
            np.set_printoptions(suppress=True)

            influence = model.get_influence()
            summary_influence = influence.summary_frame()
            cooks = influence.cooks_distance

            outlier = np.amax(cooks[0])
            outlier_index = np.where(cooks[0] == outlier)[0][0]

            self.threshold = 4/np.shape(self.x_import_wo)[0]



            if cooks[0][outlier_index] > self.threshold:
                if not self.silent: print('Outlier removed:',outlier_index)
                self.out_list.append(np.where(x_import == self.x_import_wo[outlier_index])[0][0])
                self.x_import_wo = np.delete(self.x_import_wo,outlier_index,axis=0)
                self.y_import_wo = np.delete(self.y_import_wo,outlier_index,axis=0)
            else:
                self.thd_threshold = True

        if self.thd_passed: print('Threshold reached')
        else: print('Outlier limit reached')

        self.out_list = np.sort(self.out_list)
        print(len(self.out_list),'outliers found:\n',self.out_list)

        self.outliers_removed = True

    def test_method(self):
        
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index, self.mse_mean = determine_best_model(self.x_import_wo, self.y_import_wo,self.N_val,self.alpha_list)
        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')

