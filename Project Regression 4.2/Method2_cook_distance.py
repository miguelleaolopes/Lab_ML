from initialization import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm


class method2_cooks_distance:

    def __init__(self,silent=False,show_plt=False):
        self.silent = silent
        self.show_plt = show_plt
        self.outliers_removed = False

    def remove_outliers(self):
        print("Method 2: Remove outliers with highest cook's distance")

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

        if self.silent: print("Values of Outliars:\n",self.out_vals)

        if self.show_plt:
            plt.plot(list(range(np.shape(cooks)[1])), cooks[0])
            plt.hlines(self.threshold,0,np.shape(x_import)[0], color='red')
            plt.xlabel('x')
            plt.ylabel('Cooks Distance')
            plt.show()
        
        self.outliers_removed = True
    
    def test_method(self):
        
        if self.outliers_removed:
            results = determine_best_model(self.x_import_wo, self.y_import_wo,500,np.linspace(0.001,1,500))
        else: print('Outliers not removed, please remove outliers first with self.remove_outliers()!')

