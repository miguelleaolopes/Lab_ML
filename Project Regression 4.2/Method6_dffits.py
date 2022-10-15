from initialization import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm


class method6_dffits:

    def __init__(self,silent=True,show_plt=False, thd_type='default',N_val=200,alpha_list = np.linspace(0.001,2,100)):
        self.silent = silent
        self.show_plt = show_plt
        self.outliers_removed = False
        self.thd_type = thd_type
        self.N_val = N_val
        self.alpha_list = alpha_list

    def remove_outliers_cyclical(self):
        print('\n\n\nMethod 6.1: Removing outliers with highest dffits one at the time')

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
            dffits = influence.dffits

            outlier = np.amax(np.abs(dffits[0]))
            outlier_index = np.where(np.abs(dffits[0]) == outlier)[0][0]

            if self.thd_type == 'default':
                self.threshold = dffits[1]

            elif self.thd_type == 'theoretical':
                self.threshold = 2*np.sqrt(np.shape(Xbig_import)[1])/(np.shape(x_import_wo)[0] - np.shape(Xbig_import)[1])
            

            if self.show_plt():
                plt.plot(list(range(np.shape(dffits[0])[0])), dffits[0])
                plt.hlines(self.threshold,0,np.shape(x_import)[0], color='red')
                plt.hlines(-1*self.threshold,0,np.shape(x_import)[0], color='red')
                plt.xlabel('x')
                plt.ylabel('dffits')
                plt.show()


            if np.abs(dffits[0][outlier_index]) > self.threshold:
                if not self.silent: print('Outlier removed:',outlier_index)
                out_list.append(np.where(x_import == self.x_import_wo[outlier_index])[0][0])
                self.x_import_wo = np.delete(self.x_import_wo,outlier_index,axis=0)
                self.out_listy_import_wo = np.delete(self.y_import_wo,outlier_index,axis=0)
            else:
                self.thd_passed = True

        out_list = np.sort(out_list)
        print(len(out_list),'outliers found:\n',out_list)
        self.outliers_removed = True
    
    def test_method(self):
        if self.outliers_removed:
            self.models, self.best_alphas, self.best_index = determine_best_model(self.x_import_wo, self.y_import_wo,self.N_val,self.alpha_list)


