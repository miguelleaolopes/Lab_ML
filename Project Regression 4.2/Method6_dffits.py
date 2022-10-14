from model_linear import *
from determine_best_model import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot

import statsmodels.api as sm


print("Determining outliars using cyclical Cooks Distance")


Test_passed = False
out_list = []
x_import_wo, y_import_wo = x_import.copy(), y_import.copy()

while not Test_passed and len(out_list) < 20:
    col_ones = np.ones((np.shape(x_import_wo)[0],1))
    Xbig_import = np.hstack([col_ones,x_import_wo])
    model = sm.OLS(y_import_wo, Xbig_import).fit() 
    np.set_printoptions(suppress=True)

    influence = model.get_influence()
    summary_influence = influence.summary_frame()
    dffits = influence.dffits

    outlier = np.amax(np.abs(dffits[0]))
    outlier_index = np.where(np.abs(dffits[0]) == outlier)[0][0]

    #threshold = 2*np.sqrt(np.shape(Xbig_import)[1])/(np.shape(x_import_wo)[0] - np.shape(Xbig_import)[1])
    threshold = dffits[1]
    
    # plt.plot(list(range(np.shape(dffits[0])[0])), dffits[0])
    # plt.hlines(threshold,0,np.shape(x_import)[0], color='red')
    # plt.hlines(-1*threshold,0,np.shape(x_import)[0], color='red')
    # plt.xlabel('x')
    # plt.ylabel('dffits')
    # plt.show()


    if np.abs(dffits[0][outlier_index]) > threshold:
        print('Outlier removed:',outlier_index)
        out_list.append(np.where(x_import == x_import_wo[outlier_index])[0][0])
        x_import_wo = np.delete(x_import_wo,outlier_index,axis=0)
        y_import_wo = np.delete(y_import_wo,outlier_index,axis=0)
    else:
        Test_passed = True

out_list = np.sort(out_list)
print(len(out_list),'outliers found:\n',out_list)


alpha_list = np.linspace(0.001,1,50)
determine_best_model(x_import_wo,y_import_wo,500,alpha_list)