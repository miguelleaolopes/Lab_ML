from model_linear import *
from determine_best_model import *
from yellowbrick.base import Visualizer
from yellowbrick.regressor import CooksDistance
from yellowbrick.regressor import ResidualsPlot
import statsmodels.api as sm

class method2_2_cyclical_se_comparisson

print("Determining outliars using cyclical Cooks Distance")


Test_passed = False
out_list = []
x_import_wo, y_import_wo = x_import.copy(), y_import.copy()

while not Test_passed:# and len(out_list) < 20:
    col_ones = np.ones((np.shape(x_import_wo)[0],1))
    Xbig_import = np.hstack([col_ones,x_import_wo])
    model = sm.OLS(y_import_wo, Xbig_import).fit() 
    np.set_printoptions(suppress=True)

    influence = model.get_influence()
    summary_influence = influence.summary_frame()
    cooks = influence.cooks_distance

    outlier = np.amax(cooks[0])
    outlier_index = np.where(cooks[0] == outlier)[0][0]

    threshold_cook = 4/np.shape(x_import_wo)[0]



    if cooks[0][outlier_index] > threshold_cook:
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