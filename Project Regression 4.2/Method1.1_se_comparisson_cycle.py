from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *
from determine_best_model import determine_best_model

print("Determining outliars using ciclical SE comparisson method")

Test_passed = False
out_list = []
x_import_wo, y_import_wo = x_import.copy(), y_import.copy()


while not Test_passed and len(out_list) < 20:

    modelwout = linear_model(x_import_wo,y_import_wo)
    y_pred = modelwout.predict(x_import_wo)
    SE_list = []

    for i in range(len(x_import_wo)):
        SE_list = np.hstack([SE_list,(y_import_wo[i]-y_pred[i])**2])
    
    threshold = np.mean(SE_list) + 4.3*np.std(SE_list)
    outlier = np.amax(SE_list)
    outlier_index = np.where(SE_list == outlier)[0][0]

 
    if SE_list[outlier_index] > threshold:
        print('Outlier removed:',outlier_index)
        out_list.append(np.where(x_import == x_import_wo[outlier_index])[0][0])
        x_import_wo = np.delete(x_import_wo,outlier_index,axis=0)
        y_import_wo = np.delete(y_import_wo,outlier_index,axis=0)
    else:
        Test_passed = True

out_list = np.sort(out_list)
print(len(out_list),'outliers found:\n',out_list)

determine_best_model(x_import_wo,y_import_wo,500,np.linspace(0.001,2,50))