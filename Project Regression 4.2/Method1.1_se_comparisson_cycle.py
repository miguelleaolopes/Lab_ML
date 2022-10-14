from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *
from determine_best_model import determine_best_model

print("Determining outliars using ciclical SE comparisson method")

Test_passed = False
out_list = []

x_import_wo, y_import_wo = x_import.copy(), y_import.copy()

while not Test_passed:

    modelwout = linear_model(x_import_wo,y_import_wo)
    y_pred = modelwout.predict(x_import_wo)
    SE_list = []

    for i in range(len(x_import_wo)):
        SE_list = np.hstack([SE_list,(y_import_wo[i]-y_pred[i])**2])
    
    sigma = np.std(SE_list)
    threshold = np.mean(SE_list) + 0.1*sigma

    outlier = np.amax(SE_list)
    outlier_index = np.where(SE_list == outlier)[0][0]
 
    if SE_list[outlier_index] > threshold:
        print('Outlier removed:',outlier_index)
        out_list.append(np.where(x_import == x_import_wo[outlier_index]))
        np.delete(x_import_wo,[outlier_index],axis=0)
        np.delete(y_import_wo,[outlier_index],axis=0)

    else:
        Test_passed == True





    




modelwout = linear_model(x_import,y_import)
SE_list, out_list = [], []

y_pred = modelwout.predict(x_import)

for i in range(len(x_import)):
    SE_list.append((y_import[i]-y_pred[i])**2)

sigma = np.std(SE_list)
threshold = np.mean(SE_list) + 0.1*sigma


plt.plot(index_list, SE_list, color='blue', marker='o')
plt.hlines(threshold,index_list[0],index_list[-1], color='red')
plt.title('Squared Error vs Data Index', fontsize=14)
plt.ylabel('SE', fontsize=14)
plt.xlabel('Data Index', fontsize=14)
plt.grid(True)
# plt.show()


for i in range(len(x_import)):
    if SE_list[i] > threshold:
        out_list.append(i)

print('There are',len(out_list),' outliers :\n',out_list)

x_import_wo = np.delete(x_import, out_list, axis=0)
y_import_wo = np.delete(y_import, out_list, axis=0)


alpha_list = np.linspace(0.001,2,50)
determine_best_model(x_import_wo,y_import_wo,100,alpha_list)
# It also works with just this and much faster