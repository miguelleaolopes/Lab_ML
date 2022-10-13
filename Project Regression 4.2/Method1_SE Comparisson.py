from initialization import *
from model_ridge import *
from model_lasso import *
from model_linear import *
from determine_best_model import determine_best_model

modelwout = linear_model(x_import,y_import)

# Now we want to see the se for each x and remove the outliers

SE_list, out_list = [], []
index_list = [i for i in range(np.shape(x_import)[0])]
y_pred = modelwout.predict(x_import)

for i in range(len(x_import)):
    SE_list.append((y_import[i]-y_pred[i])**2)

sigma = np.mean(SE_list)
threshold = 3*sigma

plt.plot(index_list, SE_list, color='blue', marker='o')
plt.hlines(threshold,index_list[0],index_list[-1], color='red')
plt.title('Squared Error vs Data Index', fontsize=14)
plt.xlabel('SE', fontsize=14)
plt.ylabel('Data Index', fontsize=14)
plt.grid(True)
plt.show()


for i in range(len(x_import)):
    if SE_list[i] > threshold:
        out_list.append(i)

print('There are',len(out_list),' outliers :\n',out_list)

x_import_wo = np.delete(x_import, out_list, axis=0)
y_import_wo = np.delete(y_import, out_list, axis=0)


determine_best_model(x_import_wo,y_import_wo,500)