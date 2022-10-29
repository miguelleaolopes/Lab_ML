import numpy as np
import tqdm

y_pred_1 = np.load('data/y_pred_1.npy')
y_pred_2 = np.load('data/ytest_Classification1.npy')

if np.shape(y_pred_1) == np.shape(y_pred_2): same_size = True
else: same_size = False


print('\n\n Files with the same size:', same_size, '\n y pred 1 shape:', np.shape(y_pred_1))

total_dif = 0
total_size = int(np.shape(y_pred_1)[0])
dif_arr = []
if same_size:
    for i in tqdm.trange(total_size):
        if y_pred_1[i] != y_pred_2[i]:
            total_dif += 1
            dif_arr.append(i)
    print('Found', total_dif,'different predictions')  
    print('Different predictions:', total_dif/total_size*100,'%')
    print(dif_arr)

else: print('Comparisson not made due to different size')

