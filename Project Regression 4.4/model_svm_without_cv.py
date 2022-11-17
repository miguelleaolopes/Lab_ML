import numpy as np
from numpy import mean
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, RepeatedStratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, balanced_accuracy_score
import pandas as pd
import winsound

kappa_scorer = make_scorer(cohen_kappa_score)
bca_scorer = make_scorer(balanced_accuracy_score)


x_import = np.load('data/Xtrain_Classification2.npy')/255.0
y_import = np.load('data/Ytrain_Classification2.npy')


rows = np.shape(x_import)[0]
colls = np.shape(x_import)[1]

x_import_small = x_import[0:int(rows),:]
y_import_small= y_import[0:int(rows)]

def test_model():

    print('Testing SVM with GridSearchCV')
    
    scoring = {'accuracy':'accuracy','baccuracy': bca_scorer ,'kappa':kappa_scorer}
    # param_grid = {'C': (Cs), 
    #     'gamma': (gammas),
    #     'kernel': ['rbf']} # rbf
    modelsvc = SVC(class_weight= 'balanced', gamma = 0.8, kernel = 'rbf', C=1, verbose = 3)
    modelsvc.fit(x_import_small, y_import_small)
    

    return modelsvc


modelsvc = test_model()


x_final_test = np.load('data/Xtest_Classification2.npy')/255.0

y_final_test = modelsvc.predict(x_final_test)

print("Y_pred:", np.shape(y_final_test))
np.save("data/y_pred.npy",y_final_test)

x_final_test = np.reshape(x_final_test,(33800,5,5,3))
y_final_test = np.reshape(y_final_test,(33800,1,1))
x_final_test_reconstructed = np.zeros((50,26,26,3))
y_final_test_reconstructed = np.zeros((50,26,26))

for i in range(0,49):
    image = reconstruct_from_patches_2d(x_final_test[i*676:(i+1)*676],(30,30,3))[2:28,2:28]
    y_image = reconstruct_from_patches_2d(y_final_test[i*676:(i+1)*676],(26,26))
    x_final_test_reconstructed[i] = image
    y_final_test_reconstructed[i] = y_image

def show_images(x,y,index):
    plt.subplot(1,2,1)
    plt.imshow(x[index])
    plt.subplot(1,2,2)
    plt.imshow(y[index])
    plt.show()

print()
for i in range(50): show_images(x_final_test_reconstructed,y_final_test_reconstructed,i)


duration = 2000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)