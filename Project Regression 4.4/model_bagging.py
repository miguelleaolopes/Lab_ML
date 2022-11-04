import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from keras.utils import np_utils

#Import libraries
x_import = np.load('data/Xtrain_Classification2.npy')/255.0
y_import = np.load('data/Ytrain_Classification2.npy')


# model = BaggingClassifier()
model = BalancedBaggingClassifier()

# define evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1) #RepeatedStratifiedKfold does not support multiclassification
cv = KFold(n_splits=20)


# evaluate model
scores = cross_validate(model, x_import, y_import, scoring='accuracy', cv=cv, n_jobs=-1) #metrics.balanced_accuracy_score

# summarize performance
print('Mean Accuracy: %.3f' % mean(scores['test_score']))
