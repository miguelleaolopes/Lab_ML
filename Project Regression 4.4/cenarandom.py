import numpy as np
from numpy import mean
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d


kappa_scorer = make_scorer(cohen_kappa_score)
bca_scorer = make_scorer(balanced_accuracy_score)


y_bb = np.load('ypred_BB.npy')
y_svm = np.load('y_pred_svm.npy')

y_svm2 = y_svm[0:33124]
y_bb_2 = y_bb[33124:33800]
y_pred = np.concatenate((y_svm2,y_bb_2))

print(np.shape(y_pred))
np.save('y_pred_finalisssimo.npy',y_pred)