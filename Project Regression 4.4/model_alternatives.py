import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, balanced_accuracy_score


kappa_scorer = make_scorer(cohen_kappa_score)
bca_scorer = make_scorer(balanced_accuracy_score)


x_import = np.load('data/Xtrain_Classification2.npy')/255.0
y_import = np.load('data/Ytrain_Classification2.npy')


class alternative_model:

    def __init__(self, model_type, n_splits = 10):

        if model_type == 'Balanced Bagging':
            self.model = BalancedBaggingClassifier()
        elif model_type == 'Bagging':
            self.model = BaggingClassifier()
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=10, class_weight='balanced')
        
        self.n_splits = n_splits 

    def test_model(self, cv_type='KFold'):
        self.cv_type = cv_type

        if self.cv_type == 'KFold':
            self.cv = KFold(self.n_splits)
            self.scores = cross_validate(self.model, x_import, y_import , scoring={'accuracy':'accuracy','baccuracy': bca_scorer ,'kappa':kappa_scorer}, cv=self.cv, n_jobs=-1)
        
        if self.cv_type == 'RepeatedStratifiedKFold':
            # Does not support multiclassification
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        
        else: 
            print('CV type not supported')
            quit()

        print('Mean Accuracy {}\n Mean Accuracy: {}\n Mean Kappa: {}'.format(mean(scores['test_accuracy']),mean(scores['test_baccuracy']),mean(scores['test_kappa'])))




