import numpy as np
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, RepeatedStratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.metrics import cohen_kappa_score, make_scorer, balanced_accuracy_score


kappa_scorer = make_scorer(cohen_kappa_score)
bca_scorer = make_scorer(balanced_accuracy_score)


x_import = np.load('data/Xtrain_Classification2.npy')/255.0
y_import = np.load('data/Ytrain_Classification2.npy')


class alternative_model:

    def __init__(self, model_type='RandomForest', n_splits = 10, class_weight='balanced'):
        '''
        model_type -> Specifies the model we want to use
        n_splits -> Necessary for all
        class_weights -> Necesary for Random Forest ['balanced', 'balanced_subsample']
        '''



        self.model_type = model_type
        print('Created {} model with {} n_splits'.format(model_type, n_splits))
        if model_type == 'Balanced Bagging':
            self.model = BalancedBaggingClassifier()
        
        elif model_type == 'Bagging':
            self.model = BaggingClassifier()
        
        elif model_type == 'Random Forest':
            self.model = RandomForestClassifier(n_estimators=10, class_weight=class_weight)

        elif model_type == 'Balanced Random Forest':
            self.model = BalancedRandomForestClassifier(n_estimators=10, class_weight=class_weight)
        
        self.n_splits = n_splits 

    def test_model(self, cv_type='KFold'):

        print('Testing {} with cv {}'.format(self.model_type,cv_type))
        self.cv_type = cv_type

        if self.cv_type == 'KFold':
            self.cv = KFold(self.n_splits)
            self.scoring = {'accuracy':'accuracy','baccuracy': bca_scorer ,'kappa':kappa_scorer}
            self.scores = cross_validate(self.model, x_import, y_import , scoring=self.scoring, cv=self.cv, n_jobs=-1)
        
        elif self.cv_type == 'RepeatedStratifiedKFold':
            # Does not support multiclassification
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        
        else: 
            print('CV type not supported')
            quit()

        print('\n\n {} Results ======'.format(self.model_type))
        print('\n Mean Accuracy {}\n Mean Balenced Accuracy: {}\n Mean Kappa: {}\n\n\n'.format(mean(self.scores['test_accuracy']),mean(self.scores['test_baccuracy']),mean(self.scores['test_kappa'])))






# modelB = alternative_model(
#     model_type = 'Bagging',
#     n_splits = 15
# )

modelBB = alternative_model(
    model_type ='Balanced Bagging',
    n_splits = 15
    )

# modelRF = alternative_model(
#     model_type = 'Random Forest',
#     n_splits = 15,
#     class_weight='balanced'
#     )

modelBRF = alternative_model(
    model_type = 'Balanced Random Forest',
    n_splits = 15,
    class_weight='balanced'
)



# modelB.test_model()
modelBB.test_model()
#modelRF.test_model()
modelBRF.test_model()