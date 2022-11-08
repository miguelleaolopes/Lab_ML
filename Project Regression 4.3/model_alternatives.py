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


x_import = np.load('data/Xtrain_Classification1.npy')/255.0
y_import = np.load('data/Ytrain_Classification1.npy')


class alternative_model:

    def __init__(self, model_type='RandomForest', n_splits = 10, class_weight='balanced',kernel='rbf', gamma=0, n_neighbors = 0, manual_random_oversampling = 'None'):
        '''
        model_type -> Specifies the model we want to use
        n_splits -> Necessary for all
        class_weights -> Necesary for Random Forest ['balanced', 'balanced_subsample']
        kernel -> Necessary for svf ['rbf','poly','linear']
        gamma -> Necessary for svf
        n_neighbors -> Necessary for KNN 
        '''


        
        self.model_type = model_type
        print('Created {} model with {} n_splits'.format(model_type, n_splits))
        global x_import, y_import

        if manual_random_oversampling == 'RandomOverSampler':
            x_import = np.load('data/Xtrain_Classification2.npy')
            print('Using {} oversampling'.format(manual_random_oversampling))
            print('Size of original lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))
            x_resample, y_resample = RandomOverSampler().fit_resample(x_import, y_import)
            x_import, y_import = x_resample/255.0, y_resample
            print('Size of oversampled lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))
        
        elif manual_random_oversampling == 'SMOTE':
            x_import = np.load('data/Xtrain_Classification2.npy')
            print('Using {} oversampling'.format(manual_random_oversampling))
            print('Size of original lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))
            x_resample, y_resample = SMOTE().fit_resample(x_import, y_import)
            x_import, y_import = x_resample/255.0, y_resample
            print('Size of oversampled lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))

        elif manual_random_oversampling == 'ADASYN':
            x_import = np.load('data/Xtrain_Classification2.npy')
            print('Using {} oversampling'.format(manual_random_oversampling))
            print('Size of original lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))
            x_resample, y_resample =  ADASYN().fit_resample(x_import,y_import)
            x_import, y_import = x_resample/255.0, y_resample
            print('Size of oversampled lists===\nx_import:{}\ny_import:{}'.format(np.shape(x_import),np.shape(y_import)))
            
            



        if model_type == 'Balanced Bagging':
            self.model = BalancedBaggingClassifier(
                sampling_strategy='auto'
            )
        
        elif model_type == 'Bagging':
            self.model = BaggingClassifier()
        
        elif model_type == 'Random Forest':
            self.model = RandomForestClassifier(
                n_estimators=410, 
                class_weight=class_weight
                )

        elif model_type == 'Balanced Random Forest':
            self.model = BalancedRandomForestClassifier(
                n_estimators=10, 
                class_weight=class_weight
                )

        elif model_type == 'svm':
            self.kernel = kernel
            self.gamma = gamma
            self.model = SVC(kernel=kernel,gamma=gamma)

        elif model_type == 'Decision Tree':
            self.model = DecisionTreeClassifier()

        elif model_type == 'KNN':
            print('Using {} n_neighbors'.format(n_neighbors))
            self.n_neighbors  = n_neighbors
            self.model = KNeighborsClassifier(n_neighbors = self.n_neighbors)

        elif model_type == 'GNBC':
            #Gaussian Naive Bayes classifier
            self.model = GaussianNB()
        
        else:
            print('Model type does not exist')
            quit()
        
        self.n_splits = n_splits 

    def test_model(self, cv_type='KFold'):

        print('Testing {} with cv {}'.format(self.model_type,cv_type))
        self.cv_type = cv_type

        if self.cv_type == 'KFold':
            self.cv = KFold(self.n_splits)
            self.scoring = {'accuracy':'accuracy','baccuracy': bca_scorer ,'f1':'f1'}
            self.scores = cross_validate(self.model, x_import, y_import , scoring=self.scoring, cv=self.cv, n_jobs=-1)
            # self.scores = cros_validate(self.model, x_import, y_import , scoring=self.scoring, cv=self.cv, n_jobs=-1)
        
        elif self.cv_type == 'RepeatedStratifiedKFold':
            # Does not support multiclassification
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        
        else: 
            print('CV type not supported')
            quit()

        print('\n\n {} Results ======'.format(self.model_type))
        print('\n Mean Accuracy {}\n Mean Balenced Accuracy: {}\n Mean F1: {}\n\n\n'.format(mean(self.scores['test_accuracy']),mean(self.scores['test_baccuracy']),mean(self.scores['test_f1'])))



def find_best_svm_gamma(gammas):
    b_acu = []
    for gamma in gammas:

        print('\n\nTesting for gamma {}'.format(gamma))
        modelSVM = alternative_model(
            model_type = 'svm',
            n_splits = 10,
            kernel = 'rbf',
            gamma = gamma
            )

        modelSVM.test_model()
        b_acu.append(np.mean(modelSVM.scores['test_baccuracy']))

    best_b_acu = np.max(b_acu)
    index = np.where(b_acu == best_b_acu)
    best_gamma = gamma[index]

    print('The best gamma for svm is {} with an balanced accuracy of {}'.format(best_gamma, best_b_acu))


modelB = alternative_model(
    model_type = 'Bagging',
    n_splits = 5
)

modelBB = alternative_model(
    model_type ='Balanced Bagging',
    n_splits = 5
    )

modelRF = alternative_model(
    model_type = 'Random Forest',
    n_splits = 5,
    class_weight='balanced'
    )

modelBRF = alternative_model(
    model_type = 'Balanced Random Forest',
    n_splits = 5,
    class_weight='balanced',
    # manual_random_oversampling='SMOTE'
)

modelSVM = alternative_model(
    model_type = 'svm',
    n_splits = 5,
    kernel = 'rbf',
    gamma=0.01
)

modelDT = alternative_model(
    model_type='Decision Tree',
    n_splits = 5
)

modelKNN = alternative_model(
    model_type='KNN',
    n_splits = 5,
    n_neighbors = 3
)

modelGNBC = alternative_model(
    model_type='GNBC',
    n_splits=5
)

modelB.test_model()
# modelBB.test_model()
modelRF.test_model()
# modelBRF.test_model()
modelDT.test_model()
modelKNN.test_model()
modelGNBC.test_model()
# find_best_svm_gamma(np.linspace(0.001,0.1,5))

# x_test = np.load('data/Xtest_Classification2.npy')/255.0

# modelBB.model.fit(x_import,y_import)
# ypredBB = modelBB.model.predict(x_test)

# print('Y pred shape:{}'.format(np.shape(ypredBB)))
# np.save('data/ypred_BB.npy',ypredBB)
# print('npy saved')

# modelBRF.model.fit(x_import,y_import)
# ypredBRF = modelBRF.model.predict(x_test)

# print('Y pred shape:{}'.format(np.shape(ypredBRF)))
# np.save('data/ypred_BRF.npy',ypredBRF)
# print('npy saved')

# ypredBRF = np.load('y_pred_finalisssimo.npy')


# x_test = np.reshape(x_test,(np.shape(x_test)[0],5,5,3))
# y_testBB = np.reshape(ypredBB,(np.shape(x_test)[0],1,1))
# y_testBRF = np.reshape(ypredBRF,(np.shape(x_test)[0],1,1))

# x_test_reconstructed = np.zeros((int(np.shape(x_test)[0]/26/26),26,26,3))
# ypred_reconstructedBB = np.zeros((int(np.shape(x_test)[0]/26/26),26,26))
# ypred_reconstructedBRF = np.zeros((int(np.shape(x_test)[0]/26/26),26,26))

# for i in range(0,int(np.shape(x_test)[0]/26/26)):
#     image = reconstruct_from_patches_2d(x_test[i*676:(i+1)*676],(30,30,3))[2:28,2:28]
#     yBB_image = reconstruct_from_patches_2d(y_testBB[i*676:(i+1)*676],(26,26))
#     yBRF_image = reconstruct_from_patches_2d(y_testBRF[i*676:(i+1)*676],(26,26))
#     x_test_reconstructed[i] = image
#     ypred_reconstructedBB[i] = yBB_image
#     ypred_reconstructedBRF[i] = yBRF_image
    

# def show_images(x,y,z,index):
#     plt.subplot(1,3,1)
#     plt.imshow(x[index])
#     plt.subplot(1,3,2)
#     plt.imshow(y[index])
#     plt.subplot(1,3,3)
#     plt.imshow(z[index])
#     plt.show()

# for i in range(0,50): show_images(x_test_reconstructed,ypred_reconstructedBB,ypred_reconstructedBRF,i)