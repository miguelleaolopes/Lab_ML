from initialization import *



def ridge_modelcv(x,y,alpha_list,cv,fit_int=False,solv=False):
    '''returns the best parameters for a ridge model using GridSearchCV'''
    param = {'alpha':alpha_list}

    if fit_int:
        param['fit_intercep'] = [True,False]
    elif solv:
        param['solver'] = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'] #,'lbfgs'
    
    model = Ridge()
    search = GridSearchCV(model, param, scoring='r2', n_jobs=-1, cv=cv)
    result = search.fit(x, y)
    return result, result.best_params_

def ridge_model(x,y,alpha):
    model = Ridge(alpha=alpha)
    model.fit(x,y)
    return model
