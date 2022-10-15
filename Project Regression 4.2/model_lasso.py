from initialization import *

# def lasso_modelcv(x,y,alpha_list,cv=None,fit_int=False):
#     '''returns a lasso model with the best parameters and a dictionary with the best paraneters'''
    
#     param = {'alpha':alpha_list}
    
#     if fit_int:
#         param['fit_intercept'] = [True,False]
    
#     model = Lasso()
#     search = GridSearchCV(model, param, scoring='r2' ,cv = cv)
#     result = search.fit(x, y)
#     return result, result.best_params_


def lasso_modelcv(x,y,alpha_list,cv=None,fit_int=False):
    '''returns a lasso model with the best parameters and a dictionary with the best paraneters'''
    model = LassoCV()
    result = model.fit(x, np.ravel(y))
    return result, result.alpha_


def lasso_model(x,y,alpha,fit_intercept):
    '''Returns a lasso model given certain parameters'''
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(x,y)
    return model
