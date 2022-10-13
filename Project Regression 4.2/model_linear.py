#Import everything from initialization
from initialization import * 


def linear_model(x,y):
    '''Returns a Linear Model'''
    model = LinearRegression()
    model.fit(x,y)
    return model

