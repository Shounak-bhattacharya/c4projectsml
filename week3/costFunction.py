import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    '''returns cost for theta, X and y
    np.log(a)==> returns array with elementwise log on array a
    use the sigmoid function that's being imported above
'''
    m = y.size
    h =sigmoid(X.dot(theta))

    J =(1/m)*((-1*y).T.dot(np.log(h))-(1-y).T.dot(np.log(1-h)))
    print(J)
    if(np.isnan(J[0])):
        return(np.inf)
    return(J[0])


def gradient(theta, X, y):
    m=y.size
    theta=theta.reshape(-1,1)
    print(theta)
    h=sigmoid(X.dot(theta))
    grad=(1/m)*(X.T.dot(h-y))
    return(grad.flatten())

#returns copy of array in one dimension
