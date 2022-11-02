from ast import Del
from re import X
import numpy as np
import math


def SimMultiGBM(S0, v, sigma, Deltat, T):
    m = int(T / Deltat)
    p = len(S0)
    S = np.zeros((p, m + 1))
    S[:, 0] = S0
    Z = np.random.multivariate_normal(mean=v * Deltat, cov=sigma * Deltat, size=(m, 1, 1))
    Z = np.transpose(Z[:, 0, 0, :])

    for i in range(1, m + 1):
        S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + Z[:, i - 1:i])

    return S


def SimMultiGBMAV(S0, v, sigma, Deltat, T):
    m = int(T / Deltat)
    p = len(S0)
    S = np.zeros((p, m + 1))
    Stilde = np.zeros((p, m + 1))
    S[:, 0] = S0
    Stilde[:, 0] = S0

    # Applying Linear transform on standard multivariate normal samples
    Z = np.random.multivariate_normal(mean=np.zeros(len(v)), cov=np.eye(N=len(v), M=len(v)), size=(m, 1, 1))
    Z = np.transpose(Z[:, 0, 0, :])
    A = np.linalg.cholesky(sigma)

    for i in range(1, m + 1):
        S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + np.dot(A*Deltat, Z[:, i - 1:i]) + (v * Deltat).reshape(3,1))
        Stilde[:, i:i + 1] = np.exp(np.log(Stilde[:, i - 1:i]) + np.dot(A*Deltat, -Z[:, i - 1:i]) + (v * Deltat).reshape(3,1))

    return S, Stilde

def SimMultiGBMCV(S0,v,sigma,Deltat,T):
    m = int(T/Deltat)
    p = len(S0)
    X = np.zeros((p,m+1))
    X_CV = np.zeros((p,m+1))
    X[:,0] = S0
    X_CV[:,0] = S0
    Y = np.random.multivariate_normal(mean=v * Deltat, cov=sigma * Deltat, size=(m, 1, 1))
    Y = np.transpose(Y[:, 0, 0,:])

    for i in range(1, m + 1):
        X[:, i:i + 1] = np.exp(np.log(X[:, i - 1:i]) + Y[:, i - 1:i])
    
    c_star_1 = -1*np.cov(X[0,1:],Y[0,:])[0][1]/np.var(Y[0,:])
    c_star_2 = -1*np.cov(X[1,1:],Y[1,:])[0][1]/np.var(Y[1,:])
    c_star_3 = -1*np.cov(X[2,1:],Y[2,:])[0][1]/np.var(Y[2,:])
    c_star = np.array([c_star_1,c_star_2,c_star_3]).reshape(3,1)
    for i in range (1, m + 1):
        X_CV[:, i:i + 1] = X[:, i:i + 1] + c_star*(Y[:, i - 1:i] - (v*Deltat).reshape(3,1))

    return X_CV
