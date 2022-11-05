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

    return S, Z


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
        S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + np.dot(A*np.sqrt(Deltat), Z[:, i - 1:i]) + (v * Deltat).reshape(3,1))
        Stilde[:, i:i + 1] = np.exp(np.log(Stilde[:, i - 1:i]) + np.dot(A*np.sqrt(Deltat), -Z[:, i - 1:i]) + (v * Deltat).reshape(3,1))

    return S, Stilde


def SimMultiGBMpmh(S0, v, sigma, Deltat, T, Z):
    m = int(T/Deltat)
    p = len(S0)
    S = np.zeros((3*p, m+1))

    h = []
    S0_ph = []
    S0_mh = []

    for i in range(p):
        h.append(0.01*S0[i])
        S0_ph.append(S0[i] + h[i])
        S0_mh.append(S0[i] - h[i])

    Z_ = np.zeros((3 * p, m))

    for i in range(p):
        S[p*i][0] = S0_mh[i]
        S[p*i+1][0] = S0[i]
        S[p*i+2][0] = S0_ph[i]
        Z_[3*i] = Z[i]
        Z_[3*i+1] = Z[i]
        Z_[3*i+2] = Z[i]
    # print("Z_")
    # print(Z_)
    # print("-------")
    # print("Z_ transpose")
    # print(Z_)
    # print("-------")
    # print("Z")
    # print(Z)
    # print("-------")
    # print("S")
    # print(S)
    # print("-------")
    for i in range(1, m+1):
        S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + Z_[:, i - 1:i])

    return S, h



# def SimMultiGBMCV(S0,v,sigma,N1,N2,r,Deltat,T,total_trading_days,q2_index,q3_index):
#     m = int(T/Deltat)
#     p = len(S0)
#     S = np.zeros((p,m+1))
#     S[:,0] = S0

#     C = []
#     for j in range(N1):
#         Z = np.random.multivariate_normal(mean=v * Deltat, cov=sigma * Deltat, size=(m, 1, 1))
#         Z = np.transpose(Z[:, 0, 0,:])

#         for i in range(1, m + 1):
#             S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + Z[:, i - 1:i])
        
#         C.append(calculate_option_price(aapl=S[0],amzn=S[1],googl=S[2],T=T,total_trading_days=total_trading_days,r=r,q2_index=q2_index,q3_index=q3_index))
    
    
#     c_star_1 = -1*np.cov(X[0,1:],Y[0,:])[0][1]/np.var(Y[0,:])
#     c_star_2 = -1*np.cov(X[1,1:],Y[1,:])[0][1]/np.var(Y[1,:])
#     c_star_3 = -1*np.cov(X[2,1:],Y[2,:])[0][1]/np.var(Y[2,:])
#     c_star = np.array([c_star_1,c_star_2,c_star_3]).reshape(3,1)
#     for i in range (1, m + 1):
#         X_CV[:, i:i + 1] = X[:, i:i + 1] + c_star*(Y[:, i - 1:i] - (v*Deltat).reshape(3,1))

#     return X_CV
