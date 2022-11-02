import numpy as np

def EMSCorrection(S,Nsim,r,Deltat,T):

    m = int(T/Deltat)
    S_Star = np.zeros((Nsim*2,m+1))
    S = np.array(S)
    S_Star[:,0] = S[:,0]
    Z = np.zeros((Nsim*2,m))

    for i in range(1,m):
        Z[:,i-1] = S_Star[:,i-1] * (S[:,i]/S[:,i-1])
        Z_0 = 1/(Nsim*2) * np.exp(-1*r*i*Deltat) * np.sum(Z[:,i-1])
        S_Star[:,i] = S_Star[:,0] * Z[:,i-1]/Z_0

    return S_Star.tolist()
        


