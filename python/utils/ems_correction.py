import numpy as np

def EMSCorrection(S,Nsim,r,Deltat,T):

    m = int(T/Deltat)
    S_Star = np.zeros((Nsim,m+1))
    S = np.array(S)
    S_Star[:,0] = S[:,0]
    Z = np.zeros((Nsim,m))

    for i in range(1,m+1):
        Z[:,i-1] = S_Star[:,i-1] * (S[:,i]/S[:,i-1])
        Z_0 = 1/(Nsim) * np.exp(-1*r*(i*Deltat)/T) * np.sum(Z[:,i-1])
        S_Star[:,i] = S_Star[:,0] * Z[:,i-1]/Z_0

    return S_Star.tolist()
        


