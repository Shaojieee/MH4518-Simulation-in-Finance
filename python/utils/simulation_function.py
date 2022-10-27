import numpy as np

def SimMultiGBM(S0,v,sigma,Deltat,T):
    m=int(T/Deltat)
    p=len(S0)
    S=np.zeros((p,m+1))
    S[:,0]=S0
    Z = np.random.multivariate_normal(mean=v*Deltat,cov=sigma*Deltat,size=(m,1,1))
    Z = np.transpose(Z[:,0,0,:])

    for i in range(1,m+1):
        S[:,i:i+1]=np.exp(np.log(S[:,i-1:i])+Z[:,i-1:i])

    return S

def SimMultiGBMAV(S0,v,sigma,Deltat,T):
    m=int(T/Deltat)
    p=len(S0)
    S=np.zeros((p,m+1))
    Stilde=np.zeros((p,m+1))
    S[:,0]=S0
    Stilde[:,0]=S0
    Z = np.random.multivariate_normal(mean=v*Deltat,cov=sigma*Deltat,size=(m,1,1))
    Z = np.transpose(Z[:,0,0,:])

    for i in range(1,m+1):
        S[:,i:i+1]=np.exp(np.log(S[:,i-1:i])+Z[:,i-1:i])
        Stilde[:,i:i+1]=np.exp(np.log(Stilde[:,i-1:i])-Z[:,i-1:i])

    return S,Stilde