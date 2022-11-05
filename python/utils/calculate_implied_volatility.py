import math
from scipy.stats import norm

#min_sigma = some value
#max_sigma = some value
#error = some value
#sigma_hat_list = [x for x in range(min_sigma,max_sigma+error,error)]
#right = len(sigma_hat_list)
#left = 0 
def estimate_IV(C_mkt,S_t,K,r,T,sigma_hat_list,left,right,error):
    # C_mkt : Market Price of Call option
    # K : The strike price of Call option
    # r : Risk-free interest rate
    # S_t : Current stock price of the Call option
    # T : Time to maturity
    # sigma_hat : estimated implied volatility (Trial and error)

    #Initialise a list of implied volatility with step size = error 
    #between min_sigma and max_sigma to conduct a binary search on to find the sigma
    #that estimates call price to be within + - absolute error of the market price
    if right>=left:
        mid = (left + right)//2
        sigma_hat = sigma_hat_list[mid]
        d1 = 1/(sigma_hat*math.sqrt(T))*(math.log(S_t/K)+(r+(sigma_hat**2)/2)*T)
        d2 = d1 - sigma_hat*math.sqrt(T)
        estimated_call_price = norm.cdf(d1)*S_t - norm.cdf(d2)*K*math.exp(-r*T)

        if C_mkt-estimated_call_price>0:
            # sigma_hat is too small and the better answer exist in the right subarray
            value = estimate_IV(C_mkt,S_t,K,r,T,sigma_hat_list,mid+1,right,error)
            return value if value!=-1 else estimated_call_price
        
        else:
            # sigma_hat is too big and the better approximate exist in the left subarray
            value = estimate_IV(C_mkt,S_t,K,r,T,sigma_hat_list,left,mid-1,error)
            return value if value!=-1 else estimated_call_price
    else:
        # unable to find a good enough estimate of implied volatility try changing range of sigma or precision
        return -1