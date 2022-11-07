import math
from scipy.stats import norm
from utils.extract_data_function import extract_current_price
import numpy as np

import math
from scipy.stats import norm
from utils.extract_data_function import extract_current_price
import numpy as np

def estimate_call_IV(C_mkt,S_t,K,r,T,sigma_hat_list,left,right):
    # C_mkt : Market Price of Call option
    # K : The strike price of Call option
    # r : Risk-free interest rate
    # S_t : Current stock price of the Call option
    # T : Time to maturity
    # sigma_hat : estimated implied volatility (Trial and error)

    #Initialise a list of implied volatility with step size = error 
    #between min_sigma and max_sigma to conduct a binary search on to find the sigma
    #that estimates call price to be within + - absolute error of the market price

    if left>right:
        return -1

    if right>=left:
        mid = (left + right)//2
        sigma_hat = sigma_hat_list[mid]
        d1 = 1/(sigma_hat*math.sqrt(T))*(math.log(S_t/K)+(r+(sigma_hat**2)/2)*T)
        d2 = d1 - sigma_hat*math.sqrt(T)
        estimated_call_price = norm.cdf(d1)*S_t - norm.cdf(d2)*K*math.exp(-r*T)

        if C_mkt-estimated_call_price>0:
            # sigma_hat is too small and the better answer exist in the right subarray
            value = estimate_call_IV(C_mkt,S_t,K,r,T,sigma_hat_list,mid+1,right)
            return value if value!=-1 else sigma_hat
        
        else:
            # sigma_hat is too big and the better approximate exist in the left subarray
            value = estimate_call_IV(C_mkt,S_t,K,r,T,sigma_hat_list,left,mid-1)
            return value if value!=-1 else sigma_hat
  

def cov_estimated_IV(aapl_call_df,amzn_call_df,googl_call_df,r,alternative_option_ttm,sigma_hat_list,left,right,date_to_predict,AAGlogreturns):

    aapl_current_price = extract_current_price('../data/04-11-2022/aapl_04_11_2022.csv',date_to_predict)
    amzn_current_price = extract_current_price('../data/04-11-2022/amzn_04_11_2022.csv',date_to_predict)
    googl_current_price = extract_current_price('../data/04-11-2022/googl_04_11_2022.csv',date_to_predict)

    if aapl_call_df[aapl_call_df['Date']==date_to_predict]['Closing Price'].isnull().all():
        #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol
        aapl_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[0][0])
    else:
        # There exist at least one option traded on this day, we take the average IV of all option traded on this day
        aapl_call_list = aapl_call_df[aapl_call_df['Date']==date_to_predict].values.tolist()
        aapl_IV_list = []

        for row in aapl_call_list:
            aapl_K = row[-2]
            aapl_C_mkt = row[-1]
            if math.isnan(aapl_C_mkt):
                continue   
            aapl_IV = estimate_call_IV(aapl_C_mkt,aapl_current_price,aapl_K,r/alternative_option_ttm,alternative_option_ttm,sigma_hat_list,left,right)
            if aapl_IV!=-1 and aapl_IV!=0:         
                aapl_IV_list.append(aapl_IV)

        aapl_sigma_hat = np.mean(aapl_IV_list)
    
    if amzn_call_df[amzn_call_df['Date']==date_to_predict]['Closing Price'].isnull().all():
        #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol
        amzn_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[1][1])

    else:
        # There exist at least one option traded on this day, we take the average IV of all option traded on this day
        amzn_call_list = amzn_call_df[amzn_call_df['Date']==date_to_predict].values.tolist()
        amzn_IV_list = []

        for row in amzn_call_list:
            amzn_K = row[-2]
            amzn_C_mkt = row[-1]
            if math.isnan(amzn_C_mkt):
                continue
            amzn_IV = estimate_call_IV(amzn_C_mkt,amzn_current_price,amzn_K,r/alternative_option_ttm,alternative_option_ttm,sigma_hat_list,left,right)
            if amzn_IV!=-1 and amzn_IV!=0:         
                amzn_IV_list.append(amzn_IV)
        
        amzn_sigma_hat = np.mean(amzn_IV_list)                

    if googl_call_df[googl_call_df['Date']==date_to_predict]['Closing Price'].isnull().all():
        #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol
        googl_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[2][2])
    
    else:
        # There exist at least one option traded on this day, we take the average IV of all option traded on this day
        googl_call_list = googl_call_df[googl_call_df['Date']==date_to_predict].values.tolist()
        googl_IV_list = []

        for row in googl_call_list:
            googl_K = row[-2]
            googl_C_mkt = row[-1]
            if math.isnan(googl_C_mkt):
                continue
            googl_IV = estimate_call_IV(googl_C_mkt,googl_current_price,googl_K,r/alternative_option_ttm,alternative_option_ttm,sigma_hat_list,left,right)
            if googl_IV!=-1 and googl_IV!=0:         
                googl_IV_list.append(googl_IV)        
        
        googl_sigma_hat = np.mean(googl_IV_list)

    #Calculating covariance matrix from implied volatility and correlation matrix
    rho = np.corrcoef(AAGlogreturns, rowvar=False)
    V = np.diag([aapl_sigma_hat,amzn_sigma_hat,googl_sigma_hat])
    sigma = np.dot(np.dot(V,rho),V)

    return sigma

def cov_actual_IV(aapl_IV_df,amzn_IV_df,googl_IV_df,date_to_predict,AAGlogreturns):

    #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol
    if aapl_IV_df[aapl_IV_df['Date']==date_to_predict]['IV'].isnull().all():
        aapl_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[0][0])
    
    # There exist at least one option traded on this day, we take the average IV of all option traded on this day
    else:

        aapl_IV_df = aapl_IV_df[aapl_IV_df['Date']==date_to_predict].values.tolist()
        aapl_IV_list = []
        for row in aapl_IV_df:
            aapl_IV_list.append(row[-1])
        aapl_sigma_hat = np.mean(aapl_IV_list)
    
    #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol

    if amzn_IV_df[amzn_IV_df['Date']==date_to_predict]['IV'].isnull().all():
        amzn_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[1][1])
    
    # There exist at least one option traded on this day, we take the average IV of all option traded on this day
    else:

        amzn_IV_df = amzn_IV_df[amzn_IV_df['Date']==date_to_predict].values.tolist()
        amzn_IV_list = []
        for row in amzn_IV_df:
            amzn_IV_list.append(row[-1])
        amzn_sigma_hat = np.mean(amzn_IV_list)
    
    #No option traded on this day, unable to backcalculate the IV for this day --> Use historical vol
    if googl_IV_df[googl_IV_df['Date']==date_to_predict]['IV'].isnull().all():
        
        googl_sigma_hat = math.sqrt(np.cov(AAGlogreturns,rowvar=False)[2][2])
    
    # There exist at least one option traded on this day, we take the average IV of all option traded on this day
    else:

        googl_IV_df = googl_IV_df[googl_IV_df['Date']==date_to_predict].values.tolist()
        googl_IV_list = []
        for row in googl_IV_df:
            googl_IV_list.append(row[-1])
        googl_sigma_hat = np.mean(googl_IV_list)
    
    #Calculating covariance matrix from implied volatility and correlation matrix
    rho = np.corrcoef(AAGlogreturns, rowvar=False)
    V = np.diag([aapl_sigma_hat,amzn_sigma_hat,googl_sigma_hat])
    sigma = np.dot(np.dot(V,rho),V)

    return sigma