from ast import Del
from calendar import month, week
from datetime import datetime, timedelta
from tkinter import NS
import payoff_function
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def SimMultiGBM(S0,v,sigma,Deltat,T):
    m=int(T/Deltat)
    p=len(S0)
    S=np.zeros((p,m))
    S[:,0]=S0
    Z = np.random.multivariate_normal(mean=v*Deltat,cov=sigma*Deltat,size=(m,1,1))
    Z = np.transpose(Z[:,0,0,:])

    for i in range(1,m):
        S[:,i:i+1]=np.exp(np.log(S[:,i-1:i])+Z[:,i-1:i])

    return S

def extract_data(path,start_date,end_date):
    data = pd.read_csv(path)
    data['Close/Last']=data['Close/Last'].str.replace('$','')
    for index,row in data.iterrows():
        row['Date'] = datetime.strptime(row['Date'],"%m/%d/%Y")
    data['Close/Last'] = data['Close/Last'].astype(float)
    data['Date'] = data['Date'].astype('datetime64[ns]')
    data = data.drop(columns=['Volume','Open','High','Low'])
    data = data[data['Date']>=start_date]
    data= data[data['Date']<=end_date]
    return data

w_1,w_2,w_3=1/3,1/3,1/3
holidays=['2022-09-05','2022-11-24','2022-12-26','2023-01-01','2023-01-16','2023-02-20','2023-04-07','2023-05-29','2023-07-04']
weekmask=[1,1,1,1,1,0,0]
start_date = datetime.strptime('2022-08-19',"%Y-%m-%d")
end_date = datetime.strptime('2022-10-24',"%Y-%m-%d")
maturity_date = datetime.strptime('2023-08-22',"%Y-%m-%d")
trading_days = np.busday_count(start_date.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

q2 = start_date+relativedelta(months=+6)
# check if q2 date is a weekday or holiday
while q2 in holidays or q2.weekday()==5 or q2.weekday()==6:
    if q2.weekday()==5:
        q2 = q2+relativedelta(days=+2)
    elif q2.weekday()==6:
        q2 = q2+relativedelta(days=+1)
    else:
        q2 = q2 +relativedelta(days=+1)
q2_to_maturity = np.busday_count(q2.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

q3 = start_date+relativedelta(months=+9)
# Check if q3 date is a weekday or holiday
while q3 in holidays or q3.weekday()==5 or q3.weekday()==6:
    if q3.weekday()==5:
        q3 = q3+relativedelta(days=+2)
    elif q2.weekday()==6:
        q3 = q3+relativedelta(days=+1)
    else:
        q3 = q3 +relativedelta(days=+1)
q3_to_maturity = np.busday_count(q3.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

cur_date = datetime.strptime(start_date.strftime('%m/%d/%Y'),'%m/%d/%Y')
end_date = datetime.strptime(end_date.strftime('%m/%d/%Y'),'%m/%d/%Y')
# q2 = q2.strftime('%m/%d/%Y')
# q3 = q3.strftime('%m/%d/%Y')
predicted_option_price = []
expected_payoff_maturity = []

while cur_date<=end_date:

    hist_start = cur_date - timedelta(days=365)

    aapl = extract_data('./data/24-10-2022/aapl.csv',hist_start,cur_date).rename(columns={'Close/Last':'AAPL'})
    amzn = extract_data('./data/24-10-2022/amzn.csv',hist_start,cur_date).rename(columns={'Close/Last':'AMZN'})
    googl = extract_data('./data/24-10-2022/googl.csv',hist_start,cur_date).rename(columns={'Close/Last':'GOOGL'})
    temp_df = aapl.merge(amzn,on=['Date'])
    AAG = temp_df.merge(googl,on=['Date'])
    n0 = len(AAG)
    AAGprices = np.array(AAG.drop(columns=['Date']))
    AAGlogprices = np.log(AAGprices)
    AAGlogreturns = AAGlogprices[:n0-1,:] - AAGlogprices[1:,:]

    v = np.mean(AAGlogreturns,axis=0)
    sigma = np.cov(AAGlogreturns,rowvar=False)
    Nsim=100
    T=trading_days
    dt=1
    m=int(T/dt)

    S0=AAGprices[0,:]
    S1=np.zeros((Nsim,m))
    S2=np.zeros((Nsim,m))
    S3=np.zeros((Nsim,m))
    random.seed(4518)

    for i in range(1,Nsim+1):
        S=SimMultiGBM(S0,v,sigma,dt,T)
        S1[i-1:i,:] = S[0:1,:]
        S2[i-1:i,:] = S[1:2,:]
        S3[i-1:i,:] = S[2:3,:]

    payoff_maturity = []
    payoff_q2 = []
    payoff_q3 = []
    print(S1.shape)
    for i in range(0,Nsim):
        payoff_maturity.append(payoff_function.payoff_maturity(aapl=S1[i,:],amzn=S2[i,:],googl=S3[i,:]))

        # if cur_date<q2:
        #     payoff_q2.append(payoff_function.payoff_quarterly(S1[i,:len(S1[i,:])-q2_to_maturity],S2[i,:len(S2[i,:])-q2_to_maturity],S3[i,:len(S3[i,:])-q2_to_maturity],2))

        # if cur_date<q3:
        #     payoff_q3.append(payoff_function.payoff_quarterly(S1[i,:len(S1[i,:])-q3_to_maturity],S2[i,:len(S2[i,:])-q3_to_maturity],S3[i,:len(S3[i,:])-q3_to_maturity],3))
    
    cur_expected_payoff = np.mean(payoff_maturity)
    r=0.04716
    expected_payoff_maturity.append(cur_expected_payoff)
    # option_price = np.exp(-r*1)*cur_expected_payoff*w_1 + np.exp(-r/2)*np.mean(payoff_q2)*w_2 + np.exp(-r*3/4)*np.mean(payoff_q3)*w_3
    option_price = np.exp(-r*1)*cur_expected_payoff
    predicted_option_price.append(option_price)

    cur_date = cur_date + relativedelta(days=+1)
    trading_days-=1

print(predicted_option_price)
print(expected_payoff_maturity)
plt.plot(predicted_option_price)
plt.savefig('test.png')

