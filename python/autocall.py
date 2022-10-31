from datetime import timedelta
from utils.counting_days_function import days
from utils.extract_data_function import extract_data
from utils.payoff_function import maturity_payoff, discounted_quarterly_payoff, quarterly_payoff
from utils.simulation_function import SimMultiGBM
import numpy as np
from dateutil.relativedelta import relativedelta
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.features import get_features
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

aapl_barrier = 85.760
amzn_barrier = 69.115
googl_barrier = 58.605
aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21

date_to_predict, hist_end, end_date, q2_to_maturity, q3_to_maturity, q2, q3, total_trading_days, holidays = days(
    latest_price_date='2022-10-24')

print(f"date_to_predict: {date_to_predict}")
print(f"hist_end: {hist_end}")
print(f"end_date: {end_date}")
print(f"q2_to_maturity: {q2_to_maturity}")
print(f"q3_to_maturity: {q3_to_maturity}")
print(f"q2: {q2}")
print(f"q3: {q3}")
print(f"total_trading_days: {total_trading_days}")
print(f"holidays: {holidays}")
trading_days_to_simulate = total_trading_days

predicted_option_price = []
expected_payoff_maturity = []
# counter = 0

while date_to_predict <= end_date:
    # if counter == 5:
    #     break

    if date_to_predict in holidays or date_to_predict.weekday() == 5 or date_to_predict.weekday() == 6:
        date_to_predict += relativedelta(days=+1)
        trading_days_to_simulate -= 1
        hist_end += relativedelta(days=+1)
        continue

    hist_start = hist_end - timedelta(days=365)

    aapl = extract_data('/Users/irwinding/Desktop/MH4518/CZ4518-Simulation-in-Finance/data/24-10-2022/aapl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AAPL'})
    amzn = extract_data('/Users/irwinding/Desktop/MH4518/CZ4518-Simulation-in-Finance/data/24-10-2022/amzn.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AMZN'})
    googl = extract_data('/Users/irwinding/Desktop/MH4518/CZ4518-Simulation-in-Finance/data/24-10-2022/googl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'GOOGL'})
    temp_df = aapl.merge(amzn, on=['Date'])
    AAG = temp_df.merge(googl, on=['Date'])
    n0 = len(AAG)
    dates = AAG['Date']
    AAGprices = np.array(AAG.drop(columns=['Date']))
    AAGlogprices = np.log(AAGprices)
    AAGlogreturns = AAGlogprices[:n0 - 1, :] - AAGlogprices[1:, :]

    v = np.mean(AAGlogreturns, axis=0)
    sigma = np.cov(AAGlogreturns, rowvar=False)
    Nsim = 10000
    T = trading_days_to_simulate
    dt = 1
    m = int(T / dt)
    r = 0.04716
    print(f"trading_days_to_simulate: {trading_days_to_simulate}")
    # print(f"m: {m}")

    S0 = AAGprices[0, :]
    S1 = np.zeros((Nsim, m + 1))
    S2 = np.zeros((Nsim, m + 1))
    S3 = np.zeros((Nsim, m + 1))
    random.seed(4518)

    for i in range(1, Nsim + 1):
        S = SimMultiGBM(S0, v, sigma, dt, T)
        # print("S Matrix")
        # print(S)
        # print("-------")
        # print("S[0:1, :]")
        # print(S[0:1, :])
        # print("-------")
        # print(S[0])
        # S1[i - 1:i, :] = S[0:1, :]
        # S2[i - 1:i, :] = S[1:2, :]
        # S3[i - 1:i, :] = S[2:3, :]
        S1[i - 1] = S[0]
        S2[i - 1] = S[1]
        S3[i - 1] = S[2]

    q2_index = total_trading_days - q2_to_maturity
    q3_index = total_trading_days - q3_to_maturity

    payoff = []

    for i in range(Nsim):
        # check for early redemption opportunity at q2
        if S1[i][q2_index] > aapl_initial and S2[i][q2_index] > amzn_initial and S3[i][q2_index] > google_initial:
            payoff_ = quarterly_payoff(S1[i], S2[i], S3[i], 2)
            discounted_payoff_ = np.exp(-r * (T-q2_index)/total_trading_days) * payoff_
            payoff.append(discounted_payoff_)
            continue

        # check for early redemption opportunity at q3
        elif S1[i][q3_index] > aapl_initial and S2[i][q3_index] > amzn_initial and S3[i][q3_index] > google_initial:
            payoff_ = quarterly_payoff(S1[i], S2[i], S3[i], 3)
            discounted_payoff_ = np.exp(-r * (T-q3_index)/total_trading_days) * payoff_
            payoff.append(discounted_payoff_)
            continue

        # no early redemption opportunities
        else:
            payoff_ = maturity_payoff(S1[i], S2[i], S3[i])
            discounted_payoff_ = np.exp(-r * T/total_trading_days) * payoff_
            payoff.append(discounted_payoff_)

    option_price = np.mean(payoff)
    expected_payoff_maturity.append(option_price)
    predicted_option_price.append(option_price)
    print(f"Derivative Price for {date_to_predict}")
    print(option_price)
    print("-------")
    # print("S1")
    # print(S1)
    # print("-------")
    # print("S2")
    # print(S2)
    # print("-------")
    # print("S3")
    # print(S3)
    # print("-------")

    # S1 = EMS(S1,dt,r)
    # # S2 = EMS(S2,dt,r)
    # S3 = EMS(S3,dt,r)

    # cur_expected_payoff = np.mean(payoff_maturity)
    # expected_payoff_maturity.append(cur_expected_payoff)

    # TODO Apply regression to find the coefficient for each payoff
    # option_price = np.exp(-r*(trading_days_to_predict/total_trading_days))*cur_expected_payoff*w_1 + np.exp(-r/2)*np.mean(payoff_q2)*w_2 + np.exp(-r*3/4)*np.mean(payoff_q3)*w_3
    date_to_predict += relativedelta(days=+1)
    trading_days_to_simulate -= 1
    hist_end += relativedelta(days=+1)
    # counter+=1

plt_1 = plt.figure(figsize=(30,10))
plt.plot(predicted_option_price)

plt.savefig('/Users/irwinding/Desktop/MH4518/CZ4518-Simulation-in-Finance/results/autocall_MC.png')