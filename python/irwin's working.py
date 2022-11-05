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
counter = 0

while date_to_predict <= end_date:
    if counter == 1:
        break

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
    Nsim = 100000
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
        S, Z = SimMultiGBM(S0, v, sigma, dt, T)
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

    pathMatrix = np.zeros((Nsim, 3, m+1))
    for i in range(Nsim):
        pathMatrix[i][0] = S1[i]
        pathMatrix[i][1] = S2[i]
        pathMatrix[i][2] = S3[i]
    # print(pathMatrix)

    q2_index = total_trading_days - q2_to_maturity
    q3_index = total_trading_days - q3_to_maturity

    print(min(pathMatrix[0][0][0:q2_index]))
    print(pathMatrix[0][0][q2_index])

    payoff_matrix = np.zeros((Nsim, 3))
    payoff_maturity = []

    for i in range(0, Nsim):
        maturity_payoff_ = maturity_payoff(aapl=S1[i], amzn=S2[i], googl=S3[i])
        payoff_maturity.append(maturity_payoff_)
        payoff_matrix[i][2] = maturity_payoff_
        if i < 2:
            print("Payoff maturity")
            print(payoff_maturity)
            print("-------")

    discounted_payoff_q3 = np.zeros(Nsim)
    q3_path_included = []
    q3_path_excluded = []
    q3_features = np.zeros((Nsim, 10))  # S1, S2, S3, S1^2, S2^2, S3^2, S1S2, S1S3, S2S3, current_worst_performing stock price
    discounted_payoff_q3_ = discounted_quarterly_payoff(next_q_payoff=payoff_maturity[i], q_to_next_q=q3_to_maturity, interest_rate=r)
    for i in range(0, Nsim):
        if S1[i][q3_index] < aapl_initial or S2[i][q3_index] < amzn_initial or S3[i][q3_index] < google_initial: # no early redemption, skip path
            # print(f"apple: {S1[q3_index]}, amzn: {S2[q3_index]}, google: {S3[q3_index]}\n")
            # print(f"No early redemption opportunity for path {i+1}\n")
            q3_path_excluded.append(i)
            continue
        discounted_payoff_q3[i] = discounted_payoff_q3_ # there is early redemption opportunity
        # discounted payoff with it's associated features
        q3_features[i] = get_features(S1[i], S2[i], S3[i], q3_index, Nsim)
        q3_path_included.append(i)

    # linear model: E[Y|S1=S1_q3, S2=S2_q3, S3=S3_q3] = a+bS1+cS2+dS3+eS1^2+fS2^2+gS3^2+hS1S2+iS1S3+jS2S3
    print("q3 features")
    print(q3_features)
    print("-------")
    # print("discounted q3_payoff")
    # print(discounted_payoff_q3)
    # print("-------")
    print("q3 path included")
    print(q3_path_included)
    print("-------")
    # print("path excluded")
    # print(q3_path_excluded)
    # print("-------")

    # Preparing dataset for regression
    X = pd.DataFrame(data=q3_features, columns=["S1", "S2", "S3", "S1sq", "S2sq", "S3sq", "S1S2", "S1S3", "S2S3", "Worst_Stock"])
    y = pd.DataFrame(data=discounted_payoff_q3, columns=["Discounted Payoff"])
    X = X.drop(q3_path_excluded)
    y = y.drop(q3_path_excluded)
    # print("X")
    # print(X)
    # print("-------")
    # print("y")
    # print(y)
    # print("-------")
    q3_reg = LinearRegression().fit(X, y)
    print(f"regression coefficient: {q3_reg.coef_}, regression intercept: {q3_reg.intercept_}")

    continuation_payoff = []
    q3_exercise_payoff = []
    if len(q3_path_included) > 0:
        for valid_path in q3_path_included:
            continuation_payoff_ = q3_reg.predict([q3_features[valid_path]])
            continuation_payoff.append(continuation_payoff_[0])

            q3_exercise_payoff_ = quarterly_payoff(S1[valid_path][0:q3_index+1], S2[valid_path][0:q3_index+1], S3[valid_path][0:q3_index+1], 3)
            q3_exercise_payoff.append(q3_exercise_payoff_)

    print("q3 continuation payoff")
    print(continuation_payoff)
    print("-------")
    print("q3 exercise payoff")
    print(q3_exercise_payoff)
    print("-------")

    for i in range(len(q3_path_excluded)):
        path = q3_path_excluded[i]
        payoff_matrix[path][1] = discounted_payoff_q3_

    for i in range(len(q3_path_included)): # updating ITM paths
        path = q3_path_included[i]
        if continuation_payoff[i] < q3_exercise_payoff[i]: # payoff for continuing is smaller than exercising at q3, should exercise right
            payoff_matrix[path][1] = q3_exercise_payoff[i]
        else:
            payoff_matrix[path][1] = discounted_payoff_q3_

    discounted_payoff_q2 = np.zeros(Nsim)
    q3_payoff = np.transpose(payoff_matrix)[1]
    q2_path_included = []
    q2_path_excluded = []
    q2_features = np.zeros((Nsim, 10))  # S1, S2, S3, S1^2, S2^2, S3^2, S1S2, S1S3, S2S3, current_worst_performing stock price
    for i in range(0, Nsim):
        if S1[i][q2_index] < aapl_initial or S2[i][q2_index] < amzn_initial or S3[i][q2_index] < google_initial: # no early redemption, skip path
            # print(f"apple: {S1[q2_index]}, amzn: {S2[q2_index]}, google: {S3[q2_index]}\n")
            # print(f"No early redemption opportunity for path {i+1}\n")
            q2_path_excluded.append(i)
            continue
        discounted_payoff_q2_ = discounted_quarterly_payoff(next_q_payoff=q3_payoff[i],
                                                            q_to_next_q=q2_to_maturity - q3_to_maturity,
                                                            interest_rate=r)
        discounted_payoff_q2[i] = discounted_payoff_q2_ # there is early redemption opportunity
        # discounted payoff with it's associated features
        q2_features[i] = get_features(S1[i], S2[i], S3[i], q2_index, Nsim)
        q2_path_included.append(i)

    # linear model: E[Y|S1=S1_q2, S2=S2_q2, S3=S3_q2] = a+bS1+cS2+dS3+eS1^2+fS2^2+gS3^2+hS1S2+iS1S3+jS2S3
    print("q2 features")
    print(q2_features)
    print("-------")
    # print("discounted q2_payoff")
    # print(discounted_payoff_q2)
    # print("-------")
    print("q2 path included")
    print(q2_path_included)
    print("-------")
    # print("path excluded")
    # print(q2_path_excluded)
    # print("-------")

    # Preparing dataset for regression
    X = pd.DataFrame(data=q2_features, columns=["S1", "S2", "S3", "S1sq", "S2sq", "S3sq", "S1S2", "S1S3", "S2S3", "Worst_Stock"])
    y = pd.DataFrame(data=discounted_payoff_q2, columns=["Discounted Payoff"])
    X = X.drop(q2_path_excluded)
    y = y.drop(q2_path_excluded)
    # print("X")
    # print(X)
    # print("-------")
    # print("y")
    # print(y)
    # print("-------")
    q2_reg = LinearRegression().fit(X, y)
    print(f"regression coefficient: {q2_reg.coef_}, regression intercept: {q2_reg.intercept_}")

    continuation_payoff = []
    q2_exercise_payoff = []
    if len(q2_path_included) > 0:
        for valid_path in q2_path_included:
            continuation_payoff_ = q2_reg.predict([q2_features[valid_path]])
            continuation_payoff.append(continuation_payoff_[0])

            q2_exercise_payoff_ = quarterly_payoff(S1[valid_path][0:q2_index+1], S2[valid_path][0:q2_index+1], S3[valid_path][0:q2_index+1], 2)
            q2_exercise_payoff.append(q2_exercise_payoff_)

    print("q2 continuation payoff")
    print(continuation_payoff)
    print("-------")
    print("q2 exercise payoff")
    print(q2_exercise_payoff)
    print("-------")

    for i in range(len(q2_path_excluded)): # updating OTM paths
        path = q2_path_excluded[i]
        payoff_matrix[path][0] = discounted_payoff_q2_

    for i in range(len(q2_path_included)):
        path = q2_path_included[i]
        if continuation_payoff[i] < q2_exercise_payoff[i]: # payoff for continuing is smaller than exercising at q2, should exercise right
            payoff_matrix[path][0] = q2_exercise_payoff[i]
        else:
            payoff_matrix[path][0] = discounted_payoff_q2_

    print("Payoff matrix")
    print(payoff_matrix)
    print("-------")
    print("Derviative Price")
    payoff_matrix_t = np.transpose(payoff_matrix)
    print(np.mean(payoff_matrix_t[0]))
    print("-------")
        # if cur_date<q2:
        #     payoff_q2.append(quarterly_payoff(S1[i,:len(S1[i,:])-q2_to_maturity],S2[i,:len(S2[i,:])-q2_to_maturity],S3[i,:len(S3[i,:])-q2_to_maturity],2))

        # if cur_date<q3:
        #     payoff_q3.append(quarterly_payoff(S1[i,:len(S1[i,:])-q3_to_maturity],S2[i,:len(S2[i,:])-q3_to_maturity],S3[i,:len(S3[i,:])-q3_to_maturity],3))

    cur_expected_payoff = np.mean(payoff_maturity)
    expected_payoff_maturity.append(cur_expected_payoff)

    # TODO Apply regression to find the coefficient for each payoff
    # option_price = np.exp(-r*(trading_days_to_predict/total_trading_days))*cur_expected_payoff*w_1 + np.exp(-r/2)*np.mean(payoff_q2)*w_2 + np.exp(-r*3/4)*np.mean(payoff_q3)*w_3
    option_price = np.exp(-r * (trading_days_to_simulate / total_trading_days)) * cur_expected_payoff
    predicted_option_price.append(option_price)

    date_to_predict += relativedelta(days=+1)
    trading_days_to_simulate -= 1
    hist_end += relativedelta(days=+1)
    counter+=1
