from datetime import timedelta
from utils.counting_days_function import days
from utils.extract_data_function import extract_data
from utils.payoff_function import calculate_option_price
from utils.simulation_function import SimMultiGBM
from utils.ems_correction import EMSCorrection
from utils.evaluation import evaluate_option_price
from utils.simulation_function import SimMultiGBMpmh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import random

import json
import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

experiment_details = {
    'Nsim': 10000,
    'latest_price_date': '2022-10-24',
    'variance_reduction': None,
    'GBM': 'multivariate',
    'r': 0.045
}

aapl_barrier = 85.760
amzn_barrier = 69.115
googl_barrier = 58.605
aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21


date_to_predict, hist_end, end_date, q2_to_maturity, q3_to_maturity, q2, q3, total_trading_days, alternative_option_ttm, holidays = days(
    latest_price_date=experiment_details['latest_price_date'])

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
no_q2_payoff_arr = []
no_q3_payoff_arr = []
no_barrier_event_arr = []
delta1 = []
delta2 = []
delta3 = []
pmh_indexes = [
    [0, 4, 7],
    [2, 4, 7],
    [1, 3, 7],
    [1, 5, 7],
    [1, 4, 6],
    [1, 4, 8]
]
# S in line174 is framed as such: appl_initial-h, appl_initial, appl_initial+h, amzn_initial-h, amzn_initial, amzn_initial+h,...
counter = 0

while date_to_predict <= end_date:
    # if counter == 1:
    #     break

    if date_to_predict in holidays or date_to_predict.weekday() == 5 or date_to_predict.weekday() == 6:
        date_to_predict += relativedelta(days=+1)
        trading_days_to_simulate -= 1
        hist_end += relativedelta(days=+1)
        continue

    hist_start = hist_end - timedelta(days=365)

    aapl = extract_data('../data/24-10-2022/aapl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AAPL'})
    amzn = extract_data('../data/24-10-2022/amzn.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AMZN'})
    googl = extract_data('../data/24-10-2022/googl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'GOOGL'})
    temp_df = aapl.merge(amzn, on=['Date'])
    AAG = temp_df.merge(googl, on=['Date'])
    n0 = len(AAG)
    dates = AAG['Date']
    AAGprices = np.array(AAG.drop(columns=['Date']))
    AAGlogprices = np.log(AAGprices)
    AAGlogreturns = AAGlogprices[:n0 - 1, :] - AAGlogprices[1:, :]

    v = np.mean(AAGlogreturns, axis=0)
    sigma = np.cov(AAGlogreturns, rowvar=False)
    Nsim = experiment_details['Nsim']
    T = trading_days_to_simulate
    dt = 1
    m = int(T / dt)
    r = experiment_details['r']
    print(f"trading_days_to_simulate: {trading_days_to_simulate}")
    # print(f"m: {m}")

    S0 = AAGprices[0, :]
    sim_aapl = np.zeros((Nsim, m + 1))
    sim_amzn = np.zeros((Nsim, m + 1))
    sim_googl = np.zeros((Nsim, m + 1))
    sim_aapl_star = np.zeros((Nsim, m + 1))
    sim_amzn_star = np.zeros((Nsim, m + 1))
    sim_googl_star = np.zeros((Nsim, m + 1))
    random.seed(4518)
    print("S0")
    print(S0)
    print("-------")

    Z_matrix = []

    for i in range(1, Nsim + 1):
        S, Z = SimMultiGBM(S0, v, sigma, dt, T)
        sim_aapl[i - 1] = S[0]
        sim_amzn[i - 1] = S[1]
        sim_googl[i - 1] = S[2]
        Z_matrix.append(Z) # store the Z used to simulate the GBM for derivative pricing


    # sim_aapl_star = EMSCorrection(sim_aapl, Nsim, r, dt, T)
    # sim_amzn_star = EMSCorrection(sim_amzn, Nsim, r, dt, T)
    # sim_googl_star = EMSCorrection(sim_googl, Nsim, r, dt, T)
    # print("apple star")
    # print(sim_aapl_star)
    # print("-------")
    # print("apple")
    # print(sim_aapl)
    # print("-------")
    # plt.figure(figsize=(30, 10))
    # plt.plot(sim_aapl, color='red')
    # plt.legend(loc='upper left')
    # folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Path(f'../results/{folder}').mkdir(parents=True, exist_ok=True)
    # plt.savefig(f'../results/{folder}/stockpricepath_{Nsim}.png')
    #
    # plt.figure(figsize=(30, 10))
    # plt.plot(sim_aapl_star, color='blue')
    # plt.legend(loc='upper left')
    # folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Path(f'../results/{folder}').mkdir(parents=True, exist_ok=True)
    # plt.savefig(f'../results/{folder}/EMSstockpricepath_{Nsim}.png')


    q2_index = total_trading_days - q2_to_maturity if total_trading_days - q2_to_maturity>=0 else None
    q3_index = total_trading_days - q3_to_maturity if total_trading_days - q3_to_maturity>=0 else None

    option_prices = []
    delta = []
    payoff_pmh = np.zeros((Nsim, 3))
    # no_q2_autocalls = 0
    # no_q3_autocalls = 0
    # no_barrier_event = 0
    for i in range(Nsim):
        price = calculate_option_price(
                aapl=sim_aapl[i],
                amzn=sim_amzn[i],
                googl=sim_googl[i],
                T=trading_days_to_simulate,
                total_trading_days=total_trading_days,
                r=r,
                q2_index=q2_index,
                q3_index=q3_index
            )
        option_prices.append(
            price
        )

        S, h = SimMultiGBMpmh(S0, v, sigma, dt, T, Z_matrix[i])
        payoff_pmh_ = []
        for stock_id in range(len(S0)):
            mh_index = pmh_indexes[2*stock_id]
            ph_index = pmh_indexes[2*stock_id+1]

            payoff_ph = calculate_option_price(
                aapl=S[ph_index[0]], amzn=S[ph_index[1]], googl=S[ph_index[2]],
                T=trading_days_to_simulate, total_trading_days=total_trading_days,
                r=r, q2_index=q2_index, q3_index=q3_index
            )
            payoff_mh = calculate_option_price(
                aapl=S[mh_index[0]], amzn=S[mh_index[1]], googl=S[mh_index[2]],
                T=trading_days_to_simulate, total_trading_days=total_trading_days,
                r=r, q2_index=q2_index,q3_index=q3_index
            )
            payoff_pmh_temp = (payoff_ph - payoff_mh) / 2 * h[stock_id]
            payoff_pmh_.append(payoff_pmh_temp)
            # print("payoff_pmh_")
            # print(payoff_pmh_)
            # print("-------")

            # if i==0:
            #     print(f"S[ph_index[0]], ph_index: {ph_index[0]}")
            #     print(S[ph_index[0]][0:10])
            #     print("--------")
            #     print(f"S[ph_index[1]], ph_index: {ph_index[1]}")
            #     print(S[ph_index[1]][0:10])
            #     print("--------")
            #     print(f"S[ph_index[2]], ph_index: {ph_index[2]}")
            #     print(S[ph_index[2]][0:10])
            #     print("--------")
            #     print(f"S[mh_index[0]], mh_index: {mh_index[0]}")
            #     print(S[mh_index[0]][0:10])
            #     print("--------")
            #     print(f"S[mh_index[1]], mh_index: {mh_index[1]}")
            #     print(S[mh_index[1]][0:10])
            #     print("--------")
            #     print(f"S[mh_index[2]], mh_index: {mh_index[2]}")
            #     print(S[mh_index[2]][0:10])
            #     print("--------")
            #     print(f"ph_index: {ph_index}")
            #     print(f"mh_index: {mh_index}")
            #     print(f"Payoff_ph for stock {stock_id+1}")
            #     print(payoff_ph)
            #     print("-------")
            #     print(f"Payoff_mh for stock {stock_id+1}")
            #     print(payoff_mh)
            #     print("-------")
            #     print("payoff_pmh_temp")
            #     print(payoff_pmh_temp)
            #     print("-------")

        payoff_pmh[i] = payoff_pmh_

        # if q2_autocall:
        #     no_q2_autocalls+=1
        # elif q3_autocall:
        #     no_q3_autocalls+=1
        # elif barrier_event:
        #     no_barrier_event+=1

    # print("payoff_pmh")
    # print(payoff_pmh)
    # print("-------")
    payoff_pmh = np.transpose(payoff_pmh)
    # print("payoff_pmh transpose")
    # print(payoff_pmh)
    # print("-------")
    print("payoff_pmh")
    print(payoff_pmh)
    print("-------")
    delta1.append(np.mean(payoff_pmh[0]))
    delta2.append(np.mean(payoff_pmh[1]))
    delta3.append(np.mean(payoff_pmh[2]))
    for i in range(len(S0)):
        print(f"Delta for stock {i+1}: {np.mean(payoff_pmh[i])}")
    option_price = np.mean(option_prices)
    expected_payoff_maturity.append(option_price)
    predicted_option_price.append({'date':date_to_predict, 'predicted': option_price})
    print(f"Derivative Price for {date_to_predict}")
    print(option_price)
    # print(f"Number of q2 early redemption for {date_to_predict}, % = {100*no_q2_autocalls/Nsim}")
    # print(no_q2_autocalls)
    # print(f"Number of q3 early redemption for {date_to_predict}, % = {100*no_q3_autocalls/Nsim}")
    # print(no_q3_autocalls)
    # print(f"Number of barrier_event for {date_to_predict}, % = {100*no_barrier_event/Nsim}")
    # print(no_barrier_event)
    # no_q2_payoff_arr.append(no_q2_autocalls)
    # no_q3_payoff_arr.append(no_q3_autocalls)
    # no_barrier_event_arr.append(no_barrier_event)

    # sim_aapl = EMS(sim_aapl,dt,r)
    # # sim_amzn = EMS(sim_amzn,dt,r)
    # sim_googl = EMS(sim_googl,dt,r)

    # cur_expected_payoff = np.mean(payoff_maturity)
    # expected_payoff_maturity.append(cur_expected_payoff)

    date_to_predict += relativedelta(days=+1)
    trading_days_to_simulate -= 1
    hist_end += relativedelta(days=+1)
    counter+=1

plt.figure(figsize=(30, 10))
plt.plot(delta1, label='delta 1', color='blue')
plt.plot(delta2, label='delta 2', color='red')
plt.plot(delta3, label='delta 3', color='green')
plt.legend(loc='upper left')
plt.savefig(f'../results/delta_movements_{Nsim}.png')


predicted_option_price = pd.DataFrame(predicted_option_price)
predicted_option_price['date'] = pd.to_datetime(predicted_option_price['date'])
# Scale back to 100%
predicted_option_price['predicted'] = predicted_option_price['predicted']/10
actual_option_price = pd.read_csv('../data/derivative_01_11_22.csv')
actual_option_price['date'] = pd.to_datetime(actual_option_price['date'], format='%Y-%m-%d')
combined = predicted_option_price.merge(actual_option_price, left_on=['date'], right_on=['date'], validate='one_to_one')
combined = combined.set_index('date')


# evaluate_option_price(combined['predicted'], combined['value'], experiment_details)