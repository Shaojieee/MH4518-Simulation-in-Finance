from calendar import month, week
from datetime import timedelta, datetime
from tkinter import NS
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from utils.payoff_function import calculate_option_price, get_payoff_pmh
from utils.evaluation import evaluate_option_price
from utils.simulation_function import SimMultiGBMAV,SimMultiGBM,SimMultiGBMpmh
from utils.extract_data_function import extract_data
from utils.counting_days_function import days
from utils.ems_correction import EMSCorrection
from utils.calculate_implied_volatility import calculate_cov_matrix
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")

experiment_details = {
    'Nsim': 1000,
    'latest_price_date': '2022-10-21',
    'variance_reduction': True,
    'GBM': 'multivariate',
    'r': 0.0326,
    'IV': True,
    'min_sigma': 0.0001,
    'max_sigma': 5,
    'step': 0.00001,
    'error': 0.25,
    'EMS': True
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

if experiment_details['IV']:
    aapl_call_option_df = pd.read_csv('../Bloomberg_Data/aapl_call.csv')
    aapl_call_option_df['Date'] = pd.to_datetime(aapl_call_option_df['Date'])
    amzn_call_option_df = pd.read_csv('../Bloomberg_Data/amzn_call.csv')
    amzn_call_option_df['Date'] = pd.to_datetime(amzn_call_option_df['Date'])
    googl_call_option_df = pd.read_csv('../Bloomberg_Data/googl_call.csv')
    googl_call_option_df['Date'] = pd.to_datetime(googl_call_option_df['Date'])

    min_sigma = experiment_details['min_sigma']
    max_sigma = experiment_details['max_sigma']
    error = experiment_details['error']
    step = experiment_details['step']
    sigma_hat_list = np.arange(min_sigma, max_sigma + step, step)
    right = len(sigma_hat_list) - 1
    left = 0

predicted_option_price = []
expected_payoff_list = []
aapl_IV_list = []
amzn_IV_list = []
googl_IV_list = []
delta1 = []
delta2 = []
delta3 = []
gamma1 = []
gamma2 = []
gamma3 = []

counter = 0

while date_to_predict <= end_date:
    # if counter == 1:
    #     break

    if datetime.strftime(date_to_predict,
                         "%Y-%m-%d") in holidays or date_to_predict.weekday() == 5 or date_to_predict.weekday() == 6:
        date_to_predict += relativedelta(days=+1)
        trading_days_to_simulate -= 1
        hist_end += relativedelta(days=+1)
        alternative_option_ttm -= 1
        continue

    hist_start = hist_end - timedelta(days=365)

    aapl = extract_data('../data/24-10-2022/aapl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AAPL'})
    amzn = extract_data('../data/24-10-2022/amzn.csv', hist_start, hist_end).rename(columns={'Close/Last': 'AMZN'})
    googl = extract_data('../data/24-10-2022/googl.csv', hist_start, hist_end).rename(columns={'Close/Last': 'GOOGL'})
    temp_df = aapl.merge(amzn, on=['Date'])
    AAG = temp_df.merge(googl, on=['Date'])
    n0 = len(AAG)
    AAGprices = np.array(AAG.drop(columns=['Date']))
    AAGlogprices = np.log(AAGprices)
    AAGlogreturns = AAGlogprices[:n0 - 1, :] - AAGlogprices[1:, :]

    v = np.mean(AAGlogreturns, axis=0)
    Nsim = experiment_details['Nsim']
    T = trading_days_to_simulate
    dt = 1
    m = int(T / dt)
    r = experiment_details['r']

    # If IV setting is true covariance matrix is calculated from implied volatility of individual stocks  options market price
    if experiment_details['IV']:
        sigma, aapl_IV, amzn_IV, googl_IV = calculate_cov_matrix(aapl_call_option_df, amzn_call_option_df,
                                                                 googl_call_option_df, r, alternative_option_ttm,
                                                                 sigma_hat_list, left, right, error, date_to_predict,
                                                                 AAGlogreturns)
        aapl_IV_list.append(aapl_IV)
        amzn_IV_list.append(amzn_IV)
        googl_IV_list.append(googl_IV)
    else:
        sigma = np.cov(AAGlogreturns, rowvar=False)

    print(f"trading_days_to_simulate: {trading_days_to_simulate}")

    S0 = AAGprices[0, :]
    sim_aapl = []
    sim_aapl_mh = []
    sim_aapl_ph = []
    sim_amzn = []
    sim_amzn_mh = []
    sim_amzn_ph = []
    sim_googl = []
    sim_googl_mh = []
    sim_googl_ph = []

    Z_matrix = []
    random.seed(4518)

    # Antithetic Variate reduction technique is applied if Variance Reduction is set to true
    if experiment_details['variance_reduction']:
        for i in range(1, int(Nsim / 2) + 1):
            S, Stilde, Z = SimMultiGBMAV(S0, v, sigma, dt, T)
            sim_aapl.append(S[0])
            sim_aapl.append(Stilde[0])
            sim_amzn.append(S[1])
            sim_amzn.append(Stilde[1])
            sim_googl.append(S[2])
            sim_googl.append(Stilde[2])
            S_pmh, Stilde_pmh = SimMultiGBMpmh(S0, v, sigma, dt, T, Z, variance_reduction=True)
            sim_aapl_mh.append(S_pmh[0])
            sim_aapl_mh.append(Stilde_pmh[0])
            sim_aapl_ph.append(S_pmh[2])
            sim_aapl_ph.append(Stilde_pmh[2])
            sim_amzn_mh.append(S_pmh[3])
            sim_amzn_mh.append(Stilde_pmh[3])
            sim_amzn_ph.append(S_pmh[5])
            sim_amzn_ph.append(Stilde_pmh[5])
            sim_googl_mh.append(S_pmh[6])
            sim_googl_mh.append(Stilde_pmh[6])
            sim_googl_ph.append(S_pmh[8])
            sim_googl_ph.append(Stilde_pmh[8])

    else:
        for i in range(1, Nsim + 1):
            S, Z = SimMultiGBM(S0, v, sigma, dt, T)
            sim_aapl.append(S[0])
            sim_amzn.append(S[1])
            sim_googl.append(S[2])
            S_pmh = SimMultiGBMpmh(S0, v, sigma, dt, T, Z, variance_reduction=False)
            sim_aapl_mh.append(S_pmh[0])
            sim_aapl_ph.append(S_pmh[2])
            sim_amzn_mh.append(S_pmh[3])
            sim_amzn_ph.append(S_pmh[5])
            sim_googl_mh.append(S_pmh[6])
            sim_googl_ph.append(S_pmh[8])

    if experiment_details['EMS']:
        sim_aapl = EMSCorrection(sim_aapl, Nsim, r, dt, T)
        sim_aapl_mh = EMSCorrection(sim_aapl_mh, Nsim, r, dt, T)
        sim_aapl_ph = EMSCorrection(sim_aapl_ph, Nsim, r, dt, T)
        sim_amzn = EMSCorrection(sim_amzn, Nsim, r, dt, T)
        sim_amzn_mh = EMSCorrection(sim_amzn_mh, Nsim, r, dt, T)
        sim_amzn_ph = EMSCorrection(sim_amzn_ph, Nsim, r, dt, T)
        sim_googl = EMSCorrection(sim_googl, Nsim, r, dt, T)
        sim_googl_mh = EMSCorrection(sim_googl_mh, Nsim, r, dt, T)
        sim_googl_ph = EMSCorrection(sim_googl_ph, Nsim, r, dt, T)

    q2_index = total_trading_days - q2_to_maturity if total_trading_days - q2_to_maturity >= 0 else None
    q3_index = total_trading_days - q3_to_maturity if total_trading_days - q3_to_maturity >= 0 else None

    option_prices = []
    payoff_list = []
    delta_payoff_pmh = []
    gamma_payoff_pmh = []
    for i in range(Nsim):
        option_price, payoff = calculate_option_price(
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
            option_price
        )
        payoff_list.append(payoff)
        delta_payoff_pmh_, gamma_payoff_pmh_ = get_payoff_pmh(S0, sim_aapl_mh[i], sim_aapl[i], sim_aapl_ph[i],
                                                              sim_amzn_mh[i], sim_amzn[i], sim_amzn_ph[i], sim_googl_mh[i],
                                                              sim_googl[i], sim_googl_ph[i],
                                                              trading_days_to_simulate=trading_days_to_simulate,
                                                              total_trading_days=total_trading_days, r=r, q2_index=q2_index,
                                                              q3_index=q3_index)
        # print(f"payoff_pmh_ for simulation {i + 1}")
        # print(payoff_pmh_)
        # print("-------")
        delta_payoff_pmh.append(delta_payoff_pmh_)
        gamma_payoff_pmh.append(gamma_payoff_pmh_)

    expected_payoff_list.append(np.mean(payoff_list))
    option_price = np.mean(option_prices)
    predicted_option_price.append({'date': date_to_predict, 'predicted': option_price})
    print(f"Derivative Price for {date_to_predict}")
    print(option_price)

    delta_payoff_pmh = np.transpose(delta_payoff_pmh)
    gamma_payoff_pmh = np.transpose(gamma_payoff_pmh)
    print("delta_payoff_pmh")
    print(delta_payoff_pmh)
    print("-------")
    print("gamma_payoff_pmh")
    print(gamma_payoff_pmh)
    print("-------")
    delta1.append(np.mean(delta_payoff_pmh[0]))
    delta2.append(np.mean(delta_payoff_pmh[1]))
    delta3.append(np.mean(delta_payoff_pmh[2]))
    gamma1.append(np.mean(gamma_payoff_pmh[0]))
    gamma2.append(np.mean(gamma_payoff_pmh[1]))
    gamma3.append(np.mean(gamma_payoff_pmh[2]))
    for i in range(len(S0)):
        print(f"Delta for stock {i+1}: {np.mean(delta_payoff_pmh[i])}")
        print(f"Gamma for stock {i + 1}: {np.mean(gamma_payoff_pmh[i])}")

    date_to_predict += relativedelta(days=+1)
    trading_days_to_simulate -= 1
    hist_end += relativedelta(days=+1)
    alternative_option_ttm -= 1

    counter += 1

predicted_option_price = pd.DataFrame(predicted_option_price)
predicted_option_price['date'] = pd.to_datetime(predicted_option_price['date'])
# Scale back to 100%
predicted_option_price['predicted'] = predicted_option_price['predicted'] / 10
actual_option_price = pd.read_csv('../data/derivative_01_11_22.csv')
actual_option_price['date'] = pd.to_datetime(actual_option_price['date'], format='%Y-%m-%d')
combined = predicted_option_price.merge(actual_option_price, left_on=['date'], right_on=['date'], validate='one_to_one')
combined = combined.set_index('date')

plt.figure(figsize=(30, 10))
plt.plot(delta1, label='delta 1', color='blue')
plt.plot(delta2, label='delta 2', color='red')
plt.plot(delta3, label='delta 3', color='green')
plt.legend(loc='upper left')
plt.savefig(f'../results/AV_IV_EMS_delta_movements_{Nsim}.png')

plt.figure(figsize=(30, 10))
plt.plot(gamma1, label='gamma 1', color='blue')
plt.plot(gamma2, label='gamma 2', color='red')
plt.plot(gamma3, label='gamma 3', color='green')
plt.legend(loc='upper left')
plt.savefig(f'../results/AV_IV_EMS_gamma_movements_{Nsim}.png')


# evaluate_option_price(combined['predicted'], combined['value'], expected_payoff_list, experiment_details)