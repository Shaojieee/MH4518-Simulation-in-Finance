from datetime import timedelta
from utils.counting_days_function import days
from utils.extract_data_function import extract_data
from utils.payoff_function import calculate_option_price
from utils.simulation_function import SimMultiGBMAV
from utils.evaluation import evaluate_option_price
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import random

import warnings
warnings.filterwarnings("ignore")

experiment_details = {
    'Nsim': 10000,
    'latest_price_date': '2022-10-24',
    'variance_reduction': "antithetic",
    'GBM': 'multivariate',
    'r': 0.045
}

aapl_barrier = 85.760
amzn_barrier = 69.115
googl_barrier = 58.605
aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21


date_to_predict, hist_end, end_date, q2_to_maturity, q3_to_maturity, q2, q3, total_trading_days, holidays = days(
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
    sim_aapl_tilde = np.zeros((Nsim, m + 1))
    sim_amzn_tilde = np.zeros((Nsim, m + 1))
    sim_googl_tilde = np.zeros((Nsim, m + 1))
    random.seed(4518)

    for i in range(0, int(Nsim/2)):
        S, Stilde = SimMultiGBMAV(S0, v, sigma, dt, T)
        sim_aapl[i] = S[0]
        sim_aapl_tilde[i] = Stilde[0]
        sim_amzn[i] = S[0]
        sim_amzn_tilde[i] = Stilde[0]
        sim_googl[i] = S[0]
        sim_googl_tilde[i] = Stilde[0]


    q2_index = total_trading_days - q2_to_maturity if total_trading_days - q2_to_maturity>=0 else None
    q3_index = total_trading_days - q3_to_maturity if total_trading_days - q3_to_maturity>=0 else None

    option_prices = []

    for i in range(int(Nsim/2)):
        S_payoff = calculate_option_price(aapl=sim_aapl[i],
                                          amzn=sim_amzn[i],
                                          googl=sim_googl[i],
                                          T=trading_days_to_simulate,
                                          total_trading_days=total_trading_days,
                                          r=r,
                                          q2_index=q2_index,
                                          q3_index=q3_index
                                          )

        S_tilde_payoff = calculate_option_price(aapl=sim_aapl_tilde[i],
                                          amzn=sim_amzn_tilde[i],
                                          googl=sim_googl_tilde[i],
                                          T=trading_days_to_simulate,
                                          total_trading_days=total_trading_days,
                                          r=r,
                                          q2_index=q2_index,
                                          q3_index=q3_index
                                          )
        option_prices.append(S_payoff + S_tilde_payoff)

    option_price = np.mean(option_prices) / 2
    expected_payoff_maturity.append(option_price)
    predicted_option_price.append({'date':date_to_predict, 'predicted': option_price})
    print(f"Derivative Price for {date_to_predict}")
    print(option_price)

    # sim_aapl = EMS(sim_aapl,dt,r)
    # # sim_amzn = EMS(sim_amzn,dt,r)
    # sim_googl = EMS(sim_googl,dt,r)

    # cur_expected_payoff = np.mean(payoff_maturity)
    # expected_payoff_maturity.append(cur_expected_payoff)

    # TODO Apply regression to find the coefficient for each payoff
    # option_price = np.exp(-r*(trading_days_to_predict/total_trading_days))*cur_expected_payoff*w_1 + np.exp(-r/2)*np.mean(payoff_q2)*w_2 + np.exp(-r*3/4)*np.mean(payoff_q3)*w_3
    date_to_predict += relativedelta(days=+1)
    trading_days_to_simulate -= 1
    hist_end += relativedelta(days=+1)
    # counter+=1


predicted_option_price = pd.DataFrame(predicted_option_price)
predicted_option_price['date'] = pd.to_datetime(predicted_option_price['date'])
# Scale back to 100%
predicted_option_price['predicted'] = predicted_option_price['predicted']/10
actual_option_price = pd.read_csv('../data/derivative_01_11_22.csv')
actual_option_price['date'] = pd.to_datetime(actual_option_price['date'], format='%Y-%m-%d')
combined = predicted_option_price.merge(actual_option_price, left_on=['date'], right_on=['date'], validate='one_to_one')
combined = combined.set_index('date')


evaluate_option_price(combined['predicted'], combined['value'], experiment_details)