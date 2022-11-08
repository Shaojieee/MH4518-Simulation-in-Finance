
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from utils.payoff_function import calculate_option_price,calculate_r, get_payoff_pmh
from utils.evaluation import evaluate_option_price
from utils.simulation_function import SimMultiGBMAV,SimMultiGBM, SimMultiGBMpmh
from utils.extract_data_function import extract_data
from utils.counting_days_function import days
from utils.ems_correction import EMSCorrection
from utils.calculate_implied_volatility import cov_actual_IV, cov_estimated_IV
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings("ignore")


def calculate_derivative(experiment_details):

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

    if experiment_details['estimated_IV']:
        aapl_call_df = pd.read_csv('../Bloomberg_Data/aapl_call.csv')
        aapl_call_df['Date'] = pd.to_datetime(aapl_call_df['Date'])
        amzn_call_df = pd.read_csv('../Bloomberg_Data/amzn_call.csv')
        amzn_call_df['Date'] = pd.to_datetime(amzn_call_df['Date'])
        googl_call_df = pd.read_csv('../Bloomberg_Data/googl_call.csv')
        googl_call_df['Date'] = pd.to_datetime(googl_call_df['Date'])

        min_sigma = experiment_details['min_sigma']
        max_sigma = experiment_details['max_sigma']
        step = experiment_details['step']
        sigma_hat_list = np.arange(min_sigma, max_sigma + step, step)
        right = len(sigma_hat_list) - 1
        left = 0

    if experiment_details['actual_IV']:
        aapl_IV_df = pd.read_csv('../Bloomberg_Data/aapl_IV.csv')
        aapl_IV_df['Date'] = pd.to_datetime(aapl_IV_df['Date'], format='%Y-%m-%d')
        amzn_IV_df = pd.read_csv('../Bloomberg_Data/amzn_IV.csv')
        amzn_IV_df['Date'] = pd.to_datetime(amzn_IV_df['Date'], format='%Y-%m-%d')
        googl_IV_df = pd.read_csv('../Bloomberg_Data/googl_IV.csv')
        googl_IV_df['Date'] = pd.to_datetime(googl_IV_df['Date'], format='%Y-%m-%d')

    if 'r' in experiment_details:
        rates = pd.read_csv('../data/04-11-2022/USTREASURY-YIELD_04_11_22.csv')
        rates['Date'] = pd.to_datetime(rates['Date'], format='%Y-%m-%d')
        rates = rates.set_index('Date')
        rates = rates.asfreq('D')
        rates = rates.ffill()
    else:
        rates = None

    predicted_option_price = []
    expected_payoff_list = []
    delta_aapl = []
    delta_amzn = []
    delta_googl = []
    gamma_aapl = []
    gamma_amzn = []
    gamma_googl = []
    r_list = []
    v_aapl = []
    v_amzn = []
    v_googl = []
    sigma_aapl = []
    sigma_amzn = []
    sigma_googl = []
    aapl_before_emc = []
    amzn_before_emc = []
    googl_before_emc = []
    prediction_dates = []


    while date_to_predict <= end_date:
        prediction_dates.append(date_to_predict)
        if datetime.strftime(date_to_predict,
                             "%Y-%m-%d") in holidays or date_to_predict.weekday() == 5 or date_to_predict.weekday() == 6:
            date_to_predict += relativedelta(days=+1)
            hist_end += relativedelta(days=+1)
            continue

        hist_start = hist_end - timedelta(days=experiment_details['window_length'])

        aapl = extract_data('../data/04-11-2022/aapl_04_11_2022.csv', hist_start, hist_end).rename(
            columns={'Close/Last': 'AAPL'})
        amzn = extract_data('../data/04-11-2022/amzn_04_11_2022.csv', hist_start, hist_end).rename(
            columns={'Close/Last': 'AMZN'})
        googl = extract_data('../data/04-11-2022/googl_04_11_2022.csv', hist_start, hist_end).rename(
            columns={'Close/Last': 'GOOGL'})
        temp_df = aapl.merge(amzn, on=['Date'])
        AAG = temp_df.merge(googl, on=['Date']).drop(columns=['Unnamed: 0'])
        n0 = len(AAG)
        AAGprices = np.array(AAG.drop(columns=['Date']))
        AAGlogprices = np.log(AAGprices)
        AAGlogreturns = AAGlogprices[:n0 - 1, :] - AAGlogprices[1:, :]

        v = np.mean(AAGlogreturns, axis=0)
        v_aapl.append(v[0])
        v_amzn.append(v[1])
        v_googl.append(v[2])
        Nsim = experiment_details['Nsim']
        T = trading_days_to_simulate
        dt = 1
        m = int(T / dt)
        r = experiment_details['r'] if 'r' in experiment_details else calculate_r(0, date_to_predict,
                                                                                  interpolate_r=False, rates=rates)

        # If IV setting is true covariance matrix is calculated from implied volatility of individual stocks  options market price
        if experiment_details['estimated_IV']:
            sigma = cov_estimated_IV(aapl_call_df, amzn_call_df, googl_call_df, r, alternative_option_ttm,
                                     sigma_hat_list, left, right, date_to_predict, AAGlogreturns)

        elif experiment_details['actual_IV']:
            sigma = cov_actual_IV(aapl_IV_df, amzn_IV_df, googl_IV_df, date_to_predict, AAGlogreturns)

        else:
            sigma = np.cov(AAGlogreturns, rowvar=False)

        sigma_aapl.append(np.sqrt(sigma.diagonal())[0])
        sigma_amzn.append(np.sqrt(sigma.diagonal())[1])
        sigma_googl.append(np.sqrt(sigma.diagonal())[2])

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
                S_pmh, Stilde_pmh = SimMultiGBMpmh(S0, v, sigma, dt, T, Z, variance_reduction=True,
                                                   h_prop=experiment_details['h_prop'])
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
                S_pmh = SimMultiGBMpmh(S0, v, sigma, dt, T, Z, variance_reduction=False,
                                       h_prop=experiment_details['h_prop'])
                sim_aapl_mh.append(S_pmh[0])
                sim_aapl_ph.append(S_pmh[2])
                sim_amzn_mh.append(S_pmh[3])
                sim_amzn_ph.append(S_pmh[5])
                sim_googl_mh.append(S_pmh[6])
                sim_googl_ph.append(S_pmh[8])

        aapl_before_emc = sim_aapl
        amzn_before_emc = sim_amzn
        googl_before_emc = sim_googl

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
            option_price, payoff, r = calculate_option_price(
                aapl=sim_aapl[i],
                amzn=sim_amzn[i],
                googl=sim_googl[i],
                T=trading_days_to_simulate,
                total_trading_days=total_trading_days,
                q2_index=q2_index,
                q3_index=q3_index,
                q2=q2,
                q3=q3,
                maturity=datetime.strptime('2023-08-22', "%Y-%m-%d"),
                cur=date_to_predict,
                interpolate_r=experiment_details['interpolate_r'],
                r=experiment_details['r'] if 'r' in experiment_details else None,
                rates=rates
            )
            r_list.append(r)
            option_prices.append(
                option_price
            )
            payoff_list.append(payoff)
            delta_payoff_pmh_, gamma_payoff_pmh_ = get_payoff_pmh(S0, sim_aapl_mh[i], sim_aapl[i], sim_aapl_ph[i],
                                                                  sim_amzn_mh[i], sim_amzn[i], sim_amzn_ph[i],
                                                                  sim_googl_mh[i],
                                                                  sim_googl[i], sim_googl_ph[i],
                                                                  trading_days_to_simulate=trading_days_to_simulate,
                                                                  total_trading_days=total_trading_days,
                                                                  r=experiment_details[
                                                                      'r'] if 'r' in experiment_details else None,
                                                                  q2_index=q2_index, q3_index=q3_index,
                                                                  q2=q2, q3=q3,
                                                                  interpolate_r=experiment_details['interpolate_r'],
                                                                  h_prop=experiment_details['h_prop'],
                                                                  maturity=datetime.strptime('2023-08-22', "%Y-%m-%d"),
                                                                  cur=date_to_predict,
                                                                  rates=rates)
            delta_payoff_pmh.append(delta_payoff_pmh_)
            gamma_payoff_pmh.append(gamma_payoff_pmh_)

        expected_payoff_list.append(np.mean(payoff_list))
        option_price = np.mean(option_prices)
        predicted_option_price.append({'date': date_to_predict, 'predicted': option_price})
        print(f"Derivative Price for {date_to_predict}")
        print(option_price)

        delta_payoff_pmh = np.transpose(delta_payoff_pmh)
        gamma_payoff_pmh = np.transpose(gamma_payoff_pmh)

        delta_aapl.append(np.mean(delta_payoff_pmh[0]))
        delta_amzn.append(np.mean(delta_payoff_pmh[1]))
        delta_googl.append(np.mean(delta_payoff_pmh[2]))
        gamma_aapl.append(np.mean(gamma_payoff_pmh[0]))
        gamma_amzn.append(np.mean(gamma_payoff_pmh[1]))
        gamma_googl.append(np.mean(gamma_payoff_pmh[2]))

        date_to_predict += relativedelta(days=+1)
        trading_days_to_simulate -= 1
        hist_end += relativedelta(days=+1)
        alternative_option_ttm -= 1

    predicted_option_price = pd.DataFrame(predicted_option_price)
    predicted_option_price['date'] = pd.to_datetime(predicted_option_price['date'])
    # Scale back to 100%
    predicted_option_price['predicted'] = predicted_option_price['predicted'] / 10
    actual_option_price = pd.read_csv('../data/04-11-2022/derivative_04_11_2022.csv')
    actual_option_price['date'] = pd.to_datetime(actual_option_price['date'], format='%Y-%m-%d')
    combined = predicted_option_price.merge(actual_option_price, left_on=['date'], right_on=['date'],
                                            validate='one_to_one')
    combined = combined.set_index('date')

    evaluate_option_price(combined['predicted'], combined['value'], expected_payoff_list, experiment_details,
                          delta_aapl, delta_amzn, delta_googl, gamma_aapl, gamma_amzn, gamma_googl)

    results = {
        'sim_aapl': sim_aapl,
        'sim_amzn': sim_amzn,
        'sim_googl': sim_googl,
        'actual_derivative': combined['value'],
        'sim_derivative': combined['predicted'],
        'delta_aapl': delta_aapl,
        'delta_amzn': delta_amzn,
        'delta_googl': delta_googl,
        'gamma_aapl': gamma_aapl,
        'gamma_amzn': gamma_amzn,
        'gamma_googl': gamma_googl,
        'r': r if 'r' in experiment_details else r_list,
        'v_aapl': v_aapl,
        'v_amzn': v_amzn,
        'v_googl': v_googl,
        'sigma_aapl': sigma_aapl,
        'sigma_amzn': sigma_amzn,
        'sigma_googl': sigma_googl,
        'prediction_dates': prediction_dates
    }

    if experiment_details['EMS']:
        results['aapl_before_emc'] = aapl_before_emc
        results['amzn_before_emc'] = amzn_before_emc
        results['googl_before_emc'] = googl_before_emc

    return results
