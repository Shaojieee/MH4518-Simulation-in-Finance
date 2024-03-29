import numpy as np
import pandas as pd

aapl_barrier = 85.760
amzn_barrier = 69.115
googl_barrier = 58.605
aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21


def calculate_option_price(aapl, amzn, googl, T, total_trading_days, q2, q3, maturity, cur, q2_index, q3_index, interpolate_r, r, rates):
    q2_autocall = False
    q3_autocall = False
    barrier_event = False

    if q2_index:
        if aapl[q2_index] > aapl_initial and amzn[q2_index] > amzn_initial and googl[q2_index] > google_initial:
            payoff_ = 1000*1.05
            q2_autocall = True
            if r==None:
                num_days = (q2 - cur).days
                r = calculate_r(num_days, cur, interpolate_r, rates)
            return np.exp(-r * (q2_index) / total_trading_days) * payoff_,payoff_, r

    if q3_index:
        if aapl[q3_index] > aapl_initial and amzn[q3_index] > amzn_initial and googl[q3_index] > google_initial:
            payoff_ = 1000*1.075
            q3_autocall = True
            if r==None:
                num_days = (q3 - cur).days
                r = calculate_r(num_days, cur, interpolate_r, rates)
            return np.exp(-r * (q3_index) / total_trading_days) * payoff_,payoff_, r

    maturity_payoff_ = maturity_payoff(aapl, amzn, googl)
    if r==None:
        num_days = (maturity - cur).days
        r = calculate_r(num_days, cur, interpolate_r, rates)
    return np.exp(-r * T / total_trading_days) * maturity_payoff_,maturity_payoff_, r


def calculate_r(num_days, cur, interpolate_r, rates):

    rates = rates.loc[cur.strftime('%Y-%m-%d'), ].to_dict()

    if not interpolate_r:
        return rates['1 YR']/100.0

    if num_days>=60 and num_days<90:
        grad = (rates['3 MO'] - rates['2 MO'])/30
        x_intercept = -(grad*90 - rates['3 MO'])

        return (grad*num_days+x_intercept)/100.0

    elif num_days>=90 and num_days<180:
        grad = (rates['6 MO'] - rates['3 MO']) / 90
        x_intercept = -(grad * 90 - rates['3 MO'])

        return (grad * num_days + x_intercept)/100.0

    elif num_days>=180 and num_days<365.25:
        grad = (rates['1 YR'] - rates['6 MO']) / (365.25-180)
        x_intercept = -(grad * 180 - rates['6 MO'])

        return (grad * num_days + x_intercept)/100.0

    return (rates['1 YR'])/100.0


def maturity_payoff(aapl, amzn, googl):
    # Checking if any stock broke barrier
    if min(aapl) <= aapl_barrier or min(amzn) <= amzn_barrier or min(googl) <= googl_barrier:
        aapl_percent = aapl[-1] / aapl_initial
        amzn_percent = amzn[-1] / amzn_initial
        googl_percent = googl[-1] / google_initial

        # Checking for worse performing stocks
        if aapl_percent < amzn_percent and aapl_percent < googl_percent:
            lowest_stock = 'aapl'
            lowest_percent = aapl_percent
            lowest_price = aapl[-1]
            conversion_ratio = 5.8302

        elif amzn_percent < googl_percent and amzn_percent < aapl_percent:
            lowest_stock = 'amzn'
            lowest_percent = amzn_percent
            lowest_price = amzn[-1]
            conversion_ratio = 7.2343
        else:
            lowest_stock = 'googl'
            lowest_percent = googl_percent
            lowest_price = googl[-1]
            conversion_ratio = 8.5317

        # Checking if worse performing stock closes below its intial level
        if lowest_percent >= 1:
            return 1000 * 1.1
        else:
            return lowest_price * conversion_ratio + 1000 * 0.1

    else:
        return 1000 * 1.1


def get_payoff_pmh(S0, sim_aapl_mh, sim_aapl, sim_aapl_ph,
                   sim_amzn_mh, sim_amzn, sim_amzn_ph, sim_googl_mh, 
                   sim_googl, sim_googl_ph, trading_days_to_simulate, total_trading_days,
                   r, q2_index, q3_index, q2, q3, interpolate_r, h_prop, maturity, cur, rates):
    
    aapl_h = h_prop * S0[0]
    amzn_h = h_prop * S0[1]
    googl_h = h_prop * S0[2]
    
    option_aapl_mh, _, _ = calculate_option_price(aapl=sim_aapl_mh, amzn=sim_amzn, googl=sim_googl, T=trading_days_to_simulate,
                                               total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_aapl, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn, googl=sim_googl, T=trading_days_to_simulate,
                                            total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_aapl_ph, _, _ = calculate_option_price(aapl=sim_aapl_ph, amzn=sim_amzn, googl=sim_googl, T=trading_days_to_simulate,
                                               total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)
    
    option_amzn_mh, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn_mh, googl=sim_googl, T=trading_days_to_simulate,
                                               total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_amzn, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn, googl=sim_googl,
                                            T=trading_days_to_simulate, total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                            q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_amzn_ph, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn_ph, googl=sim_googl, T=trading_days_to_simulate,
                                               total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)
    
    option_googl_mh, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn, googl=sim_googl_mh, T=trading_days_to_simulate,
                                                total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                               q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_googl, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn, googl=sim_googl,
                                             T=trading_days_to_simulate, total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                             q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)

    option_googl_ph, _, _ = calculate_option_price(aapl=sim_aapl, amzn=sim_amzn, googl=sim_googl_ph, T=trading_days_to_simulate,
                                                total_trading_days=total_trading_days, r=r, q2_index=q2_index, q3_index=q3_index,
                                                q2=q2, q3=q3, interpolate_r=interpolate_r, maturity=maturity, cur=cur, rates=rates)
    
    delta_payoff_aapl_pmh = (option_aapl_ph - option_aapl_mh) / (2 * aapl_h)
    delta_payoff_amzn_pmh = (option_amzn_ph - option_aapl_mh) / (2 * amzn_h)
    delta_payoff_googl_pmh = (option_googl_ph - option_aapl_mh) / (2 * googl_h)

    gamma_payoff_aapl_pmh = (option_aapl_ph - 2*option_aapl + option_aapl_mh) / (aapl_h*aapl_h)
    gamma_payoff_amzn_pmh = (option_amzn_ph - 2*option_amzn + option_amzn_mh) / (amzn_h*amzn_h)
    gamma_payoff_googl_pmh = (option_googl_ph - 2*option_googl + option_googl_mh) / (googl_h*googl_h)
    
    delta_payoff_pmh_ = []
    delta_payoff_pmh_.append(delta_payoff_aapl_pmh)
    delta_payoff_pmh_.append(delta_payoff_amzn_pmh)
    delta_payoff_pmh_.append(delta_payoff_googl_pmh)

    gamma_payoff_pmh_ = []
    gamma_payoff_pmh_.append(gamma_payoff_aapl_pmh)
    gamma_payoff_pmh_.append(gamma_payoff_amzn_pmh)
    gamma_payoff_pmh_.append(gamma_payoff_googl_pmh)
    
    return delta_payoff_pmh_, gamma_payoff_pmh_




