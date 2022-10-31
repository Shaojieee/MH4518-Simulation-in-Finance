import numpy as np

aapl_barrier = 85.760
amzn_barrier = 69.115
googl_barrier = 58.605
aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21


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


def discounted_quarterly_payoff(next_q_payoff, q_to_next_q, interest_rate):
    discounted_q_payoff = next_q_payoff * np.exp(-interest_rate * (q_to_next_q)/252)
    # print("discounted q3 payoff")
    # print(discounted_q3_payoff)
    # print("-------")
    return discounted_q_payoff


def quarterly_payoff(aapl, amzn, googl, quarter):
    barrier_effect = False

    if min(aapl) <= aapl_barrier or min(amzn) <= amzn_barrier or min(googl) <= googl_barrier:
        aapl_percent = aapl[-1] / aapl_initial
        amzn_percent = amzn[-1] / amzn_initial
        googl_percent = googl[-1] / google_initial
        barrier_effect = True

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

    if quarter == 2:
        if barrier_effect: # barrier effect has occurred
            return lowest_price * conversion_ratio + 1000 * 0.05
        else:  # no barrier effect, return coupon payment
            return 1000 * 1.05
    elif quarter == 3:
        if barrier_effect:
            return lowest_price * conversion_ratio + 1000 * 0.075
        else:
            return 1000 * 1.075
    return 0

# def quarterly_payoff(aapl,amzn,googl,quarter):
#     aapl_initial = 171.52
#     amzn_initial = 138.23
#     googl_initial = 117.21
#
#     if aapl[-1]<aapl_initial or amzn[-1]<amzn_initial or googl[-1]<googl_initial:
#         return 0
#     elif quarter==2:
#         return 1000*1.05
#     elif quarter==3:
#         return 1000*1.075
#     return 0
