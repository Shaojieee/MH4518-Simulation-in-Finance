import numpy as np

aapl_initial = 171.52
amzn_initial = 138.23
google_initial = 117.21


def get_worst_stock(aapl, amzn, googl):
    min_aapl = min(aapl)
    min_amzn = min(amzn)
    min_googl = min(googl)

    aapl_percent = min_aapl / aapl_initial
    amzn_percent = min_amzn / amzn_initial
    googl_percent = min_googl / google_initial
    lowest_price = 1e9

    # Checking for worse performing stocks
    if aapl_percent < amzn_percent and aapl_percent < googl_percent:
        lowest_price = min_aapl

    elif amzn_percent < googl_percent and amzn_percent < aapl_percent:
        lowest_price = min_amzn

    else:
        lowest_price = min_googl

    return lowest_price


def get_features(S1, S2, S3, q_index, Nsim):
    features = np.zeros(10)
    features[0] = S1[q_index]
    features[1] = S2[q_index]
    features[2] = S3[q_index]
    features[3] = S1[q_index] * S1[q_index]
    features[4] = S2[q_index] * S2[q_index]
    features[5] = S3[q_index] * S3[q_index]
    features[6] = S1[q_index] * S2[q_index]
    features[7] = S1[q_index] * S3[q_index]
    features[8] = S2[q_index] * S3[q_index]
    features[9] = get_worst_stock(S1, S2, S3)
    return features
