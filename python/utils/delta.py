import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import random


def SimMultiGBMpmh(S0, v, sigma, Deltat, T, Z):
    m = int(T/Deltat)
    p = len(S0)
    S = np.zeros((3*p, m+1))

    h = []
    S0_ph = []
    S0_mh = []

    for i in range(p):
        h.append(0.01*S0[i])
        S0_ph.append(S0[i] + h[i])
        S0_mh.append(S0[i] - h[i])

    Z_ = np.zeros((3 * p, m))

    for i in range(p):
        S[p*i][0] = S0_mh[i]
        S[p*i+1][0] = S0[i]
        S[p*i+2][0] = S0_ph[i]
        Z_[3*i] = Z[i]
        Z_[3*i+1] = Z[i]
        Z_[3*i+2] = Z[i]
    # print("Z_")
    # print(Z_)
    # print("-------")
    # print("Z_ transpose")
    # print(Z_)
    # print("-------")
    # print("Z")
    # print(Z)
    # print("-------")
    # print("S")
    # print(S)
    # print("-------")
    for i in range(1, m+1):
        S[:, i:i + 1] = np.exp(np.log(S[:, i - 1:i]) + Z_[:, i - 1:i])

    return S, h

