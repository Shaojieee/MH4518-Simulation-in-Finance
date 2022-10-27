def payoff_maturity(aapl,amzn,googl):
    aapl_barrier = 85.760
    amzn_barrier = 69.115
    googl_barrier = 58.605
    aapl_initial = 171.52
    amzn_initial = 138.23
    google_initial = 117.21

    # Checking if any stock broke barrier
    if min(aapl)<=aapl_barrier or min(amzn)<=amzn_barrier or min(googl)<=googl_barrier:
        aapl_percent = aapl[-1]/aapl_initial
        amzn_percent = amzn[-1]/amzn_initial
        googl_percent = googl[-1]/google_initial

        # Checking for worse performing stocks
        if aapl_percent<amzn_percent and aapl_percent<googl_percent:
            lowest_stock = 'aapl'
            lowest_percent = aapl_percent
            lowest_price = aapl[-1]
            conversion_ratio = 5.8302

        elif amzn_percent<googl_percent and amzn_percent<aapl_percent:
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
        if lowest_percent>=1:
            return 1000*1.1
        else:
            return lowest_price*conversion_ratio + 1000*0.1

    else:
        return 1000*1.1


def payoff_quarterly(aapl,amzn,googl,quarter):
    aapl_initial = 171.52
    amzn_initial = 138.23
    googl_initial = 117.21

    if aapl[-1]<aapl_initial or amzn[-1]<amzn_initial or googl[-1]<googl_initial:
        return 0
    elif quarter==2:
        return 1000*1.05
    elif quarter==3:
        return 1000*1.075
    return 0

    