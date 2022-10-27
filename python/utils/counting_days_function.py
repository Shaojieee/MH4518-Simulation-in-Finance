from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

## Latest Price date is in "%Y-%m-%d" format
def days(latest_price_date):
    holidays=['2022-09-05','2022-11-24','2022-12-26','2023-01-01','2023-01-16','2023-02-20','2023-04-07','2023-05-29','2023-07-04']
    weekmask=[1,1,1,1,1,0,0]
    before_start_date = datetime.strptime('2022-08-18',"%Y-%m-%d")
    start_date = datetime.strptime('2022-08-19',"%Y-%m-%d")
    end_date = datetime.strptime(latest_price_date,"%Y-%m-%d")
    maturity_date = datetime.strptime('2023-08-22',"%Y-%m-%d")
    total_trading_days = np.busday_count(start_date.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

    q2 = start_date+relativedelta(months=+6)
    # check if q2 date is a weekday or holiday
    while q2 in holidays or q2.weekday()==5 or q2.weekday()==6:
        if q2.weekday()==5:
            q2 = q2+relativedelta(days=+2)
        elif q2.weekday()==6:
            q2 = q2+relativedelta(days=+1)
        else:
            q2 = q2 +relativedelta(days=+1)
    q2_to_maturity = np.busday_count(q2.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

    q3 = start_date+relativedelta(months=+9)
    # Check if q3 date is a weekday or holiday
    while q3 in holidays or q3.weekday()==5 or q3.weekday()==6:
        if q3.weekday()==5:
            q3 = q3+relativedelta(days=+2)
        elif q2.weekday()==6:
            q3 = q3+relativedelta(days=+1)
        else:
            q3 = q3 +relativedelta(days=+1)
    q3_to_maturity = np.busday_count(q3.strftime("%Y-%m-%d"),(maturity_date+relativedelta(days=+1)).strftime("%Y-%m-%d"),weekmask=weekmask,holidays=holidays)

    cur_date = datetime.strptime(before_start_date.strftime('%m/%d/%Y'),'%m/%d/%Y')
    end_date = datetime.strptime(end_date.strftime('%m/%d/%Y'),'%m/%d/%Y')

    return cur_date,end_date,q2_to_maturity,q3_to_maturity,total_trading_days