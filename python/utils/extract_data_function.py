import pandas as pd
from datetime import datetime

def extract_data(path,start_date,end_date):
    data = pd.read_csv(path)
    data['Close/Last']=data['Close/Last'].str.replace('$','')
    for index,row in data.iterrows():
        row['Date'] = datetime.strptime(row['Date'],"%m/%d/%Y")
    data['Close/Last'] = data['Close/Last'].astype(float)
    data['Date'] = data['Date'].astype('datetime64[ns]')
    data = data.drop(columns=['Volume','Open','High','Low'])
    data = data[data['Date']>=start_date]
    data= data[data['Date']<=end_date]
    data = data.sort_values(by='Date', ascending=False).reset_index(drop=True)
    return data

def extract_current_price(path,current_date):
    data = pd.read_csv(path)
    data['Close/Last']=data['Close/Last'].str.replace('$','')
    data['Close/Last'] = data['Close/Last'].astype(float)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.drop(columns=['Volume','Open','High','Low'])
    data = data[data['Date']==current_date]
    return data['Close/Last'].values.tolist()[0]