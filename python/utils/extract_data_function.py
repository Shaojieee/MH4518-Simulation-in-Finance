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
    return data