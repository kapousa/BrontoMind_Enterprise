import pandas as pd

df = pd.DataFrame({'DOB': {0: '26/1/2016', 1: '26/1/2016 12:15 pm'}})

df['DOB']=pd.to_datetime(df['DOB'].astype(str), format='%d/%m/%Y')
print((df.dtypes))