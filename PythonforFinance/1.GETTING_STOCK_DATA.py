import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
# we are setting the start and end datetime object,
# this will be the range of dates that we'ar gonna grab stock pricing information for
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)

# get data
df = web.DataReader('TSLA', 'yahoo', start, end)
print(df.head())

