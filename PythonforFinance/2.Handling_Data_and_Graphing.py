df.to_csv('TSLA.csv')

df = pd.read_csv('TSLA.csv', parse_dates = True, index_col = 0)

df.plot()
plt.show()

# in the first graph, all what we can clearly see is the volume,
# since on a scale it is much larger than stock price
# in the following, we'd like to plot what we are interested in
df['Adj Close'].plot()
plt.show()
# we could also plot multiple variables at the same time
df[['High', 'Low']].plot()