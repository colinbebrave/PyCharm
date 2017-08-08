import pandas as pd
import statsmodels.api as sm

#read data and modify it
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")
df.columns = ['admit', 'gre', 'gpa', 'prestige']
print(df.head())
df.to_csv('0412.csv')
df = pd.read_csv('0412.csv')
#print(df.describe())

#define dummy variables
dummy_ranks = pd.get_dummies(df['prestige'],prefix='prestige')
print(dummy_ranks.head())

cols_to_keep = ['admit', 'gre', 'gpa']

data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige':])
print(data.head())
#add the intercept
data['intercept'] = 1.0
#cols = data.columns.tolist(); cols
#cols = cols[-1:] + cols[:-1]
#data = data[cols]
#print(data.head())

# create the training set and then do the regression
train_cols = data.columns[1:]
print(train_cols)
logitfit = sm.Logit(data['admit'], data[train_cols])
result = logitfit.fit()
print(result.summary())

# to deal with the perfect multicollinearity
training_set = data[['gre', 'gpa', 'prestige_2','prestige_3', 'prestige_4', 'intercept']]
#training_set = data.iloc[:,1:]
print(training_set.head())
logitfit1 = sm.Logit(data['admit'], training_set)
result1 = logitfit1.fit()
print(result1.summary())

import pandas as pd
