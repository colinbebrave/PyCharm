import pandas as pd
import numpy as np
import statsmodels.api as st
import os

os.getcwd()
os.chdir('/Users/wangqi/Documents/R/Jun_01_2017')
# ----------------------------Exercise 1----------------------------

df = pd.read_csv('Table3_1.csv')
df.head()
df.rename(columns = {'Real Investment': 'Real_Investment'}, inplace = True)

linearfit = st.OLS(df['Real_Investment'], df[['Constant', 'Trend', 'Real_GNP', 'Interest_Rate','Inflation_Rate']])

result = linearfit.fit()
print(result.summary())

# or you can do it in this way
X = np.mat(df.loc[:, ['Constant', 'Trend', 'Real_GNP', 'Interest_Rate', 'Inflation_Rate']])
print(X.shape); print(type(X))
Y = np.mat(df.loc[:, 'Real_Investment']).T
print(Y.shape); print(type(Y))

beta = (X.T * X).I * X.T * Y

# ----------------------------Exercise 2----------------------------
Identity_matrix = np.identity(df.shape[0])
X3 = np.mat(df[['Constant', 'Trend', 'Interest_Rate', 'Inflation_Rate']])
M3 = Identity_matrix - X3 * (X3.T * X3).I * X3.T

X3star = M3 * np.mat(df['Real_GNP']).T
Ystar  = M3 * Y

beta3 = (X3star.T * X3star).I * X3star.T * Ystar; beta3
# ----------------------------Exercise 3----------------------------


# ----------------------------Exercise 5----------------------------
type(result.params)
beta = result.params

def r2(data, b):
    y = data['Real_Investment']
    xb = np.matrix(data.iloc[:, 1:]) * np.matrix(beta).T
    error = np.matrix(y).T - xb
    yy = np.matrix(y) * np.matrix(y).T
    return 1 - error.T * error * (yy - data.shape[0] * y.mean()**2) ** ( -1 )

R2 = r2(df, beta)
print(R2)

# ----------------------------Exercise 6----------------------------
# firstly we do the regression
df_6 = pd.read_csv('TableF4-1.csv')
dff = df_6[df_6['LFP'] == 1]
dff['ln_earnings'] = np.log(dff['WHRS'] * dff['WW'])
dff['age'] = dff['WA']
dff['age2'] = dff['WA'] ** 2
dff['education'] = dff['WE']
dff['kids'] = (dff['KL6'] + dff['K618'] >0) * 1
dff['constant'] = 1

train_set = dff[['ln_earnings', 'constant','age', 'age2', 'education', 'kids']]
regress_col = ['constant', 'age', 'age2', 'education', 'kids']

linearfit_6 = st.OLS(train_set['ln_earnings'], train_set[regress_col])
result_6 = linearfit_6.fit()
print(result_6.summary())

# to compute the covariance matrix
x_6 = train_set[regress_col]
beta_6 = result_6.params
error_6 = np.matrix(train_set['ln_earnings']).T - np.matrix(x_6) * np.matrix(beta_6).T

sse_6 = error_6.T * error_6
s2_6 = sse_6 * (x_6.shape[0] - x_6.shape[1]) ** (-1)
covariance_6 = (np.matrix(x_6).T * np.matrix(x_6)) ** (-1) * float(s2_6)


# ----------------------------Exercise 8----------------------------

df_8 = pd.read_csv('TableF5-3.csv')

Y = df_8['VALUEADD']
L = df_8['LABOR']
K = df_8['CAPITAL ']

df_8['lnY'] = np.log(Y)
df_8['lnL'] = np.log(L)
df_8['lnK'] = np.log(K)
df_8['int_sec'] = np.multiply(df_8['lnL'], df_8['lnK'])
df_8['lnL2'] = 0.5 * np.multiply(df_8['lnL'], df_8['lnL'])
df_8['lnK2'] = 0.5 * np.multiply(df_8['lnK'], df_8['lnK'])
df_8['cons'] = 1

exp_var_cd = df_8[['cons', 'lnL', 'lnK']]
exp_var_tr = df_8[['cons', 'lnL', 'lnK', 'lnL2', 'lnK2', 'int_sec']]
de_var = df_8['lnY']

xcd = np.mat(exp_var_cd)
xtr = np.mat(exp_var_tr)
y   = np.mat(de_var).T

betacd = (xcd.T * xcd).I * xcd.T * y
betatr = (xtr.T * xtr).I * xtr.T * y

errorofcd = y - xcd * betacd
erroroftr = y - xtr * betatr

sseofcd = errorofcd.T * errorofcd
sseoftr = erroroftr.T * erroroftr

n = y.shape[0]
Identity_matrix = np.identity(y.shape[0])
M0 = Identity_matrix - np.ones([27, 27]) / n

sst = float(y.T * M0 * y)

ssrofcd = float((xcd * betacd).T * M0 * xcd * betacd)
ssroftr = float((xtr * betatr).T * M0 * xtr * betatr)

r2ofcd = ssrofcd / sst
r2oftr = ssroftr / sst

adjustedr2ofcd = 1 - (sseofcd / (xcd.shape[0] - xcd.shape[1])) / (sst / (xcd.shape[0] - 1))
adjustedr2oftr = 1 - (sseoftr / (xtr.shape[0] - xtr.shape[1])) / (sst / (xtr.shape[0] - 1))

s2ofcd = sseofcd / (xcd.shape[0] - xcd.shape[1])
s2oftr = sseoftr / (xtr.shape[0] - xtr.shape[1])

covarianceofcd = (xcd.T * xcd).I * float(s2ofcd)
covarianceoftr = (xtr.T * xtr).I * float(s2oftr)

list1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
list2 = [0, 1, 1]
restriction_matrix_1 = np.mat(np.reshape(list1, (3, 6)))
restriction_matrix_2 = np.mat(list2)
res1 = restriction_matrix_1.copy()
res2 = restriction_matrix_2.copy()

Ftest3_21 = (res1 * betatr).T * (res1 * covarianceoftr * res1.T).I * res1 * betatr / 3

Ftest1_24 = (res2 * betacd - 1).T * (res2 * covarianceofcd * res2.T).I * (res2 * betacd - 1)