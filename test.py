import pandas as pd
import numpy as np

grades = [48, 99, 75, 80, 42, 80, 72, 68, 36, 78]

df = pd.DataFrame({'ID': ['x%d' % r for r in range(10)],
                   'Gender':['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                   'ExamYear': ['2007', '2008', '2009', '2007', '2008'] * 2,
                   'Class': ['algebra', 'stats', 'bio', 'algebra', 'algebra', 'stats', 'stats', 'algebra', 'bio', 'bio'],
                   'Participated': ['yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
                   'Passed': ['yes' if x  > 50 else 'no' for x in grades],
                   'Employed': [True, True, True, False, False, False, False, True, True, False],
                   'Grade':grades})

df.groupby('ExamYear').agg({'Participated': lambda x: x.value_counts()['yes'],
                            'Passed': lambda x: sum(x == 'yes'),
                            'Employed': lambda x: sum(x),
                            'Grade': lambda x : sum(x)/ len(x)})

df = pd.DataFrame({'value': np.random.randn(36)}, index = pd.date_range('2011-01-01', freq = 'M', periods = 36))

pd.pivot_table(df, index = df.index.month, columns = df.index.year, values = 'value', aggfunc = 'sum')

# Apply

df = pd.DataFrame({'A': [[2, 4, 8, 16], [100, 200], [10, 20, 30]],
                   'B': [['a', 'b', 'c'], ['jj', 'kk'], ['ccc']]}, index = ['I', 'II', 'III'])
def SeriesFromList(aList):
    return pd.Series(aList)

df = pd.DataFrame(np.random.randn(2000, 2) / 10000, index = pd.date_range('2001-01-01', periods = 2000), columns = ['A', 'B']); df

def gm(aDF, Const):
    v = ((((aDF.A + aDF.B) + 1).cumprod()) - 1)*Const
    return (aDF.index[0], v.iloc[-1])

"""
The Beauty of Coding
"""
s = pd.Series(['six', 'seven', 'six', 'seven', 'six'], index = ['a', 'b', 'c', 'd', 'e']); s
t = pd.Series({'six': 6., 'seven': 7.}); t
s.map(t)

df_1 = pd.DataFrame({'one': pd.Series(np.random.randint(3), index = list('abc')),
                     'two': pd.Series(np.random.randint(4), index = list('abcd')),
                     'three': pd.Series(np.random.randint(3), index = list('bcd'))})

df_1.rename(columns = {'one': 'foo', 'two': 'bar'}, index = {'a': 'apple', 'b': 'banana', 'd': 'durian'})

dft = pd.DataFrame(dict(A = np.random.rand(3),
                        B = 1,
                        C = 'foo',
                        D = pd.Timestamp('20000102'),
                        E = pd.Series([1.0] * 3).astype('float32'),
                        F = False,
                        G = pd.Series([1] * 3, dtype = 'int8')))

df = pd.DataFrame({'string': list('abc'),
                   'int64': list(range(1, 4)),
                   'unit8': np.arange(3, 6).astype('u1'),
                   'float32': np.arange(4.0, 7.0),
                   'bool1': [True, False, True],
                   'bool2': [False, True, False],
                   'dates': pd.date_range('now', periods = 3).values,
                   'category': pd.Series(list('ABC')).astype('category')})

arrays = [
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two'] * 4
]

df_1 = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         'foo', 'bar', 'foo', 'foo'],
                     'B': ['one', 'one', 'two', 'three',
                         'two', 'two', 'one', 'three'],
                     'C': np.random.randn(8),
                     'D': np.random.randn(8)})


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



df = pd.DataFrame({"name":["Foo", "Foo", "Baar", "Foo", "Baar", "Foo", "Baar", "Baar"], "count_1":[5,10,12,15,20,25,30,35], "count_2" :[100,150,100,25,250,300,400,500]})
df.groupby(["name"])["count_1"].nlargest(3)
df.groupby(["name"]).apply(lambda x: x.sort_values(["count_1"], ascending = False)).reset_index(drop=True)


"""replace values using dict"""
df = pd.DataFrame({'col2': {0: 'a', 1: 2, 2: np.nan}, 'col1': {0: 'w', 1: 1, 2: 2, 3:1, 4:2, 5:2, 6:2}})
di = {1:'A', 2:'B'}
df.replace({'col1':di})


