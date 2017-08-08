import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import timeit
import os
os.chdir('/Users/wangqi/Documents/PyCharm/ML_Regression/Week5')

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)

#1 Create new features by perfoming following transformation on inputs
sales['sqft_living_sqrt'] = np.sqrt(sales['sqft_living'])
sales['sqft_lot_sqrt'] = np.sqrt(sales['sqft_lot'])
sales['bedrooms_square'] = np.power(sales['bedrooms'], 2)
sales['floors_square'] = np.power(sales['floors'], 2)
# timeit(np.power(sales['floors'], 2))
# timeit(sales['floors'] * sales['floors'])

#2 Regress using an L1 penalty of 5e2
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha = 5e2, normalize = True) # initialize parameters
model_all.fit(sales[all_features], sales['price']) # learn weights
print(model_all.coef_.shape)

# based on the estimated coefficients, there are only 3 features are chose by LASSO

#3 To determine features that are chosen by LASSO
coef_all = model_all.coef_
print(np.array(all_features)[coef_all != 0])

# so, only 'sqft_living', 'view' and 'grade' are chosen by LASSO

#4 To find a good L1 penalty, we will explore multiple values using a validation set
testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype = dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype = dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype = dtype_dict)

# re-create the 4 features as we did in #1

testing['sqft_living_sqrt'] = np.sqrt(testing['sqft_living'])
testing['sqft_lot_sqrt'] = np.sqrt(testing['sqft_lot'])
testing['bedrooms_square'] = np.power(testing['bedrooms'], 2)
testing['floors_square'] = np.power(testing['floors'], 2)

training['sqft_living_sqrt'] = np.sqrt(training['sqft_living'])
training['sqft_lot_sqrt'] = np.sqrt(training['sqft_lot'])
training['bedrooms_square'] = np.power(training['bedrooms'], 2)
training['floors_square'] = np.power(training['floors'], 2)

validation['sqft_living_sqrt'] = np.sqrt(validation['sqft_living'])
validation['sqft_lot_sqrt'] = np.sqrt(validation['sqft_lot'])
validation['bedrooms_square'] = np.power(validation['bedrooms'], 2)
validation['floors_square'] = np.power(validation['floors'], 2)

#5 Initialize a list of L1 penalty
l1_penalty = np.logspace(1, 7, num = 13)

coef_dict = {}
for the_alpha in l1_penalty:
    model = linear_model.Lasso(alpha = the_alpha, normalize = True)
    model.fit(training[all_features], training['price'])
    coef_dict[the_alpha] = model.coef_

# np.sort(list(coef_dict.keys())) == np.sort(l1_penalty)
coef_dict_keys = np.sort(list(coef_dict.keys()))

for key in coef_dict_keys:
    print(coef_dict[key])

rss_dict = {}
for key in coef_dict_keys:
    coeff = coef_dict[key]
    res = np.mat(validation['price']).T - np.mat(validation[all_features]) * np.mat(coeff).T
    rss_dict[key] = float(res.T * res)

for key in coef_dict_keys:
    print(key, rss_dict[key])

#6 best value
# the best value for l1 penalty is 1000.0 because it gives the smallest RSS on validation data

#7 Compute the RSS on the TEST data with the l1_penalty = 1000.0

res_test_1000 = np.mat(testing['price']).T - np.mat(testing[all_features]) * np.mat(coef_dict[1000.0]).T
rss_test_1000 = res_test_1000.T * res_test_1000

# key_list = []
# value_list = []

# for key in list(rss_dict.keys()):
#    key_list.append(key)
#    value_list.append(rss_dict[key])
# series_of_rss = pd.Series(value_list, index = key_list)
# series_of_rss.sort_index().plot()

#8
model_best = linear_model.Lasso(alpha = 1000.0, normalize = True)
model_best.fit(training[all_features], training['price'])
np.count_nonzero(model_best.coef_) + np.count_nonzero(model.intercept_)

#10
max_nonzeros = 7
l1_penalty = np.logspace(1, 4, num = 20)

coef_dict = {}
for the_alpha in l1_penalty:
    model = linear_model.Lasso(alpha = the_alpha, normalize = True)
    model.fit(training[all_features], training['price'])
    coef_dict[the_alpha] = model.coef_

for key in coef_dict:
    print('Penalty: %d'%key, 'Nonzero features: %d' %(np.count_nonzero(coef_dict[key])+1))

l1_penalty_min = 0
l1_penalty_max = max(coef_dict.keys())
for key in coef_dict.keys():
    if np.count_nonzero(coef_dict[key]) + 1 > max_nonzeros and key > l1_penalty_min:
        l1_penalty_min = key
    elif np.count_nonzero(coef_dict[key]) + 1 < max_nonzeros and key < l1_penalty_max:
        l1_penalty_max = key

#14
l1_penalty = np.linspace(l1_penalty_min, l1_penalty_max, 20)
coef_dict = {}
for the_alpha in l1_penalty:
    model = linear_model.Lasso(alpha = the_alpha, normalize = True)
    model.fit(training[all_features], training['price'])
    coef_dict[the_alpha] = model.coef_

rss_dict = {}
for key in coef_dict.keys():
    coeff = coef_dict[key]
    res = np.mat(validation['price']).T - np.mat(validation[all_features]) * np.mat(coeff).T
    rss_dict[key] = float(res.T * res)

list_of_keys = []
for key in coef_dict.keys():
    if np.count_nonzero(coef_dict[key]) +1 == max_nonzeros:
        list_of_keys.append(key)
list_of_keys

for key in list_of_keys:
    print(key, rss_dict[key])

#so the l1 penalty that gives the smallest RSS on validation data is list_of_keys[1]
list_of_keys[1]

#16
model = linear_model.Lasso(alpha = list_of_keys[1], normalize = True)
model.fit(training[all_features], training['price'])
np.array(all_features)[model.coef_ != 0]










