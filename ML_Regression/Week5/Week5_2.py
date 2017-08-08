import numpy as np
import pandas as pd
import os
os.chdir('/Users/wangqi/Documents/PyCharm/ML_Regression/Week5')

sales = pd.read_csv('kc_house_data.csv')
train = pd.read_csv('kc_house_train_data.csv')
test = pd.read_csv('kc_house_test_data.csv')

#3
def get_numpy_data(data_set, features, output):
    feature_matrix = np.mat(data_set[features])
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    output_array = np.mat(data_set[output])

    return (feature_matrix, output_array)
#4
def predict_output(feature_matrix, weights):
    predictions = feature_matrix * weights.T
    return predictions
#5
"""Features should be normalized because the small weights will be dropped first
as l1_penalty goes up even if the corresponding feature might be very predictive
"""

#6
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis = 0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

# features_of_interest = ['sqft_living', 'bedrooms']
# output_of_interest = 'price'
# n_train, norms = normalize_features(train[features_of_interest])

#9
features_of_interest = ['sqft_living', 'bedrooms']
output_of_interest = 'price'

feature_matrix, output = get_numpy_data(sales, features_of_interest, output_of_interest)
n_feature_matrix, norms = normalize_features(feature_matrix)

initial_weights = np.mat([1, 4, 1]).T

prediction = predict_output(n_feature_matrix, initial_weights)

ro = []
for i in range(n_feature_matrix.shape[1]):
    ro.append(float(n_feature_matrix[:, i].T * (output.T - prediction + float(initial_weights[i]) * n_feature_matrix[:, i])))

#10
# the range might be [-8100000, 8100000]

#11
# the range should include the ro[1] and ro[2], so it should be wider than max(ro[1], ro[2])

#12
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    #weights = np.mat(weights).T
    output = np.mat(output).T
    feature_matrix = np.mat(feature_matrix)
    prediction = predict_output(feature_matrix, weights)
    ro_i = feature_matrix[:, i].T * (output - prediction + float(weights[i]) * feature_matrix[:, i])
    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -(l1_penalty / 2):
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > (l1_penalty / 2):
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.
    return new_weight_i

# to verify this function by the following
import math
print(lasso_coordinate_descent_step(1, np.mat(np.array([[3./math.sqrt(13),1./math.sqrt(10)],
                   [2./math.sqrt(13),3./math.sqrt(10)]])), np.mat(np.array([1., 1.])).T, np.array([1., 1.]), 0.1))

#13
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    the_initial_weights = np.mat(initial_weights).T
    initial_weights = np.mat(initial_weights).T
    while True:
        for i in range(len(initial_weights)):
            initial_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty)
        diff = np.abs(the_initial_weights - initial_weights)
        if np.max(diff) < tolerance:
            break
    return initial_weights

initial_weights = np.zeros(feature_matrix.shape[1])
l1_penalty = 1e7
tolerance = 1.0

l1_penalty = 2976000000
ww = lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance)

