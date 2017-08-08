import numpy as np
import pandas as pd
import statsmodels.api as st
import matplotlib.pyplot as plt
import os
os.chdir('/Users/wangqi/Documents/PyCharm/ML_Regression/Week4')

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)
train_set = pd.read_csv('kc_house_train_data.csv', dtype = dtype_dict)
test_set = pd.read_csv('kc_house_test_data.csv', dtype = dtype_dict)
sales = sales.sort_values(by = ['sqft_living','price'])

def get_numpy_data(data_set, features, output):
    feature_matrix = np.mat(data_set[features])
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    output_array = np.mat(data_set[output])

    return (feature_matrix, output_array)

def predict_output(feature_matrix, weights):
    predictions = feature_matrix * weights
    return predictions

def Cost(weights, feature_matrix, output, l2_penalty):
    rss = (predict_output(feature_matrix, weights) - output).T * predict_output(feature_matrix, weights) - output
    regularization = l2_penalty * weights.T * weights
    return  rss + regularization

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if feature_is_constant:
        derivative = 2 * np.sum(np.dot(feature, errors))
    else:
        derivative = 2 * np.sum(np.dot(feature, errors)) + 2 * l2_penalty * weight

    return derivative

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
example_output = example_output.T
my_weights = np.mat([1., 10.]).T
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

print(feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
print(np.sum(errors.T*example_features[:,1])*2+20.)
print('')

print(feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
print(np.sum(errors)*2.)

#8

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations = 100):
    weights = np.mat(initial_weights).T
    number_of_iterations = 0
    while number_of_iterations < max_iterations:
        predicted_output = predict_output(feature_matrix, weights)
        errors = predicted_output - output

        for i in np.arange(len(weights)):
            if i == 0:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, 0], weights[0], l2_penalty, True)
            else:
                derivative = feature_derivative_ridge(errors, feature_matrix[:, 0], weights[i], l2_penalty, False)
                weights[i] = weights[i] - step_size * derivative

        number_of_iterations += 1
    return the_weights

#11
simple_features = ['sqft_living']
my_output = ['price']
simple_feature_matrix, output = get_numpy_data(train_set, simple_features, my_output)
simple_test_feature_matrix, test_output = get_numpy_data(test_set, simple_features, my_output)

#12
step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros(2)
l2_penalty_0 = 0.0

a = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty_0, max_iterations)
#13
step_size = 1e-12
max_iterations = 1000
initial_weights = np.zeros(2)
l2_penalty_1 = 1e11
b = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, l2_penalty_1, max_iterations)

#14
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, a),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, b),'r-')

#17
Y = np.mat(test_set['price']).T
X_test = test_set['sqft_living']
X_test = st.add_constant(X_test)
X_test = np.mat(X_test)
initial_weights = np.zeros(2)
res_0 = Y - X_test * np.mat(initial_weights).T
rss_0 = res_0.T * res_0

res_1 = Y - X_test * a
rss_1 = res_1.T * res_1

res_2 = Y - X_test * b
rss_2 = res_2.T * res_2

#19
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_set, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_set, model_features, my_output)

#20
initial_weights = np.zeros(3)
step_size = 1e-12
max_iterations = 1000
l2 = 0

c = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2, max_iterations)


st.OLS(output.T, feature_matrix).fit().params

feature_matrix.shape
output.shape