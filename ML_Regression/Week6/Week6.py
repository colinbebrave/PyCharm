import numpy as np
import pandas as pd
import os
os.chdir('/Users/wangqi/Documents/PyCharm/ML_Regression/Week6')

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

train = pd.read_csv('kc_house_data_small_train.csv', dtype = dtype_dict)
test = pd.read_csv('kc_house_data_small_test.csv', dtype = dtype_dict)
valid = pd.read_csv('kc_house_data_validation.csv', dtype = dtype_dict)

def get_numpy_data(data_set, features, output):
    feature_matrix = np.mat(data_set[features])
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    output_array = np.mat(data_set[output])

    return (feature_matrix, output_array)

def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis = 0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition',
                'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, ['price'])
features_test, output_test = get_numpy_data(test, feature_list, ['price'])
features_valid, output_valid = get_numpy_data(valid, feature_list, ['price'])

features_train, norms = normalize_features(features_train)
features_test = features_test / norms
features_valid = features_valid / norms

#7
print(features_test[0])
print(features_train[9])

the_distance = np.sqrt((features_test[0] - features_train[9]) * (features_test[0] - features_train[9]).T)
"""the distance is 0.05972359"""

#9
query_house = features_test[0]
the_distance_10 = np.sqrt(np.sum(np.multiply(query_house - features_train[:10], query_house - features_train[:10]), axis = 1))
print(np.argmin(the_distance_10)) # represents the house that looks most like the query house.
"""so, the 9th house is the closest to query house"""

#11
for i in range(3):
    print(features_train[i] - features_test[0])

# verify that vectorization works
results = features_train[0:3] - features_test[0]
print(results[0] - (features_train[0] - features_test[0]))
# should print all 0's if results[0] == (features_train[0]-features_test[0])
print(results[1] - (features_train[1] - features_test[0]))
# should print all 0's if results[1] == (features_train[1]-features_test[0])
print(results[2] - (features_train[2] - features_test[0]))
# should print all 0's if results[2] == (features_train[2]-features_test[0])

#12
diff_1 = features_train - query_house

#13
diff = np.sqrt(np.sum(np.multiply(query_house - features_train[:], query_house - features_train[:]), axis = 1))
#14
diff[100]

#15
def compute_distances(features_instances, features_query):
    dist = np.sqrt(np.sum(np.multiply(features_query - features_instances[:], features_query - features_instances[:]), axis = 1))
    return dist

#16
np.argmin(compute_distances(features_train, features_test[2]))
"""the 383th house is closest to the features_test[2]"""

#17
y_hat = output_train[382]; print(float(y_hat))
"""the predicted price is 249000.0"""

#18
def k_nearest_neighbors(k, feature_train, features_query):
    dist2knn = compute_distances(feature_train[:k], features_query)
    the_dist_list = []
    for i in range(dist2knn.shape[0]):
        the_dist_list.append(float(dist2knn[i]))
    y = np.argsort(the_dist_list)
    x = np.sort(the_dist_list)
    for i in range(k, len(feature_train)):
        the_dist = compute_distances(features_train[i], features_query)
        the_dist = float(the_dist)
        if the_dist < x[k -1]:
            for j in range(k-1):
                if (x[j-1] < the_dist) & (x[j] > the_dist):
                    y[j+1:] = y[j:k-1]
                    x[j+1:] = x[j:k-1]
                    x[j] = the_dist
                    y[j] = i

    return y

k = 4
features_query = features_test[2]
feature_train = features_train
k_nearest_neighbors(k, feature_train, features_query)



compute_distances(features_train[[3, 2298, 3005, 527]], features_query)
compute_distances(features_train[[4, 2299, 3006, 528]], features_query)

#20
def predict_output_of_query(k, features_train, output_train, features_query):
    indexes = k_nearest_neighbors(k, features_train, features_query)
    the_output = output_train[indexes]
    prediction = np.mean(the_output)
    return prediction

predict_output_of_query(4, features_train, output_train, features_test[2])
"""the predicted value for 3rd house in test set is 667250"""
#22
def predict_output(k, features_train, output_train, features_query):
    prediction_list = []
    for i in range(features_query.shape[0]):
        prediction_list.append(predict_output_of_query(k, features_train, output_train, features_query[i]))
    return prediction_list

kk = 10
ii = predict_output(kk, features_train, output_train, features_test[:10])
jj = np.argmin(ii)
ii[jj]
"""the 3rd house has the lowest predicted value which is 335500"""
#24
rss_list = []
for k in range(1, 16):
    the_prediction = predict_output(k, features_train, output_train, features_valid)
    res = np.mat(the_prediction).T - output_valid
    rss = res.T * res
    rss_list.append(rss)

the_best_k = np.argmin(rss_list) + 1

#25
prediction_on_test_set = predict_output(the_best_k, features_train, output_train, features_test)
res_on_test_set = prediction_on_test_set - output_test
rss_on_test_set = res_on_test_set.T * res_on_test_set
