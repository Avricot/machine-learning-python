import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import datasets, linear_model
import math
import seaborn
from matplotlib import pyplot as plt
from pandas.tools.plotting import scatter_matrix

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

print(diabetes_X_train)
print(diabetes_y_train)

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)



data = pd.read_csv('../home_data.csv')
test = data[data['zipcode'] == 98039]
test['price'].mean()

square = data[(data['sqft_living'] >=2000) & (data['sqft_living'] <=4000) ]

print(square.size / data.size)

data_X = data['sqft_living']

train = data[:int(data.shape[0]*0.8)]
test = data[-int(data.shape[0]*0.2):]
#ou split avec du random
#X_train, X_test, y_train, y_test = train_test_split(data['sqft_living'], data['price'], test_size=0.2, random_state=0)

#Uniquement avec les sqft_living
regr = linear_model.LinearRegression()
regr.fit(train.as_matrix(['sqft_living']), train['price'].values)
print("Residual sum of squares: %.2f" % math.sqrt(np.mean((regr.predict(test.as_matrix(['sqft_living'])) - test['price'].values) ** 2)))

#Avec plein de features
#my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
regr2 = linear_model.LinearRegression()
print(train.as_matrix(my_features))
regr2.fit(train[my_features], train['price'].values)
print("Residual sum of squares: %.2f" % math.sqrt(np.mean((regr2.predict(test[my_features]) - test['price'].values) ** 2)))


#Let's do a simple histogram on available house prices to see what we'll be working on
plt.interactive(False)
plt.hist(test['price'],bins=20)
plt.suptitle('Boston Housing Prices in $1000s', fontsize=15)
plt.xlabel('Prices in $1000s')
plt.ylabel('Count')
plt.savefig('price.png')
plt.show(block=True)

with seaborn.axes_style('white'):
    smaller_frame = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode', 'price']]
    scatter_matrix(smaller_frame, alpha=0.8, figsize=(12, 12), diagonal="kde")
plt.savefig('features.png')

print("Residual sum of squares: %.2f" % math.sqrt(np.mean((regr2.predict(test[my_features]) - test['price'].values) ** 2)))
