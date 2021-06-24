import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pickle
from sklearn import metrics

pd.set_option('max_row', None)

car_data=pd.read_csv('car data.csv')
print(car_data.head())

#Shape
print(car_data.shape)

#Categorical data
print(car_data['Seller_Type'].unique())
print(car_data['Transmission'].unique())
print(car_data['Fuel_Type'].unique())

#Missing values
print(car_data.isnull().sum())

#Stats description
#print(car_data.describe())

#All columns 
#print(car_data.columns)

#creating new datafrae as not all columns are important 
final_car_data=car_data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

#Get how many years the car is old 
final_car_data['Current_Year']=2021

final_car_data['Years_Used']=final_car_data['Current_Year']-final_car_data['Year']

#Droping unnecessary columns
final_car_data.drop(['Year'],axis=1,inplace=True)
final_car_data.drop(['Current_Year'],axis=1,inplace=True)


#Take care of Categorical variables 
final_car_data=pd.get_dummies(final_car_data,drop_first=True)

#print(final_car_data.columns)

#Correlation 
#print(final_car_data.corr())

#sns.pairplot(final_car_data)
#plt.show()

# correlation_matrix=final_car_data.corr()
# top_corr_features=correlation_matrix.index
# plt.figure(figsize=(20,20))
# sns.heatmap(final_car_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.show()

#print(final_car_data.head())

#Independent and Dependent features
X=final_car_data.iloc[:,1:]
Y=final_car_data.iloc[:,0]

print(X.head())
print(Y.head())

# Feature Importance 
model=ExtraTreesRegressor()
model.fit(X,Y)

print(model.feature_importances_)

#Visualize Feature Importances 
feat_importances=pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(9).plot(kind='barh')
plt.show()

#Train and test split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


# Random Forest
rf_model=RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 5, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)

print(rf_random.best_params_)
predictions=rf_random.predict(X_test)
sns.distplot(y_test-predictions)
plt.show()

#METRICS
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)








