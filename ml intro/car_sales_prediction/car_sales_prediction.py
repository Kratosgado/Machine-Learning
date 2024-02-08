# %% [markdown]
# The car dealership industry in Ghana keeps growing competitive as more and more dealers enter the car dealership business. The competition is worsening the sales performance of many small and mid-size dealerships including Frankot Motors, a car dealeship operating in Ghana. As a management consultant, Frankot Motors has approached us for an insight on how best to price its cars to maximize revenues and profits. As an enthusiast of Python data analytics, we are required to perform exploratory data analysis witht he aim of extracting the features which impact car sales the most and build a predictive model to intelligently guess the sales price of a car based on those important features.
# 

# %%
import pandas as pd;

car_sales = pd.read_csv("car_sales.csv")
car_sales.head() # display first 5 rows

# %%
car_sales.info() # give a description about the data

# %%
car_sales["price"].value_counts() # view more info about the price

# %% [markdown]
# convert object datas into numerical
# 

# %%
numerical_cols = ['price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']
for col in numerical_cols:
   car_sales[col] = pd.to_numeric(car_sales[col], errors='coerce')

# %%
car_sales.describe() # give a description about the data

# %%
car_sales.info()

# %% [markdown]
# ## Clean Data
# 

# %%
car_sales.isnull().sum() # check for missing values

# %%
# create a copy of the data without the text attributes
for col in numerical_cols:
    car_sales[col].fillna(car_sales[col].median(), inplace=True)

# %%
car_sales.isnull().sum()

# %% [markdown]
# ## Handling Text Categorical Attributes
# 

# %%
car_sales_cat = car_sales[['manufact', 'model']]
car_sales_cat.head(10)

# %%
# convert the categorical data to numbers
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
car_sales_cat_1hot = cat_encoder.fit_transform(car_sales_cat)
car_sales_cat_1hot.toarray() # converting it to dense numpy array

# %%
# get list of categories
cat_encoder.categories_

# %% [markdown]
# plot the data to have a visual presentation
# 

# %%
# %matplotlib inline
import matplotlib.pyplot as plt 

car_sales.hist(bins=50, figsize=(20,15))
plt.show()

# %%
# create a scatter plot
car_sales.plot(kind="scatter", x="horsepow", y="price", alpha=0.5)

# %%
car_sales_num = pd.get_dummies(car_sales, columns=["manufact", "model"])
corr_matrix = car_sales_num.corr()
corr_matrix["price"].sort_values(ascending=False)

# %%
# pandas to analyze correlation
from pandas.plotting import scatter_matrix

num_attribs = [ "sales", "price", "engine_s", "horsepow", "wheelbas", "width", "length", "curb_wgt", "fuel_cap", "mpg"]
scatter_matrix(car_sales[num_attribs], figsize=(12, 8))

# %% [markdown]
# ### Create a Test Set
# 

# %%
import numpy as np
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(car_sales, test_size=0.2, random_state=42)
car_sales = train_set.drop("price", axis=1)
car_sales_labels = train_set["price"].copy()

# %%
len(train_set)

# %%
# pipeline for data transformation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
      ('imputer', SimpleImputer(strategy='median')),
      ('std_scaler', StandardScaler())
   ])
car_sales_num_tr = num_pipeline.fit_transform(car_sales_num)

# %%
# using sklearn-learn ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attribs = [ "sales", "engine_s", "horsepow", "wheelbas", "width", "length", "curb_wgt", "fuel_cap", "mpg"]
cat_attribs = ['manufact', 'model']

full_pipeline = ColumnTransformer([
      ('num', num_pipeline, num_attribs),
      ('cat', OneHotEncoder(handle_unknown="ignore"), cat_attribs)
   ])

car_sales_prepared = full_pipeline.fit_transform(car_sales)

# %% [markdown]
# # Select and Train a Model
# 

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(car_sales_prepared, car_sales_labels)

# %%
# test the model
some_data = car_sales.iloc[:5]
some_labels = car_sales_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))

# %%
print("Labels:", list(some_labels))

# %% [markdown]
# ### Measure model's mean square error
# 

# %%
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(car_sales_prepared)
lin_mse = mean_squared_error(car_sales_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# %% [markdown]
# ## Use Decision Tree Regressor
# 

# %%
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(car_sales_prepared, car_sales_labels)

# %%
# measure the model
car_sales_predictions = tree_reg.predict(car_sales_prepared)
tree_mse = mean_squared_error(car_sales_labels, car_sales_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

# %% [markdown]
# ## Better Evaluation Using Cross Validation
# 

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, car_sales_prepared, car_sales_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

# %% [markdown]
# 

# %%
# check the scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

# %%
# computet the same scores for the linear regression model
lin_scores = cross_val_score(lin_reg, car_sales_prepared, car_sales_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# %% [markdown]
# ## Fine-Tune the model using grid search
# 

# %%
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 3, 4]}
]
tree_reg_grid = DecisionTreeRegressor()
grid_search = GridSearchCV(tree_reg_grid, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(car_sales_prepared, car_sales_labels)

# %%
# display the best hyperparameters
grid_search.best_params_

# %%
# get best estimator
grid_search.best_estimator_

# %%
# evaluate scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# %% [markdown]
# ## Analyze the best models and their errors
# 

# %%
# display importance scores next to their attribute names
feature_importances = grid_search.best_estimator_.feature_importances_
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[ 1])
attributes = num_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)

# %% [markdown]
# ## Evaluate System on the Test Set
# 

# %%
final_model = grid_search.best_estimator_

x_test = test_set.drop("price", axis=1)
y_test = test_set["price"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse

# %%



